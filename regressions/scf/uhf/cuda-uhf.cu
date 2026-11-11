/// @file regressions/scf/cuda-uhf/reg_cuda-uhf.cu
/// @brief O2 and OH STO-3G UHF on CPU and CUDA against the same PySCF pins.
///
/// `--device=cpu|cuda|both` (default both). The CUDA leg must report
/// last_scf_on_device and last_force_on_device. See REFERENCE.md.

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include <cstring>
#include <stdexcept>
#include <string>

#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cuda/scf/uhf.cuh"
#include "cuda-uhf.h"
#include "harness/registry.h"

namespace atv_reg_scf_cuda_uhf {

namespace {

__global__ void seed_force_sentinel(ATV::cuda::Frame frame, int n) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    ATV::cuda::store_force(frame.atoms.fx, i, 0.125, frame.force_fault);
    ATV::cuda::store_force(frame.atoms.fy, i, -0.25, frame.force_fault);
    ATV::cuda::store_force(frame.atoms.fz, i, 0.5, frame.force_fault);
}

// Deliberately change device outputs before throwing: a failed run must
// release its backend without publishing any of these candidate values.
__global__ void corrupt_failed_candidate(ATV::cuda::Frame frame) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        frame.atoms.x[0] += 3;
        frame.scalars->energy = 9001;
        ATV::cuda::store_force(frame.atoms.fx, 0, 987.0, frame.force_fault);
    }
}

class EnergyRequest : public ATV::cuda::Component {
  public:
    std::string name() const override { return "energy_request_probe"; }
    void launch(const ATV::cuda::LaunchContext& ctx) override {
        ctx.runtime.exec.forces_requested = false;
        seed_force_sentinel<<<(ctx.frame.n_atoms() + 127) / 128, 128, 0,
                              static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
            ctx.frame, ctx.frame.n_atoms());
        ATV_CUDA_CHECK(cudaGetLastError());
    }
};

class ThrowOnce : public ATV::cuda::Component {
  public:
    bool fail = false;
    int every = 1;
    int builds = 0;
    int launches = 0;
    int destroys = 0;
    bool live = false;
    std::string name() const override { return "scf_failure_probe"; }
    int frequency() const override { return every; }
    void build(const ATV::cuda::BuildContext&) override {
        if (live)
            throw std::runtime_error(
                "UHF probe duplicate build got=1 want=0 tol=0");
        live = true;
        ++builds;
    }
    void launch(const ATV::cuda::LaunchContext& ctx) override {
        if (!live || ctx.step % every != 0)
            throw std::runtime_error("UHF probe invalid launch "
                                     "lifecycle/frequency got=1 want=0 tol=0");
        ++launches;
        if (fail) {
            fail = false;
            corrupt_failed_candidate<<<
                1, 1, 0, static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
                ctx.frame);
            ATV_CUDA_CHECK(cudaGetLastError());
            throw std::runtime_error("injected SCF check failure");
        }
    }
    void destroy(const ATV::cuda::DestroyContext&) override {
        if (!live)
            throw std::runtime_error(
                "UHF probe duplicate destroy got=1 want=0 tol=0");
        live = false;
        ++destroys;
    }
    void require_balanced(int want_builds) const {
        if (live || builds != want_builds || destroys != want_builds)
            throw std::runtime_error(
                "UHF probe lifecycle got build=" + std::to_string(builds)
                + " destroy=" + std::to_string(destroys)
                + " live=" + std::to_string(live) + " want build=destroy="
                + std::to_string(want_builds) + " live=0 tol=0");
    }
};

bool sentinel_preserved(const char* method, const ATV::cpu::Frame& frame) {
    using namespace ATV;
    const auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy),
               fz = frame.span(kAtom_fz);
    double want[3] = {0.125, -0.25, 0.5};
    // Compare exactly with the public fixed-point transport representation.
    for (double& value : want) {
        cuda::fixed_t encoded = 0;
        if (!cuda::encode_force(value, encoded))
            throw std::runtime_error("Invalid force sentinel");
        value = cuda::decode_force<double>(encoded);
    }
    for (int i = 0; i < frame.nlocal(); ++i) {
        const double got[3] = {fx[i], fy[i], fz[i]};
        for (int axis = 0; axis < 3; ++axis)
            if (got[axis] != want[axis]) {
                std::printf("FAIL %s parent force veto atom=%d axis=%d "
                            "got=%.16g want=%.16g tol=0\n",
                            method, i, axis, got[axis], want[axis]);
                return false;
            }
    }
    return true;
}

std::string parse_device(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0)
            want = a + 9;
        else if (std::strcmp(a, "--device") == 0 && i + 1 < argc)
            want = argv[++i];
        else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                 || std::strcmp(a, "both") == 0)
            want = a;
    }
    return want;
}

cuda_uhf_case::Out run_cpu(ATV::cpu::Frame& frame, int na, int nb, int charge) {
    ATV::cpu::scf::UHF uhf{
        ATV::cpu::scf::BasisSet::STO3G, na, nb, charge, 1e-12, 100};
    uhf.init_guess = ATV::cpu::scf::InitGuess::MINAO;
    uhf.damping_factor = 0.0;
    uhf.diis_start_iter = 1;
    uhf.diis_max_vectors = 8;
    const auto g = uhf.forces(frame);
    cuda_uhf_case::Out out;
    out.energy_ha = uhf.energy();
    out.converged = uhf.converged();
    out.forces = g.forces;
    return out;
}

cuda_uhf_case::Out run_cuda(ATV::cpu::Frame& frame, int na, int nb,
                            int charge) {
    using namespace ATV;
    ThrowOnce failure;
    ThrowOnce periodic;
    periodic.every = 3;
    cuda::scf::UHF uhf{cpu::scf::BasisSet::STO3G, na, nb, charge, 1e-12, 100};
    uhf.on(scf::Phase::check).add_ref(failure, -1000);
    uhf.on(scf::Phase::check).add_ref(periodic, 1000);
    uhf.build();
    int expected_builds = 0;
    auto nested_energy = [&] {
        const int previous_launches = periodic.launches;
        ++expected_builds;
        cuda::Evaluate eval;
        eval.on(evaluate::Phase::prepare).add(EnergyRequest{}, 0);
        eval.on(evaluate::Phase::evaluate).add_ref(uhf);
        eval.build();
        eval.run(frame);
        failure.require_balanced(expected_builds);
        periodic.require_balanced(expected_builds);
        const int want_launches =
            (uhf.iterations() + periodic.every - 1) / periodic.every;
        if (periodic.launches - previous_launches != want_launches)
            throw std::runtime_error(
                "UHF periodic launch count got="
                + std::to_string(periodic.launches - previous_launches)
                + " want=" + std::to_string(want_launches) + " tol=0");
    };
    nested_energy();
    const double nested_e = frame.meta.get(kEnergy) / units::kHartreeInEv;
    bool nested_cache_rejected = false;
    try {
        (void)uhf.energy();
    } catch (const std::logic_error&) {
        nested_cache_rejected = true;
    }
    if (!nested_cache_rejected)
        throw std::runtime_error(
            "UHF nested energy mirror got=available want=unavailable tol=0");
    if (uhf.last_force_on_device() || !sentinel_preserved("UHF", frame)) {
        std::printf("FAIL UHF parent force veto flag got=%d want=0 tol=0\n",
                    uhf.last_force_on_device());
        return {};
    }
    // Identity parameters belong to the method. A same-size nuclear edit that
    // violates its electron count must fail, then permit a corrected retry.
    auto types = frame.span(kAtom_atomic_number);
    const int original_z = types[1];
    types[1] = original_z == 1 ? 8 : 1;
    bool identity_rejected = false;
    try {
        nested_energy();
    } catch (const std::invalid_argument& e) {
        identity_rejected =
            std::string(e.what()).find("n_alpha + n_beta must equal")
            != std::string::npos;
    }
    types[1] = original_z;
    failure.require_balanced(expected_builds);
    periodic.require_balanced(expected_builds);
    if (!identity_rejected) {
        std::printf("FAIL UHF changed identity rejection got=0 want=1 tol=0\n");
        return {};
    }
    nested_energy();
    failure.fail = true;
    bool rejected = false;
    try {
        nested_energy();
    } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()) == "injected SCF check failure";
    }
    failure.require_balanced(expected_builds);
    periodic.require_balanced(expected_builds);
    if (!rejected) {
        std::printf("FAIL UHF injected failure got=0 want=1 tol=0\n");
        return {};
    }
    nested_energy();
    // The standalone failure path previously called teardown_pass, which
    // retrieved incomplete device science into the caller's Frame.
    const double accepted_x = frame.span(kAtom_x)[0];
    const double accepted_fx = frame.span(kAtom_fx)[0];
    const double accepted_energy = frame.meta.get(kEnergy);
    failure.fail = true;
    bool standalone_rejected = false;
    ++expected_builds;
    try {
        uhf.run(frame);
    } catch (const std::runtime_error& e) {
        standalone_rejected =
            std::string(e.what()) == "injected SCF check failure";
    }
    failure.require_balanced(expected_builds);
    periodic.require_balanced(expected_builds);
    if (!standalone_rejected || frame.span(kAtom_x)[0] != accepted_x
        || frame.span(kAtom_fx)[0] != accepted_fx
        || frame.meta.get(kEnergy) != accepted_energy) {
        std::printf(
            "FAIL UHF failed standalone publication got=%d want=1 tol=0\n",
            standalone_rejected);
        return {};
    }
    // Reuse the same method after failure; the numerical oracle below still
    // checks the physical result of this successful retry.
    nested_energy();
    if (!sentinel_preserved("UHF retry", frame))
        return {};
    ++expected_builds;
    const int before_forces_launches = periodic.launches;
    const auto g = uhf.forces(frame);
    failure.require_balanced(expected_builds);
    periodic.require_balanced(expected_builds);
    const int want_force_launches =
        (uhf.iterations() + periodic.every - 1) / periodic.every;
    if (periodic.launches - before_forces_launches != want_force_launches)
        throw std::runtime_error(
            "UHF standalone frequency count got="
            + std::to_string(periodic.launches - before_forces_launches)
            + " want=" + std::to_string(want_force_launches) + " tol=0");
    if (!std::isfinite(nested_e) || std::abs(nested_e - uhf.energy()) > 1e-10) {
        std::printf("FAIL UHF same-N nested energy got=%.16g want=%.16g "
                    "tol=1e-10 Ha\n",
                    nested_e, uhf.energy());
        return {};
    }
    if (!uhf.last_scf_on_device() || !uhf.last_force_on_device()) {
        std::printf("FAIL: CUDA UHF left the device path "
                    "(scf=%d force=%d)\n",
                    uhf.last_scf_on_device() ? 1 : 0,
                    uhf.last_force_on_device() ? 1 : 0);
        cuda_uhf_case::Out bad;
        bad.converged = false;
        return bad;
    }
    cuda_uhf_case::Out out;
    out.energy_ha = uhf.energy();
    out.converged = uhf.converged();
    out.forces = g.forces;
    return out;
}

} // namespace

int run(int argc, char** argv) {
    const std::string want = parse_device(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::printf("FAIL: unknown --device=%s\n", want.c_str());
        return 1;
    }
    int rc = 0;
    if (want == "cpu" || want == "both")
        rc |= cuda_uhf_case::run_all(run_cpu);
    if (want == "cuda" || want == "both")
        rc |= cuda_uhf_case::run_all(run_cuda);
    return rc;
}

} // namespace atv_reg_scf_cuda_uhf

ATV_REGRESSION("scf/uhf", "cuda-uhf", atv_reg_scf_cuda_uhf::run);
