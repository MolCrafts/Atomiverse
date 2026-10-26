/// @file regressions/scf/es-cuda-rhf-driver/reg_es-cuda-rhf-driver.cu
/// @brief Staged RHF driver vs PySCF — one regression, multi-device.
///
/// `--device=cpu|cuda|both` (default both). CPU grades H2/STO-3G and
/// H2O/6-31G. CUDA grades only H2/STO-3G and requires the device SCF and
/// force flags. H2O/6-31G is not a device ERI kernel; the host-fill path
/// is not a pass. See REFERENCE.md.

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cuda/scf/rhf.cuh"
#include "harness/registry.h"

namespace atv_reg_scf_es_cuda_rhf_driver {

namespace {

__global__ void seed_force_sentinel(ATV::cuda::Frame frame, int n) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    ATV::cuda::store_force(frame.atoms.fx, i, 0.125, frame.force_fault);
    ATV::cuda::store_force(frame.atoms.fy, i, -0.25, frame.force_fault);
    ATV::cuda::store_force(frame.atoms.fz, i, 0.5, frame.force_fault);
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
    std::string name() const override { return "scf_failure_probe"; }
    void launch(const ATV::cuda::LaunchContext&) override {
        if (fail) {
            fail = false;
            throw std::runtime_error("injected SCF check failure");
        }
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

enum class Device { Cpu, Cuda };

const char* device_name(Device d) { return d == Device::Cuda ? "cuda" : "cpu"; }

std::string parse_device_arg(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0) {
            want = a + 9;
        } else if (std::strcmp(a, "--device") == 0 && i + 1 < argc) {
            want = argv[++i];
        } else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                   || std::strcmp(a, "both") == 0) {
            want = a;
        }
    }
    return want;
}

ATV::cpu::Frame make_h2() {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

ATV::cpu::Frame make_h2o() {
    using namespace ATV;
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

int check_cpu(const char* label, ATV::cpu::Frame frame,
              ATV::cpu::scf::BasisSet basis, double e_want) {
    using namespace ATV;
    constexpr double kTol = 1e-8;
    cpu::scf::RHF rhf{basis, 0, 1e-12, 200};
    rhf.run(frame);
    const double e_ha = rhf.energy();
    const double meta_ha = frame.meta.get(kEnergy) / units::kHartreeInEv;
    if (!rhf.converged() || !std::isfinite(e_ha)
        || std::abs(e_ha - e_want) > kTol || !std::isfinite(meta_ha)
        || std::abs(meta_ha - e_want) > kTol) {
        std::fprintf(stderr,
                     "FAIL [cpu]: %s E=%.12f meta=%.12f want=%.12f "
                     "conv=%d tol=%.0e\n",
                     label, e_ha, meta_ha, e_want, rhf.converged() ? 1 : 0,
                     kTol);
        return 1;
    }
    std::printf("PASS [cpu]: %s E=%.12f Ha (want %.12f +/- %.0e) "
                "[PySCF 2.14.0]\n",
                label, e_ha, e_want, kTol);
    return 0;
}

int check_cuda_h2() {
    using namespace ATV;
    // Device energy band from scf/h2o_rhf_sto3g_grad (REFERENCE.md).
    constexpr double kTol = 1e-6;
    constexpr double kWant = -1.1167593073964255;
    cpu::Frame frame = make_h2();
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, 1e-12, 200};
    ThrowOnce failure;
    rhf.on(scf::Phase::check).add_ref(failure, -1000);
    rhf.build();
    auto nested_energy = [&] {
        cuda::Evaluate eval;
        eval.on(evaluate::Phase::prepare).add(EnergyRequest{}, 0);
        eval.on(evaluate::Phase::evaluate).add_ref(rhf);
        eval.build();
        eval.run(frame);
    };
    nested_energy();
    const double nested_energy_ha =
        frame.meta.get(kEnergy) / units::kHartreeInEv;
    bool nested_cache_rejected = false;
    try {
        (void)rhf.energy();
    } catch (const std::logic_error&) {
        nested_cache_rejected = true;
    }
    if (rhf.last_force_on_device() || !sentinel_preserved("RHF", frame)
        || !rhf.converged() || !nested_cache_rejected
        || std::abs(nested_energy_ha - kWant) > kTol) {
        std::printf("FAIL RHF energy-only got=%.12f want=%.12f tol=%.0e "
                    "force=%d want=0 converged=%d want=1\n",
                    nested_energy_ha, kWant, kTol, rhf.last_force_on_device(),
                    rhf.converged());
        return 1;
    }
    // Same N, different element: H-He has three electrons and must be
    // rejected, rather than silently reusing the cached H-H identity.
    frame.span(kAtom_type)[1] = 2;
    bool rejected = false;
    try {
        nested_energy();
    } catch (const std::invalid_argument& e) {
        rejected =
            std::string(e.what()).find("closed-shell") != std::string::npos;
    }
    if (!rejected) {
        std::printf("FAIL RHF same-N identity rejection got=0 want=1 tol=0\n");
        return 1;
    }
    frame.span(kAtom_type)[1] = 1;
    failure.fail = true;
    rejected = false;
    try {
        nested_energy();
    } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()) == "injected SCF check failure";
    }
    if (!rejected) {
        std::printf("FAIL RHF injected failure got=0 want=1 tol=0\n");
        return 1;
    }
    nested_energy();
    if (!sentinel_preserved("RHF retry", frame))
        return 1;
    rhf.run(frame);
    const double e_ha = rhf.energy();
    const double meta_ha = frame.meta.get(kEnergy) / units::kHartreeInEv;
    if (!rhf.last_scf_on_device() || !rhf.last_force_on_device()) {
        std::fprintf(stderr,
                     "FAIL [cuda]: H2/STO-3G host ERI path is not a device "
                     "result (scf_on_device=%d force_on_device=%d)\n",
                     rhf.last_scf_on_device() ? 1 : 0,
                     rhf.last_force_on_device() ? 1 : 0);
        return 1;
    }
    if (!rhf.converged() || !std::isfinite(e_ha)
        || std::abs(e_ha - kWant) > kTol || !std::isfinite(meta_ha)
        || std::abs(meta_ha - kWant) > kTol) {
        std::fprintf(stderr,
                     "FAIL [cuda]: H2/STO-3G E=%.12f meta=%.12f want=%.12f "
                     "conv=%d tol=%.0e\n",
                     e_ha, meta_ha, kWant, rhf.converged() ? 1 : 0, kTol);
        return 1;
    }
    std::printf("PASS [cuda]: H2/STO-3G E=%.12f Ha (want %.12f +/- %.0e) "
                "path=device [PySCF 2.14.0]\n",
                e_ha, kWant, kTol);

    // A read-only geometry solve must preserve every caller-visible field,
    // even when the method's default force kernel is unavailable. These are
    // complete SCF solves, so they belong with this product regression.
    for (const auto basis :
         {cpu::scf::BasisSet::STO3G, cpu::scf::BasisSet::Def2SVP}) {
        auto readonly_frame = make_h2();
        readonly_frame.meta.get(kEnergy) = 42.0;
        readonly_frame.meta.get(kEnergyElectronic) = -7.0;
        readonly_frame.span(kAtom_fx)[0] = 1.5;
        readonly_frame.span(kAtom_fy)[0] = -2.5;
        readonly_frame.span(kAtom_fz)[1] = 3.5;
        const double z1 = readonly_frame.span(kAtom_z)[1];
        cuda::scf::RHF readonly_rhf{basis, 0, 1e-10, 100};
        const cpu::Frame& geometry = readonly_frame;
        readonly_rhf.run(geometry);
        readonly_rhf.run(geometry);
        const bool preserved =
            readonly_frame.meta.get(kEnergy) == 42.0
            && readonly_frame.meta.get(kEnergyElectronic) == -7.0
            && readonly_frame.span(kAtom_fx)[0] == 1.5
            && readonly_frame.span(kAtom_fy)[0] == -2.5
            && readonly_frame.span(kAtom_fz)[1] == 3.5
            && readonly_frame.span(kAtom_z)[1] == z1;
        if (!preserved || !readonly_rhf.converged()
            || !readonly_rhf.last_scf_on_device()
            || readonly_rhf.last_force_on_device()
            || !std::isfinite(readonly_rhf.energy())
            || (basis == cpu::scf::BasisSet::STO3G
                && std::abs(readonly_rhf.energy() - kWant) > kTol)) {
            std::printf("FAIL [cuda]: const RHF preserved=%d want=1 "
                        "converged=%d want=1 force=%d want=0 "
                        "E=%.12f Ha (STO-3G want %.12f tol=%.0e)\n",
                        preserved, readonly_rhf.converged(),
                        readonly_rhf.last_force_on_device(),
                        readonly_rhf.energy(), kWant, kTol);
            return 1;
        }
    }
    return 0;
}

} // namespace

int run(int argc, char** argv) {
    static_assert(std::is_base_of_v<ATV::cuda::Driver<ATV::scf::Phase>,
                                    ATV::cuda::scf::RHF>,
                  "cuda::scf::RHF must be cuda::Driver<scf::Phase>");

    const std::string want = parse_device_arg(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::fprintf(stderr, "FAIL: unknown --device=%s (want cpu|cuda|both)\n",
                     want.c_str());
        return 1;
    }
    int rc = 0;
    if (want == "cpu" || want == "both") {
        rc |= check_cpu("H2/STO-3G", make_h2(), ATV::cpu::scf::BasisSet::STO3G,
                        -1.1167593073964255);
        rc |= check_cpu("H2O/6-31G", make_h2o(),
                        ATV::cpu::scf::BasisSet::Pople631G, -75.98397447272204);
    }
    if (want == "cuda" || want == "both") {
        // 6-31G stays on the CPU leg. Device kernels are STO-3G s/p only.
        std::printf("[cuda] H2O/6-31G skipped: not a device ERI kernel "
                    "(host-fill is not graded)\n");
        rc |= check_cuda_h2();
    }
    return rc;
}

} // namespace atv_reg_scf_es_cuda_rhf_driver

ATV_REGRESSION("scf/rhf", "es-cuda-rhf-driver",
               atv_reg_scf_es_cuda_rhf_driver::run);
