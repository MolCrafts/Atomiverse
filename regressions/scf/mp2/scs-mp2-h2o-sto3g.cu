/// @file regressions/scf/scs-mp2-h2o-sto3g/reg_scs-mp2-h2o-sto3g.cu
/// @brief H2O SCS-MP2/STO-3G — e_corr (and e_tot) vs offline PySCF pins.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Geometry and pins: REFERENCE.md (PySCF 2.14.0, Grimme c_OS=1.2, c_SS=1/3).
/// Never import/invoke PySCF from this binary.
///
/// Exit non-zero on failure. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/scf/mp2.cuh"
#include "atomiverse/cuda/scf/rhf.cuh"
#include "harness/registry.h"

namespace atv_reg_scf_scs_mp2_h2o_sto3g {

namespace {

ATV::cpu::Frame h2o_frame() {
    ATV::cpu::Frame frame(3);
    frame.span(ATV::kAtom_atomic_number)[0] = 8;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.1173;
    frame.span(ATV::kAtom_atomic_number)[1] = 1;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.7572;
    frame.span(ATV::kAtom_z)[1] = -0.4692;
    frame.span(ATV::kAtom_atomic_number)[2] = 1;
    frame.span(ATV::kAtom_x)[2] = 0.0;
    frame.span(ATV::kAtom_y)[2] = -0.7572;
    frame.span(ATV::kAtom_z)[2] = -0.4692;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

// The operation has one physical scalar, hence one thread. The known energy
// is accumulated on the same stream before the nested electronic solve.
__global__ void add_known_energy(ATV::SystemScalars* scalars,
                                 double energy_ev) {
    scalars->energy += energy_ev;
}

class KnownEnergy : public ATV::cuda::Component {
  public:
    explicit KnownEnergy(double energy_ev) : energy_ev_(energy_ev) {}
    std::string name() const override { return "mp2_known_energy"; }
    bool provides_energy() const override { return true; }
    void launch(const ATV::cuda::LaunchContext& ctx) override {
        add_known_energy<<<1, 1, 0,
                           static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
            ctx.frame.scalars, energy_ev_);
        ATV_CUDA_CHECK(cudaGetLastError());
    }

  private:
    double energy_ev_;
};

struct LifecycleCounts {
    int builds = 0;
    int launches = 0;
    int destroys = 0;
    bool live = false;
    bool fail_once = false;
};

class ReferenceLifecycle : public ATV::cuda::Component {
  public:
    explicit ReferenceLifecycle(std::shared_ptr<LifecycleCounts> counts)
        : counts_(std::move(counts)) {}
    std::string name() const override { return "mp2_reference_lifecycle"; }
    void build(const ATV::cuda::BuildContext&) override {
        if (counts_->live)
            throw std::logic_error("MP2 reference stage was built twice");
        counts_->live = true;
        ++counts_->builds;
    }
    void launch(const ATV::cuda::LaunchContext&) override {
        if (!counts_->live)
            throw std::logic_error(
                "MP2 reference stage launched without build");
        ++counts_->launches;
        if (counts_->fail_once) {
            counts_->fail_once = false;
            throw std::runtime_error("injected MP2 reference stage failure");
        }
    }
    void destroy(const ATV::cuda::DestroyContext&) override {
        if (!counts_->live)
            throw std::logic_error("MP2 reference stage was destroyed twice");
        counts_->live = false;
        ++counts_->destroys;
    }

  private:
    std::shared_ptr<LifecycleCounts> counts_;
};

bool frame_unchanged(const ATV::cpu::Frame& got,
                     const ATV::cpu::Frame& before) {
    if (got.blocks.size() != before.blocks.size()
        || got.meta.entries().size() != before.meta.entries().size())
        return false;
    for (const auto& [name, block] : before.blocks) {
        const auto found = got.blocks.find(name);
        if (found == got.blocks.end() || found->second.rows() != block.rows())
            return false;
        const auto wanted = block.resolve_columns();
        const auto actual = found->second.resolve_columns();
        if (actual.size() != wanted.size())
            return false;
        for (std::size_t i = 0; i < wanted.size(); ++i) {
            if (actual[i].key != wanted[i].key
                || actual[i].dtype != wanted[i].dtype
                || actual[i].cols != wanted[i].cols)
                return false;
            const std::size_t bytes = static_cast<std::size_t>(block.rows())
                                      * wanted[i].cols * wanted[i].element_size;
            if (bytes
                && std::memcmp(actual[i].bytes(), wanted[i].bytes(), bytes)
                       != 0)
                return false;
        }
    }
    for (const auto& [key, value] : before.meta.entries()) {
        const auto found = got.meta.entries().find(key);
        if (found == got.meta.entries().end()
            || found->second.serializable != value.serializable
            || found->second.value != value.value)
            return false;
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (got.box.cell()(i, j) != before.box.cell()(i, j))
                return false;
    return true;
}

ATV::cpu::Frame sentinel_frame() {
    using namespace ATV;
    auto frame = h2o_frame();
    frame.meta.get(kEnergy) = 42.0;
    frame.meta.get(kEnergyElectronic) = -7.0;
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_fx)[i] = 0.125;
        frame.span(kAtom_fy)[i] = -0.25;
        frame.span(kAtom_fz)[i] = 0.5;
        frame.span(kAtom_vx)[i] = 0.03125 * (i + 1);
    }
    return frame;
}

bool check_cuda_contracts(ATV::cuda::scf::RHF& rhf,
                          ATV::cuda::scf::MP2& correlation,
                          double want_total_ha) {
    using namespace ATV;
    constexpr double known_ev = 2.0;
    constexpr double tolerance_ha = 1e-6; // Existing external MP2 product gate.
    const auto lifecycle = std::make_shared<LifecycleCounts>();
    rhf.on(scf::Phase::check).add(ReferenceLifecycle{lifecycle}, 1000);
    rhf.build();

    const auto require = [](bool condition, const char* label) {
        if (!condition)
            std::printf("FAIL [cuda]: MP2 %s got=0 want=1 tol=0\n", label);
        return condition;
    };
    auto readonly = sentinel_frame();
    const cpu::Frame before_readonly(readonly);
    const auto const_result =
        correlation.run(static_cast<const cpu::Frame&>(readonly));
    bool ok = require(frame_unchanged(readonly, before_readonly),
                      "const Frame preservation");
    ok = require(std::isfinite(const_result.e_tot)
                     && std::abs(const_result.e_tot - want_total_ha)
                            < tolerance_ha,
                 "const Frame external energy")
         && ok;
    ok = require(lifecycle->builds == 1 && lifecycle->destroys == 1
                     && lifecycle->launches > 0 && !lifecycle->live,
                 "standalone stage lifecycle")
         && ok;

    {
        cuda::Evaluate evaluation{false, false};
        evaluation.on(evaluate::Phase::evaluate)
            .add(KnownEnergy{known_ev}, -10);
        evaluation.set_evaluator_ref(correlation);
        evaluation.build();
        auto nested = sentinel_frame();
        const auto check_energy = [&] {
            const double got = nested.meta.get(kEnergy) / units::kHartreeInEv;
            const double want = want_total_ha + known_ev / units::kHartreeInEv;
            if (!std::isfinite(got) || std::abs(got - want) > tolerance_ha) {
                std::printf("FAIL [cuda]: nested MP2 energy got=%.12f "
                            "want=%.12f tol=%.1e Ha\n",
                            got, want, tolerance_ha);
                return false;
            }
            return true;
        };
        const int builds_before = lifecycle->builds;
        const int launches_before = lifecycle->launches;
        evaluation.run(nested);
        ok = check_energy() && ok;
        ok = require(lifecycle->builds == builds_before + 1
                         && lifecycle->destroys == lifecycle->builds
                         && lifecycle->launches > launches_before
                         && !lifecycle->live,
                     "nested stage lifecycle")
             && ok;
        ok = require(nested.span(kAtom_fx)[0] == 0.125
                         && nested.span(kAtom_fy)[1] == -0.25
                         && nested.span(kAtom_fz)[2] == 0.5,
                     "energy-only force preservation")
             && ok;

        // The same parent, MP2 and reference survive a failed electronic
        // stage. Failure must release borrowed stages and retrieve nothing.
        lifecycle->fail_once = true;
        const cpu::Frame before_failure(nested);
        bool rejected = false;
        try {
            evaluation.run(nested);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what())
                       == "injected MP2 reference stage failure";
        }
        ok = require(rejected && frame_unchanged(nested, before_failure)
                         && !lifecycle->live
                         && lifecycle->builds == lifecycle->destroys,
                     "failed nested stage preserves host Frame and detaches")
             && ok;
        evaluation.run(nested);
        ok = check_energy() && ok;
        ok = require(!lifecycle->live
                         && lifecycle->builds == lifecycle->destroys,
                     "same-instance retry lifecycle")
             && ok;
    }

    {
        cuda::Evaluate force_request;
        force_request.set_evaluator_ref(correlation);
        force_request.build();
        auto frame = sentinel_frame();
        const cpu::Frame before(frame);
        bool rejected = false;
        try {
            force_request.run(frame);
        } catch (const std::invalid_argument& error) {
            rejected =
                std::string(error.what()).find("forces are not implemented")
                != std::string::npos;
        }
        ok = require(rejected && frame_unchanged(frame, before)
                         && !lifecycle->live
                         && lifecycle->builds == lifecycle->destroys,
                     "force-request rejection preserves host Frame")
             && ok;
    }

    auto failed_frame = sentinel_frame();
    const cpu::Frame before_nonconvergence(failed_frame);
    cuda::scf::RHF short_reference{cpu::scf::BasisSet::STO3G, 0, 1e-12, 1};
    cuda::scf::MP2 short_correlation{short_reference};
    bool nonconverged_rejected = false;
    try {
        short_correlation.run(failed_frame);
    } catch (const std::runtime_error& error) {
        nonconverged_rejected =
            std::string(error.what()).find("did not converge")
            != std::string::npos;
    }
    ok = require(nonconverged_rejected && !short_reference.converged()
                     && frame_unchanged(failed_frame, before_nonconvergence),
                 "nonconverged reference preserves host Frame")
         && ok;
    rhf.unfix("mp2_reference_lifecycle");
    return ok;
}

} // namespace

template <typename MeanField, typename Correlation>
int run_on() {
    using namespace ATV;

    auto frame = h2o_frame();

    MeanField rhf{cpu::scf::BasisSet::STO3G, 0, 1e-12, 200};
    // Grimme SCS-MP2 (J. Chem. Phys. 118, 9095 (2003)): members, not a Config.
    Correlation mp_corr{rhf};
    mp_corr.os_scale = 1.2;
    mp_corr.ss_scale = 1.0 / 3.0;

    const cpu::scf::MP2Result mp2 = mp_corr.run(frame);

    // PySCF 2.14.0 pins — see REFERENCE.md (never live-import PySCF here).
    constexpr double want_e_hf = -74.963023138463;
    constexpr double want_e_os = -0.033515820085;
    constexpr double want_e_ss = -0.002029831562;
    constexpr double want_e_corr = -0.040895594623; // SCS product gate
    constexpr double want_e_tot = -75.003918733086;
    constexpr double tol = 1e-6; // Ha

    const double e_hf = rhf.energy();
    std::printf("RHF     E = %.12f Ha (pin %.12f, dE = %.3e)\n", e_hf,
                want_e_hf, e_hf - want_e_hf);
    std::printf("MP2 OS  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_corr_os,
                want_e_os, mp2.e_corr_os - want_e_os);
    std::printf("MP2 SS  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_corr_ss,
                want_e_ss, mp2.e_corr_ss - want_e_ss);
    std::printf("SCS-MP2 e_corr = %.12f Ha (pin %.12f, dE = %.3e)\n",
                mp2.e_corr, want_e_corr, mp2.e_corr - want_e_corr);
    std::printf("SCS-MP2 e_tot  = %.12f Ha (pin %.12f, dE = %.3e)\n", mp2.e_tot,
                want_e_tot, mp2.e_tot - want_e_tot);
    std::printf("scales: c_os=%.6f c_ss=%.6f\n", mp_corr.os_scale,
                mp_corr.ss_scale);

    bool ok = rhf.converged();
    if constexpr (std::is_same_v<MeanField, cuda::scf::RHF>)
        ok = ok
             && std::fabs(frame.meta.get(kEnergy) / units::kHartreeInEv
                          - mp2.e_tot)
                    < 1e-10;
    ok = ok && std::fabs(e_hf - want_e_hf) < 1e-7;
    ok = ok && std::fabs(mp2.e_corr_os - want_e_os) < tol;
    ok = ok && std::fabs(mp2.e_corr_ss - want_e_ss) < tol;
    ok = ok && std::fabs(mp2.e_corr - want_e_corr) < tol;
    ok = ok && std::fabs(mp2.e_tot - want_e_tot) < tol;
    ok = ok && std::fabs(mp2.e_tot - (e_hf + mp2.e_corr)) < 1e-12;
    // Reconstruct SCS from unscaled channels.
    ok = ok
         && std::fabs(mp2.e_corr
                      - (1.2 * mp2.e_corr_os + (1.0 / 3.0) * mp2.e_corr_ss))
                < 1e-12;
    ok = ok && !mp2.ri;
    ok = ok && mp2.frozen == 0;
    ok = ok && mp2.e_corr < 0.0;
    // SCS should be more negative than plain RMP2 (~ -0.0355 Ha) on this system.
    ok = ok && mp2.e_corr < -0.038;

    if constexpr (std::is_same_v<MeanField, cuda::scf::RHF>)
        ok = check_cuda_contracts(rhf, mp_corr, want_e_tot) && ok;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int run(int argc, char** argv) {
    std::string device = "both";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--device=", 9) == 0)
            device = argv[i] + 9;
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device = argv[++i];
    }
    if (device != "cpu" && device != "cuda" && device != "both") {
        std::fprintf(stderr, "FAIL unknown --device=%s\n", device.c_str());
        return 1;
    }
    int result = 0;
    if (device != "cuda") {
        std::printf("[cpu]\n");
        result |= run_on<ATV::cpu::scf::RHF,
                         ATV::cpu::scf::MP2<ATV::cpu::scf::RHF>>();
    }
    if (device != "cpu") {
        std::printf("[cuda]\n");
        result |= run_on<ATV::cuda::scf::RHF, ATV::cuda::scf::MP2>();
    }
    return result;
}

} // namespace atv_reg_scf_scs_mp2_h2o_sto3g

ATV_REGRESSION("scf/mp2", "scs-mp2-h2o-sto3g",
               atv_reg_scf_scs_mp2_h2o_sto3g::run);
