/// @file cpu/scf/post_hf.cpp
/// @brief Post-HF entry points. Algorithms stay in the internal packs
///        (mp2.kernel, CCSD.kernel / ccsd_t, CASCI/CASSCF, NEVPT). This file
///        only moves the dispatch off MeanFieldMethod.

#include "atomiverse/cpu/scf/post_hf.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/scf/internal/casci_pack.h"
#include "atomiverse/cpu/scf/internal/casscf_cycle.h"
#include "atomiverse/cpu/scf/internal/ccsd_gradient.h"
#include "atomiverse/cpu/scf/internal/ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/ccsd_spatial.h"
#include "atomiverse/cpu/scf/internal/dispersion.h"
#include "atomiverse/cpu/scf/internal/eom_cc2_pack.h"
#include "atomiverse/cpu/scf/internal/eom_ea_ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/eom_ee_ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/eom_ip_ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/eom_sf_ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/fci_pack.h"
#include "atomiverse/cpu/scf/internal/mp2_gradient.h"
#include "atomiverse/cpu/scf/internal/mp2_pack.h"
#include "atomiverse/cpu/scf/internal/mp3_pack.h"
#include "atomiverse/cpu/scf/internal/mp4_pack.h"
#include "atomiverse/cpu/scf/internal/nevpt2_pack.h"
#include "atomiverse/cpu/scf/methods.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace {

void refuse_ks(MeanFieldType method, const char* entry) {
    if (method == MeanFieldType::KS)
        throw std::invalid_argument(
            std::string(entry)
            + ": a Kohn–Sham reference is refused. CC, CI, CAS, and MP3/MP4 "
              "take a Hartree–Fock mean field (PySCF cc.ccsd.CCSD / "
              "mcscf.CASCI / fci.FCI on scf.RHF). The explicit KS product is "
              "MP2 on RKS (double hybrid), not this entry.");
}

void write_gradient(Frame& frame, const GradientResult& gr) {
    AccReal* __restrict fx = frame.span(kAtom_fx);
    AccReal* __restrict fy = frame.span(kAtom_fy);
    AccReal* __restrict fz = frame.span(kAtom_fz);
    const int natm = static_cast<int>(frame.nlocal());
    for (int A = 0; A < natm; ++A) {
        fx[A] = gr.forces(A, 0);
        fy[A] = gr.forces(A, 1);
        fz[A] = gr.forces(A, 2);
    }
    frame.meta.get(kEnergy) = gr.energy;
}

template <typename Method>
void drive_converged(Method& mf, Frame& frame, const SCFInitialGuess* guess,
                     bool materialize_eri, const char* entry) {
    struct RestoreEri {
        bool& value;
        bool saved;
        ~RestoreEri() { value = saved; }
    } restore{mf.compute_eri, mf.compute_eri};
    if (materialize_eri)
        mf.compute_eri = true;
    mf.run(frame, guess);
    mf.compute_eri = restore.saved;
    if (!mf.converged())
        throw std::invalid_argument(
            std::string(entry)
            + ": SCF reference did not converge — refusing to compute a "
              "correlation correction on a non-stationary determinant "
              "(no fallback)");
}

/// One plan op per CC amplitude phase. Math is `CcsdSpatialCycle`.
class AmpStage : public cpu::Component {
  public:
    enum class Kind { Update, Mix, Energy, Check };

    AmpStage(std::unique_ptr<CcsdSpatialCycle>* cycle, Kind kind)
        : cycle_(cycle), kind_(kind) {}

    std::string name() const override {
        switch (kind_) {
        case Kind::Update:
            return "ccsd_update";
        case Kind::Mix:
            return "ccsd_mix";
        case Kind::Energy:
            return "ccsd_energy";
        case Kind::Check:
            return "ccsd_check";
        }
        throw std::logic_error("cpu::scf::CCSD: unknown stage");
    }

    void launch(const cpu::LaunchContext& ctx) override {
        // Bound to one phase, so the runtime only calls this on that phase.
        if (!cycle_ || !*cycle_)
            throw std::runtime_error(
                "cpu::scf::CCSD: amplitude cycle is not open");
        CcsdSpatialCycle& cycle = **cycle_;
        if (kind_ == Kind::Update) {
            cycle.update();
            return;
        }
        if (kind_ == Kind::Mix) {
            cycle.mix();
            return;
        }
        if (kind_ == Kind::Energy) {
            cycle.energy();
            return;
        }
        if (!ctx.runtime.loop.metrics)
            throw std::runtime_error(
                "cpu::scf::CCSD: check stage has no ConvergenceMetrics");
        const double ha = cycle.e_scf + cycle.e_corr;
        ctx.runtime.loop.metrics->energy = ha * units::kHartreeInEv;
        ctx.runtime.loop.metrics->residual = cycle.residual();
    }

  private:
    std::unique_ptr<CcsdSpatialCycle>* cycle_ = nullptr;
    Kind kind_ = Kind::Update;
};

/// @brief max|Δt| on `ConvergenceMetrics::residual`. Does not assign
///        convergence; `verdict` only returns a Verdict.
class AmplitudeConverged : public cpu::Criterion {
  public:
    explicit AmplitudeConverged(double tol) : tol_(tol) {}

    std::string name() const override { return "ccsd_amplitude"; }

    bool contributes_to_convergence() const override { return true; }

    void launch(const cpu::LaunchContext&) override {}

    ATV::Verdict verdict(const cpu::LoopState& state) override {
        if (state.metrics.residual < tol_)
            return ATV::Verdict::Converged;
        return ATV::Verdict::Continue;
    }

  private:
    double tol_ = 0.0;
};

/// One plan op per CASSCF macro phase. Math is `CasscfCycle`.
class McStage : public cpu::Component {
  public:
    enum class Kind { Rotate, Casci, Gradient, Check };

    McStage(std::unique_ptr<CasscfCycle>* cycle, Kind kind)
        : cycle_(cycle), kind_(kind) {}

    std::string name() const override {
        switch (kind_) {
        case Kind::Rotate:
            return "casscf_rotate";
        case Kind::Casci:
            return "casscf_casci";
        case Kind::Gradient:
            return "casscf_gradient";
        case Kind::Check:
            return "casscf_check";
        }
        throw std::logic_error("cpu::scf::CASSCF: unknown stage");
    }

    void launch(const cpu::LaunchContext& ctx) override {
        if (!cycle_ || !*cycle_)
            throw std::runtime_error(
                "cpu::scf::CASSCF: orbital cycle is not open");
        CasscfCycle& cycle = **cycle_;
        if (kind_ == Kind::Rotate) {
            cycle.rotate();
            return;
        }
        if (kind_ == Kind::Casci) {
            cycle.casci();
            return;
        }
        if (kind_ == Kind::Gradient) {
            cycle.gradient();
            return;
        }
        if (!ctx.runtime.loop.metrics)
            throw std::runtime_error(
                "cpu::scf::CASSCF: check stage has no ConvergenceMetrics");
        ctx.runtime.loop.metrics->energy = cycle.e_cas * units::kHartreeInEv;
        ctx.runtime.loop.metrics->residual = cycle.residual();
    }

  private:
    std::unique_ptr<CasscfCycle>* cycle_ = nullptr;
    Kind kind_ = Kind::Rotate;
};

/// @brief max |g_pq| on `ConvergenceMetrics::residual`. Does not assign
///        convergence; `verdict` only returns a Verdict.
class OrbitalGradientConverged : public cpu::Criterion {
  public:
    explicit OrbitalGradientConverged(double tol) : tol_(tol) {}

    std::string name() const override { return "casscf_orbital"; }

    bool contributes_to_convergence() const override { return true; }

    void launch(const cpu::LaunchContext&) override {}

    ATV::Verdict verdict(const cpu::LoopState& state) override {
        if (state.metrics.residual < tol_)
            return ATV::Verdict::Converged;
        return ATV::Verdict::Continue;
    }

  private:
    double tol_ = 0.0;
};

} // namespace

template <typename Method>
MP2<Method>::MP2(Method& mf) : mf_(&mf) {}

template <typename Method>
MP2Result MP2<Method>::run(Frame& frame, const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    if (mf.method_ == MeanFieldType::KS && mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MP2::run: Kohn–Sham MP2 is only the restricted double-hybrid path "
            "(RKS). UKS and ROKS are not that product — use RKS or an HF "
            "reference.");
    if (mf.spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "MP2::run: a GHF (spinor) reference is not supported — the MP2 "
            "pack has no spinor transformation. Use RHF or UHF.");
    drive_converged(mf, frame, guess, /*materialize_eri=*/mf.ri == RI::None,
                    "MP2::run");
    if ((mf.spin_ == SpinType::Unrestricted
         || mf.spin_ == SpinType::RestrictedOpen)
        && frozen != 0) {
        throw std::invalid_argument(
            "MP2::run: frozen-core UMP2/ROMP2 is not supported — set "
            "frozen=0, or use closed-shell RHF.");
    }
    if (mf.ri != RI::None) {
        if (mf.spin_ != SpinType::Restricted)
            throw std::invalid_argument(
                "MP2::run: RI-MP2 is closed-shell only — set ri=RI::None for "
                "UMP2/ROMP2, or use RHF.");
        return ri_mp2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                             mf.aux_basis, frozen, os_scale, ss_scale);
    }
    if (mf.spin_ == SpinType::RestrictedOpen) {
        return romp2_energy(mf.integral_view_(), mf.determinant(),
                            mf.F_alpha_uncoupled_, mf.F_beta_uncoupled_,
                            mf.energy(), os_scale, ss_scale);
    }
    if (mf.spin_ == SpinType::Unrestricted)
        return ump2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                           os_scale, ss_scale);
    return mp2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                      frozen, os_scale, ss_scale);
}

template <typename Method>
GradientResult MP2<Method>::forces(Frame& frame, const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    if (mf.method_ == MeanFieldType::KS && mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MP2::forces: Kohn–Sham MP2 is only the restricted double-hybrid "
            "path "
            "(RKS). UKS and ROKS are not that product — use RKS or an HF "
            "reference.");
    if (mf.spin_ != SpinType::Restricted && mf.spin_ != SpinType::Unrestricted)
        throw std::invalid_argument(
            "MP2::forces: RMP2 (RHF) or UMP2 (UHF) analytic gradient only — "
            "ROHF/ROMP2 gradients are a follow-up");
    if (mf.ri != RI::None)
        throw std::invalid_argument(
            "MP2::forces: RI-MP2 gradients are not implemented — set "
            "ri=RI::None");
    if (frozen != 0)
        throw std::invalid_argument(
            "MP2::forces: frozen-core MP2 gradients are not implemented — set "
            "frozen=0");
    if (std::fabs(os_scale - 1.0) > 0.0 || std::fabs(ss_scale - 1.0) > 0.0)
        throw std::invalid_argument(
            "MP2::forces: spin-component-scaled MP2 gradients are not "
            "implemented — keep os_scale=ss_scale=1");

    drive_converged(mf, frame, guess, /*materialize_eri=*/true, "MP2::forces");

    GradientResult gr;
    if (mf.spin_ == SpinType::Unrestricted) {
        const MP2Result corr =
            ump2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                        /*os_scale=*/1.0, /*ss_scale=*/1.0);
        gr = internal::ump2_analytic_forces(frame, mf.integral_view_(),
                                            mf.determinant(), mf.energy(),
                                            corr.e_corr);
    } else {
        const MP2Result corr =
            mp2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                       /*n_frozen=*/0, /*os_scale=*/1.0, /*ss_scale=*/1.0);
        gr = internal::mp2_analytic_forces(frame, mf.integral_view_(),
                                           mf.determinant(), mf.energy(),
                                           corr.e_corr);
    }
    internal::add_dispersion_forces(frame, mf.d3bj, mf.d4, gr);
    mf.last_grad_ = gr;
    write_gradient(frame, gr);
    return gr;
}

template <typename Method>
MP3<Method>::MP3(Method& mf) : mf_(&mf) {}

template <typename Method>
MP3Result MP3<Method>::run(Frame& frame, const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "MP3::run");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MP3::run: closed-shell RMP3 only — use RHF (open-shell MP3 is a "
            "follow-up).");
    if (mf.ri != RI::None)
        throw std::invalid_argument(
            "MP3::run: RI-MP3 is not implemented — set ri=RI::None");
    drive_converged(mf, frame, guess, /*materialize_eri=*/true, "MP3::run");
    return mp3_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                      frozen);
}

template <typename Method>
MP4<Method>::MP4(Method& mf) : mf_(&mf) {}

template <typename Method>
MP4Result MP4<Method>::run(Frame& frame, const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "MP4::run");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MP4::run: closed-shell RMP4 only — use RHF (open-shell MP4 is a "
            "follow-up).");
    if (mf.ri != RI::None)
        throw std::invalid_argument(
            "MP4::run: RI-MP4 is not implemented — set ri=RI::None");
    drive_converged(mf, frame, guess, /*materialize_eri=*/true, "MP4::run");
    return mp4_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                      frozen);
}

template <typename Method>
FCI<Method>::FCI(Method& mf) : mf_(&mf) {}

template <typename Method>
FCIResult FCI<Method>::run(Frame& frame) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "FCI::run");
    if (mf.spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "FCI::run: an ROHF reference is not supported — the FCI pack "
            "requires a closed-shell reference. Use RHF.");
    if (mf.spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "FCI::run: a GHF (spinor) reference is not supported — the FCI "
            "pack requires a closed-shell reference. Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true, "FCI::run");
    return fci_energy(mf.integral_view_(), mf.determinant(), mf.energy());
}

template <typename Method>
CASCI<Method>::CASCI(Method& mf) : mf_(&mf) {}

template <typename Method>
CASCIResult CASCI<Method>::run(Frame& frame, int ncas, int nelecas) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CASCI::run");
    if (mf.spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "CASCI::run: an ROHF reference is not supported — closed-shell "
            "CASCI requires RHF.");
    if (mf.spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "CASCI::run: a GHF (spinor) reference is not supported — "
            "closed-shell CASCI requires RHF.");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CASCI::run: closed-shell RHF reference only — UHF CASCI is not "
            "implemented (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true, "CASCI::run");
    return casci_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                        ncas, nelecas);
}

template <typename Method>
struct CASSCF<Method>::Loop {
    /// Opens one orbital cycle and drops it on every exit, including a throw.
    struct Hold {
        Loop* loop = nullptr;

        Hold(Loop& loop_in, const internal::IntegralView& ints,
             const SlaterDeterminant& det, double e_scf_ha, int ncas_in,
             int nelecas_in)
            : loop(&loop_in) {
            loop->cycle = std::make_unique<CasscfCycle>(ints, det, e_scf_ha,
                                                        ncas_in, nelecas_in);
        }

        ~Hold() {
            if (loop)
                loop->cycle.reset();
        }

        Hold(const Hold&) = delete;
        Hold& operator=(const Hold&) = delete;
    };

    std::unique_ptr<CasscfCycle> cycle;
    McStage rotate;
    McStage casci;
    McStage gradient;
    McStage check;
    OrbitalGradientConverged criterion;

    explicit Loop(double tol_in)
        : rotate(&cycle, McStage::Kind::Rotate),
          casci(&cycle, McStage::Kind::Casci),
          gradient(&cycle, McStage::Kind::Gradient),
          check(&cycle, McStage::Kind::Check), criterion(tol_in) {}
};

template <typename Method>
CASSCF<Method>::CASSCF(Method& mf, int ncas_in, int nelecas_in)
    : CASSCF(mf, ncas_in, nelecas_in, std::max(mf.tol, 1e-8), 50) {}

template <typename Method>
CASSCF<Method>::CASSCF(Method& mf, int ncas_in, int nelecas_in,
                       double orb_tol_in, int max_macro_in)
    : ncas(ncas_in), nelecas(nelecas_in), orb_tol(orb_tol_in),
      max_macro(max_macro_in), mf_(&mf),
      loop_(std::make_unique<Loop>(orb_tol_in)) {
    if (ncas <= 0)
        throw std::invalid_argument("CASSCF: ncas must be positive (got "
                                    + std::to_string(ncas) + ")");
    if (nelecas < 0 || (nelecas % 2) != 0)
        throw std::invalid_argument(
            "CASSCF: closed-shell CAS requires even nelecas (got "
            + std::to_string(nelecas) + ")");
    if (!(orb_tol > 0.0) || max_macro < 1)
        throw std::invalid_argument(
            "CASSCF: orb_tol must be positive and max_macro >= 1");
    this->on(ATV::mc::Phase::rotate).add_ref(loop_->rotate);
    this->on(ATV::mc::Phase::casci).add_ref(loop_->casci);
    this->on(ATV::mc::Phase::gradient).add_ref(loop_->gradient);
    this->on(ATV::mc::Phase::check).add_ref(loop_->check);
    this->add_criterion_ref(loop_->criterion);
    this->set_serial_on_replica();
    this->build();
}

template <typename Method>
CASSCF<Method>::~CASSCF() {
    if (loop_)
        loop_->cycle.reset();
    this->shutdown();
}

template <typename Method>
CASSCF<Method>::PriorEnergy CASSCF<Method>::capture(const Frame& frame) {
    PriorEnergy prior;
    if (frame.meta.contains(kEnergy.key)) {
        prior.energy = true;
        prior.energy_v = frame.meta.get(kEnergy);
    }
    if (frame.meta.contains(kEnergyElectronic.key)) {
        prior.electronic = true;
        prior.electronic_v = frame.meta.get(kEnergyElectronic);
    }
    return prior;
}

template <typename Method>
void CASSCF<Method>::store_energy(Frame& frame, double e_ev, Write write,
                                  const PriorEnergy& prior) {
    if (!frame.meta.contains(kEnergy.key))
        throw std::runtime_error(
            "CASSCF: reference SCF did not publish kEnergy");
    if (write == Write::Accumulate)
        frame.meta.get(kEnergy) = (prior.energy ? prior.energy_v : 0.0) + e_ev;
    else
        frame.meta.get(kEnergy) = e_ev;
    if (!frame.meta.contains(kEnergyElectronic.key))
        return;
    if (write == Write::Accumulate) {
        frame.meta.get(kEnergyElectronic) =
            (prior.electronic ? prior.electronic_v : 0.0) + e_ev;
    } else {
        frame.meta.get(kEnergyElectronic) = e_ev;
    }
}

template <typename Method>
CASSCFResult CASSCF<Method>::finish(Frame& frame, int executed, int cap,
                                    Write write, const PriorEnergy& prior) {
    if (!this->converged()) {
        throw std::invalid_argument(
            "CASSCF: orbital gradient did not converge in "
            + std::to_string(cap)
            + " macro iterations (orb_tol=" + std::to_string(orb_tol) + ").");
    }
    const CasscfCycle& cycle = *loop_->cycle;
    executed_ = executed;
    last_ = CASSCFResult{cycle.e_cas, cycle.e_cas - cycle.e_scf,
                         ncas,        nelecas,
                         cycle.ncore, executed,
                         true};
    if (write != Write::None)
        store_energy(frame, last_.e_casscf * units::kHartreeInEv, write, prior);
    return last_;
}

template <typename Method>
void CASSCF<Method>::require_reference(const char* what) const {
    const Method& mf = *mf_;
    refuse_ks(mf.method_, what);
    if (mf.spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            std::string(what)
            + ": an ROHF reference is not supported — closed-shell "
              "CASSCF requires RHF.");
    if (mf.spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            std::string(what)
            + ": a GHF (spinor) reference is not supported — "
              "closed-shell CASSCF requires RHF.");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            std::string(what)
            + ": closed-shell RHF reference only — UHF CASSCF is not "
              "implemented (no fallback). Use RHF.");
}

template <typename Method>
CASSCFResult CASSCF<Method>::solve(Frame& frame, Write write, int cap,
                                   cpu::Context* nested) {
    require_reference("CASSCF::run");
    if (!(orb_tol > 0.0) || cap < 1)
        throw std::invalid_argument(
            "CASSCF: orb_tol must be positive and max_macro >= 1");
    const PriorEnergy prior = capture(frame);
    Method& mf = *mf_;
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CASSCF::run");
    typename Loop::Hold hold(*loop_, mf.integral_view_(), mf.determinant(),
                             mf.energy(), ncas, nelecas);
    if (nested) {
        const int executed =
            cpu::Driver<ATV::mc::Phase>::run_nested(*nested, frame, cap);
        return finish(frame, executed, cap, write, prior);
    }
    this->setup_pass(frame);
    try {
        int executed = 0;
        for (int it = 0; it < cap; ++it) {
            this->mutable_outer_status().reset();
            this->step_once(it);
            ++executed;
            if (this->poll_criteria_(it) != ATV::Verdict::Continue)
                break;
        }
        const CASSCFResult result = finish(frame, executed, cap, write, prior);
        this->teardown_pass();
        return result;
    } catch (...) {
        this->teardown_pass();
        throw;
    }
}

template <typename Method>
CASSCFResult CASSCF<Method>::run(Frame& frame) {
    return solve(frame, Write::Overwrite, max_macro, nullptr);
}

template <typename Method>
CASSCFResult CASSCF<Method>::run(const Frame& frame) {
    Frame scratch(frame);
    return run(scratch);
}

template <typename Method>
int CASSCF<Method>::run_nested_erased(cpu::Context& ctx, cpu::Frame& frame,
                                      int budget) {
    if (!ctx.exec.runtime)
        throw std::runtime_error("cpu::scf::CASSCF: nested run has no Runtime");
    if (ctx.exec.forces_requested)
        throw std::invalid_argument(
            "CASSCF: nuclear forces are not implemented — this method "
            "provides energy only");
    const int cap = budget > 0 ? budget : max_macro;
    solve(frame, Write::Accumulate, cap, &ctx);
    return executed_;
}

template <typename Method>
NEVPT2<Method>::NEVPT2(Method& mf) : mf_(&mf) {}

template <typename Method>
NEVPT2Result NEVPT2<Method>::run(Frame& frame, int ncas, int nelecas) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "NEVPT2::run");
    if (mf.spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "NEVPT2::run: an ROHF reference is not supported — closed-shell "
            "SC-NEVPT2 requires RHF.");
    if (mf.spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "NEVPT2::run: a GHF (spinor) reference is not supported — "
            "closed-shell SC-NEVPT2 requires RHF.");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "NEVPT2::run: closed-shell RHF reference only — UHF NEVPT2 is not "
            "implemented (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "NEVPT2::run");
    return nevpt2_energy(mf.integral_view_(), mf.determinant(), mf.energy(),
                         ncas, nelecas, /*orb_tol=*/std::max(mf.tol, 1e-8),
                         /*max_macro=*/50);
}

template <typename Method>
struct CCSD<Method>::Loop {
    /// Opens one amplitude cycle and drops it on every exit, including a throw.
    struct Hold {
        Loop* loop = nullptr;

        Hold(Loop& loop_in, const internal::IntegralView& ints,
             const SlaterDeterminant& det, double e_scf_ha, bool use_diis)
            : loop(&loop_in) {
            loop->cycle = std::make_unique<CcsdSpatialCycle>(
                ints, det, e_scf_ha, use_diis);
        }

        ~Hold() {
            if (loop)
                loop->cycle.reset();
        }

        Hold(const Hold&) = delete;
        Hold& operator=(const Hold&) = delete;
    };

    std::unique_ptr<CcsdSpatialCycle> cycle;
    AmpStage update;
    AmpStage mix;
    AmpStage energy;
    AmpStage check;
    AmplitudeConverged criterion;

    explicit Loop(double tol_in)
        : update(&cycle, AmpStage::Kind::Update),
          mix(&cycle, AmpStage::Kind::Mix),
          energy(&cycle, AmpStage::Kind::Energy),
          check(&cycle, AmpStage::Kind::Check), criterion(tol_in) {}
};

template <typename Method>
CCSD<Method>::CCSD(Method& mf, bool diis_in, int frozen_in, double tol_in,
                   int max_iter_in)
    : diis(diis_in), frozen(frozen_in), tol(tol_in), max_iter(max_iter_in),
      mf_(&mf), loop_(std::make_unique<Loop>(tol_in)) {
    this->on(ATV::cc::Phase::update).add_ref(loop_->update);
    this->on(ATV::cc::Phase::mix).add_ref(loop_->mix);
    this->on(ATV::cc::Phase::energy).add_ref(loop_->energy);
    this->on(ATV::cc::Phase::check).add_ref(loop_->check);
    this->add_criterion_ref(loop_->criterion);
    this->set_serial_on_replica();
    this->build();
}

template <typename Method>
CCSD<Method>::~CCSD() {
    if (loop_)
        loop_->cycle.reset();
    this->shutdown();
}

template <typename Method>
CCSD<Method>::PriorEnergy CCSD<Method>::capture(const Frame& frame) {
    PriorEnergy prior;
    if (frame.meta.contains(kEnergy.key)) {
        prior.energy = true;
        prior.energy_v = frame.meta.get(kEnergy);
    }
    if (frame.meta.contains(kEnergyElectronic.key)) {
        prior.electronic = true;
        prior.electronic_v = frame.meta.get(kEnergyElectronic);
    }
    return prior;
}

template <typename Method>
void CCSD<Method>::store_energy(Frame& frame, double e_ev, Write write,
                                const PriorEnergy& prior) {
    if (!frame.meta.contains(kEnergy.key))
        throw std::runtime_error("CCSD: reference SCF did not publish kEnergy");
    if (write == Write::Accumulate)
        frame.meta.get(kEnergy) = (prior.energy ? prior.energy_v : 0.0) + e_ev;
    else
        frame.meta.get(kEnergy) = e_ev;
    if (!frame.meta.contains(kEnergyElectronic.key))
        return;
    if (write == Write::Accumulate) {
        frame.meta.get(kEnergyElectronic) =
            (prior.electronic ? prior.electronic_v : 0.0) + e_ev;
    } else {
        frame.meta.get(kEnergyElectronic) = e_ev;
    }
}

template <typename Method>
CCSDResult CCSD<Method>::finish(Frame& frame, int executed, int cap,
                                Write write, const PriorEnergy& prior) {
    if (!this->converged()) {
        throw std::invalid_argument(
            "ccsd_energy_spatial: amplitudes did not converge in "
            + std::to_string(cap) + " iterations (tol=" + std::to_string(tol)
            + ").");
    }
    const CcsdSpatialCycle& cycle = *loop_->cycle;
    const double e_corr = cycle.e_corr;
    const double e_scf = cycle.e_scf;
    const int reported = cycle.trivial ? 0 : executed;
    executed_ = executed;
    last_ = CCSDResult{e_corr, e_scf + e_corr, reported, true};
    if (write != Write::None) {
        store_energy(frame, last_.e_tot * units::kHartreeInEv, write, prior);
    }
    return last_;
}

template <typename Method>
CCSDResult CCSD<Method>::solve_amplitudes(Frame& frame, Write write,
                                          const PriorEnergy& prior, int cap,
                                          cpu::Context* nested) {
    if (!(tol > 0.0) || cap < 1)
        throw std::invalid_argument(
            "CCSD: tol must be positive and max_iter >= 1");
    Method& mf = *mf_;
    typename Loop::Hold hold(*loop_, mf.integral_view_(), mf.determinant(),
                             mf.energy(), diis);
    if (nested) {
        const int executed =
            cpu::Driver<ATV::cc::Phase>::run_nested(*nested, frame, cap);
        return finish(frame, executed, cap, write, prior);
    }
    this->setup_pass(frame);
    try {
        int executed = 0;
        for (int it = 0; it < cap; ++it) {
            this->mutable_outer_status().reset();
            this->step_once(it);
            ++executed;
            if (this->poll_criteria_(it) != ATV::Verdict::Continue)
                break;
        }
        const CCSDResult result = finish(frame, executed, cap, write, prior);
        this->teardown_pass();
        return result;
    } catch (...) {
        this->teardown_pass();
        throw;
    }
}

template <typename Method>
CCSDResult CCSD<Method>::run(Frame& frame) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::run");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::run: closed-shell RHF reference only — UHF / ROHF / GHF "
            "CCSD is not implemented (no fallback). Use RHF.");
    const PriorEnergy prior = capture(frame);
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true, "CCSD::run");
    return solve_amplitudes(frame, Write::Overwrite, prior, max_iter, nullptr);
}

template <typename Method>
CCSDResult CCSD<Method>::run(const Frame& frame) {
    Frame scratch(frame);
    return run(scratch);
}

template <typename Method>
GradientResult CCSD<Method>::compute_forces(Frame& frame, Write write,
                                            cpu::Context* nested) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::forces");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::forces: closed-shell RHF reference only — UCCSD gradients "
            "are not implemented (no fallback). Use RHF.");
    if (mf.ri != RI::None)
        throw std::invalid_argument(
            "CCSD::forces: RI-CCSD gradients are not implemented — set "
            "ri=RI::None");
    if (frozen != 0)
        throw std::invalid_argument(
            "CCSD::forces: frozen-core CCSD gradients are not implemented — "
            "set frozen=0");
    const PriorEnergy prior = capture(frame);
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CCSD::forces");
    const CCSDResult corr =
        solve_amplitudes(frame, Write::None, prior, max_iter, nested);
    GradientResult gr = internal::ccsd_analytic_forces(
        frame, mf.integral_view_(), mf.determinant(), mf.energy(), corr.e_corr);
    internal::add_dispersion_forces(frame, mf.d3bj, mf.d4, gr);
    mf.last_grad_ = gr;
    AccReal* __restrict fx = frame.span(kAtom_fx);
    AccReal* __restrict fy = frame.span(kAtom_fy);
    AccReal* __restrict fz = frame.span(kAtom_fz);
    const int natm = static_cast<int>(frame.nlocal());
    for (int A = 0; A < natm; ++A) {
        if (write == Write::Accumulate) {
            fx[A] += gr.forces(A, 0);
            fy[A] += gr.forces(A, 1);
            fz[A] += gr.forces(A, 2);
        } else {
            fx[A] = gr.forces(A, 0);
            fy[A] = gr.forces(A, 1);
            fz[A] = gr.forces(A, 2);
        }
    }
    store_energy(frame, gr.energy, write, prior);
    return gr;
}

template <typename Method>
GradientResult CCSD<Method>::forces(Frame& frame) {
    return compute_forces(frame, Write::Overwrite, nullptr);
}

template <typename Method>
int CCSD<Method>::run_nested_erased(cpu::Context& ctx, cpu::Frame& frame,
                                    int budget) {
    if (!ctx.exec.runtime)
        throw std::runtime_error("cpu::scf::CCSD: nested run has no Runtime");
    if (ctx.exec.forces_requested) {
        compute_forces(frame, Write::Accumulate, &ctx);
        return executed_;
    }
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::run");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::run: closed-shell RHF reference only — UHF / ROHF / GHF "
            "CCSD is not implemented (no fallback). Use RHF.");
    const PriorEnergy prior = capture(frame);
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true, "CCSD::run");
    const int cap = budget > 0 ? budget : max_iter;
    solve_amplitudes(frame, Write::Accumulate, prior, cap, &ctx);
    return executed_;
}

template <typename Method>
EOMEECCSDResult CCSD<Method>::eom_ee_ccsd(Frame& frame, int n_states) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::eom_ee_ccsd");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::eom_ee_ccsd: closed-shell RHF reference only — EOM-EE "
            "singlet is CCSD.eomee_ccsd_singlet (PySCF), not a UHF / ROHF / "
            "GHF entry (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CCSD::eom_ee_ccsd");
    return eom_ee_ccsd_energies(mf.integral_view_(), mf.determinant(),
                                mf.energy(), n_states);
}

template <typename Method>
EOMIPCCSDResult CCSD<Method>::eom_ip_ccsd(Frame& frame, int n_states) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::eom_ip_ccsd");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::eom_ip_ccsd: closed-shell RHF reference only — EOM-IP "
            "is CCSD.ipccsd (PySCF), not a UHF / ROHF / GHF entry "
            "(no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CCSD::eom_ip_ccsd");
    return eom_ip_ccsd_energies(mf.integral_view_(), mf.determinant(),
                                mf.energy(), n_states);
}

template <typename Method>
EOMEACCSDResult CCSD<Method>::eom_ea_ccsd(Frame& frame, int n_states) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::eom_ea_ccsd");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::eom_ea_ccsd: closed-shell RHF reference only — EOM-EA "
            "is CCSD.eaccsd (PySCF), not a UHF / ROHF / GHF entry "
            "(no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CCSD::eom_ea_ccsd");
    return eom_ea_ccsd_energies(mf.integral_view_(), mf.determinant(),
                                mf.energy(), n_states);
}

template <typename Method>
EOMSFCCSDResult CCSD<Method>::eom_sf_ccsd(Frame& frame, int n_states,
                                          const SCFInitialGuess* guess) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::eom_sf_ccsd");
    if (mf.spin_ == SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::eom_sf_ccsd: unrestricted UHF reference required "
            "(PySCF cc.UCCSD.eomsf_ccsd). Closed-shell EE is "
            "CCSD::eom_ee_ccsd.");
    drive_converged(mf, frame, guess, /*materialize_eri=*/true,
                    "CCSD::eom_sf_ccsd");
    return eom_sf_ccsd_energies(mf.integral_view_(), mf.determinant(),
                                mf.energy(), n_states);
}

template <typename Method>
CCSDTResult CCSD<Method>::ccsd_t(Frame& frame) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CCSD::ccsd_t");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CCSD::ccsd_t: closed-shell RHF reference only — UHF / ROHF / GHF "
            "CCSD(T) is not implemented (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CCSD::ccsd_t");
    return ccsd_t_energy(mf.integral_view_(), mf.determinant(), mf.energy());
}

template <typename Method>
CC2<Method>::CC2(Method& mf) : mf_(&mf) {}

template <typename Method>
CC2Result CC2<Method>::run(Frame& frame) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CC2::run");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CC2::run: closed-shell RHF reference only — UHF / ROHF / GHF CC2 "
            "is not implemented (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true, "CC2::run");
    return cc2_energy(mf.integral_view_(), mf.determinant(), mf.energy());
}

template <typename Method>
EOMCC2Result CC2<Method>::eom_cc2(Frame& frame, int n_states) {
    Method& mf = *mf_;
    refuse_ks(mf.method_, "CC2::eom_cc2");
    if (mf.spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "CC2::eom_cc2: closed-shell RHF reference only — EOM-CC2 is "
            "the cc2=True EOM-EE singlet (PySCF), not a UHF / ROHF / GHF "
            "entry (no fallback). Use RHF.");
    drive_converged(mf, frame, nullptr, /*materialize_eri=*/true,
                    "CC2::eom_cc2");
    return eom_cc2_energies(mf.integral_view_(), mf.determinant(), mf.energy(),
                            n_states);
}

#define ATV_INST_POSTHF(T)                                                     \
    template class MP2<T>;                                                     \
    template class MP3<T>;                                                     \
    template class MP4<T>;                                                     \
    template class FCI<T>;                                                     \
    template class CASCI<T>;                                                   \
    template class CASSCF<T>;                                                  \
    template class NEVPT2<T>;                                                  \
    template class CCSD<T>;                                                    \
    template class CC2<T>;

ATV_INST_POSTHF(RHF)
ATV_INST_POSTHF(ROHF)
ATV_INST_POSTHF(UHF)
ATV_INST_POSTHF(GHF)
ATV_INST_POSTHF(RKS)
ATV_INST_POSTHF(UKS)
ATV_INST_POSTHF(ROKS)

#undef ATV_INST_POSTHF

} // namespace scf
} // namespace cpu
} // namespace ATV
