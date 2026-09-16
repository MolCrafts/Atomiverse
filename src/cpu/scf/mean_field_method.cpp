/// @file cpu/scf/mean_field_method.cpp
/// @brief `MeanFieldMethod<D>` template implementation — Frame API; a
///        Driver<scf::Phase> loop over the sub-spec-02 stages, funneled
///        through the private drive_scf_. Explicit instantiations for the four
///        concrete methods live here; the per-method ctor defaults +
///        resolve_occupation live in rhf/uhf/rks/uks.cpp.

#include "atomiverse/cpu/scf/methods.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/reducers/xreducer.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/electronic/aspc_predictor.h"
#include "atomiverse/cpu/linalg/density.h" // build_density_matrix (CPMD fixed-density step)
#include "atomiverse/cpu/linalg/eigensolve.h" // CanonicalOrth, build_orthogonalization_matrix
#include "atomiverse/cpu/observe/mulliken.h"
#include "atomiverse/cpu/scf/ecp_potential.h"
#include "atomiverse/cpu/scf/internal/casci_pack.h"
#include "atomiverse/cpu/scf/internal/casscf_pack.h"
#include "atomiverse/cpu/scf/internal/ccsd_gradient.h"
#include "atomiverse/cpu/scf/internal/ccsd_pack.h"
#include "atomiverse/cpu/scf/internal/dispersion.h"
#include "atomiverse/cpu/scf/internal/external_field.h"
#include "atomiverse/cpu/scf/internal/fci_pack.h"
#include "atomiverse/cpu/scf/internal/gradient.h"
#include "atomiverse/cpu/scf/internal/hamiltonian_builder.h"
#include "atomiverse/cpu/scf/internal/hueckel_guess.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/level_shift.h" // kLevelShiftSkipThreshold
#include "atomiverse/cpu/scf/internal/mean_field_energy.h"
#include "atomiverse/cpu/scf/internal/minao.h"
#include "atomiverse/cpu/scf/internal/mp2_gradient.h"
#include "atomiverse/cpu/scf/internal/mp2_pack.h"
#include "atomiverse/cpu/scf/internal/mp3_pack.h"
#include "atomiverse/cpu/scf/internal/mp4_pack.h"
#include "atomiverse/cpu/scf/internal/nevpt2_pack.h"
#include "atomiverse/cpu/scf/internal/pcm.h"
#include "atomiverse/cpu/scf/internal/periodic_hf.h"
#include "atomiverse/cpu/scf/internal/point_charges.h"
#include "atomiverse/cpu/scf/internal/sap_guess.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/internal/x2c.h"
#include "atomiverse/cpu/scf/scf_guess.h"     // apply_homo_lumo_mixing
#include "atomiverse/cpu/scf/spin_analysis.h" // compute_spin_contamination

namespace ATV {
namespace cpu {
namespace scf {

namespace {

/// @brief Publish the two Car-Parrinello electronic energies [eV] on the frame
///        so Thermo, observers and `cpu::md::CpmdConserved` see the
///        adiabaticity signal (which `CpmdConserved` requires: it throws on an
///        absent key rather than silently dropping T_e from E_cons).
///
/// The keys are open-schema (not pre-inserted by `Frame(int)`), so the first
/// step inserts and later steps overwrite — the `kDipole` pattern, mirrored
/// from the GPW CPMD twin (src/cpu/gpw/gpw_evaluator.cpp).
/// @param frame     Frame to write.
/// @param te_ev     Fictitious electronic kinetic energy T_e [eV].
/// @param etherm_ev Electronic Nosé–Hoover thermostat energy [eV].
void publish_electronic_energies(Frame& frame, double te_ev, double etherm_ev) {
    if (!frame.meta.contains(kEnergyElectronKinetic.key))
        frame.meta.insert(kEnergyElectronKinetic, te_ev);
    else
        frame.meta.set(kEnergyElectronKinetic, te_ev);
    if (!frame.meta.contains(kEnergyElectronThermostat.key))
        frame.meta.insert(kEnergyElectronThermostat, etherm_ev);
    else
        frame.meta.set(kEnergyElectronThermostat, etherm_ev);
}

/// @brief Refuse a standalone (non-nested) entry under Car-Parrinello.
///
/// `run` / `forces` evaluate ONE geometry, and CP is the dynamics of the
/// electrons ALONG a nuclear trajectory — there is no trajectory here to
/// follow, so the request is meaningless rather than merely unsupported.
/// Hence `std::logic_error`, matching the GPW twin's entry-point convention.
/// @param cpmd  The method's `cpmd` flag.
/// @param entry Fully-qualified entry name, interpolated into the message.
/// @throws std::logic_error when `cpmd` is true.
void refuse_standalone_cpmd(bool cpmd, const char* entry) {
    if (!cpmd)
        return;
    throw std::logic_error(
        std::string(entry)
        + ": cpmd = true is a nested-MD-force-slot method — the electrons "
          "follow the nuclei, and a standalone single point has no nuclear "
          "trajectory. Bind this method at md::Phase::force in a cpu::md::MD "
          "loop, or clear cpmd.");
}

} // namespace

// ── PositionGuard: unwrap for the integral build, restore on exit ─────────────

template <typename D>
MeanFieldMethod<D>::PositionGuard::PositionGuard(Frame& frame, bool active)
    : frame_(frame), active_(active) {
    if (!active_)
        return;
    const int n = static_cast<int>(frame_.nlocal());
    const StorageReal* __restrict px = frame_.span(kAtom_x);
    const StorageReal* __restrict py = frame_.span(kAtom_y);
    const StorageReal* __restrict pz = frame_.span(kAtom_z);
    x_.assign(px, px + n);
    y_.assign(py, py + n);
    z_.assign(pz, pz + n);
    if (n <= 1)
        return;
    if (frame_.box.length(0) <= 0.0 || frame_.box.length(1) <= 0.0
        || frame_.box.length(2) <= 0.0)
        return;
    StorageReal* __restrict sp_x = frame_.span(kAtom_x);
    StorageReal* __restrict sp_y = frame_.span(kAtom_y);
    StorageReal* __restrict sp_z = frame_.span(kAtom_z);
    const double x0 = sp_x[0], y0 = sp_y[0], z0 = sp_z[0];
    for (int i = 1; i < n; ++i) {
        double dx = sp_x[i] - x0, dy = sp_y[i] - y0, dz = sp_z[i] - z0;
        frame_.box.min_image(dx, dy, dz);
        sp_x[i] = x0 + dx;
        sp_y[i] = y0 + dy;
        sp_z[i] = z0 + dz;
    }
}

template <typename D>
MeanFieldMethod<D>::PositionGuard::~PositionGuard() {
    if (!active_)
        return;
    StorageReal* __restrict px = frame_.span(kAtom_x);
    StorageReal* __restrict py = frame_.span(kAtom_y);
    StorageReal* __restrict pz = frame_.span(kAtom_z);
    const int n = static_cast<int>(x_.size());
    for (int i = 0; i < n; ++i) {
        px[i] = x_[static_cast<std::size_t>(i)];
        py[i] = y_[static_cast<std::size_t>(i)];
        pz[i] = z_[static_cast<std::size_t>(i)];
    }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

template <typename D>
MeanFieldMethod<D>::MeanFieldMethod() = default;
template <typename D>
MeanFieldMethod<D>::~MeanFieldMethod() = default;
template <typename D>
MeanFieldMethod<D>::MeanFieldMethod(MeanFieldMethod&&) noexcept = default;
template <typename D>
MeanFieldMethod<D>&
MeanFieldMethod<D>::operator=(MeanFieldMethod&&) noexcept = default;

template <typename D>
void MeanFieldMethod<D>::set_predictor_(int order, int corrector_steps,
                                        double corrector_tol, bool fallback) {
    use_predictor_ = true;
    corrector_steps_ = corrector_steps;
    corrector_tol_ = corrector_tol;
    fallback_full_scf_ = fallback;
    aspc_alpha_ = std::make_unique<electronic::ASPCPredictor>(order);
    aspc_beta_ = std::make_unique<electronic::ASPCPredictor>(order);
}

// ── Stage wiring + Plan compile (lazy, once) ──────────────────────────────────

template <typename D>
void MeanFieldMethod<D>::reset_built_driver_() {
    // Drop the wired stages and every trace of their Plan / criterion
    // registration, so ensure_driver_built_ can rebind cleanly against the
    // current object. Reaches the loop-node internals directly (they are
    // protected precisely so a subclass can drive its own loop).
    fock_stage_.reset();
    rohf_couple_stage_.reset();
    mix_stage_.reset();
    diag_stage_.reset();
    density_stage_.reset();
    converged_stage_.reset();
    soscf_stepper_.reset();
    this->pending_.clear();
    this->owned_components_.clear();
    this->plan_ = {};
    this->criteria_.clear();
    this->criteria_owned_.clear();
    this->backend_.reset();
    driver_built_ = false;
}

template <typename D>
void MeanFieldMethod<D>::ensure_driver_built_() {
    if (driver_built_) {
        if (wired_self_ == this)
            return;
        // Relocated since it wired (a move after a run) — the stages point at
        // the moved-from object's members. Drop the stale wiring and rebind.
        reset_built_driver_();
    }

    // Wire each stage to this method's PLAINLY-NAMED members by reference (no
    // state bag); the discrete method config is captured by value now that all
    // knobs are finalized. restricted_ / is_hf_ are set by the caller
    // (drive_scf_) before this runs.
    //
    // ROHF (RestrictedOpen) is the same stage sequence with a hybrid wiring:
    // unrestricted Fock build → RohfCoupleStage (F_α/F_β → Roothaan F_eff in
    // the α channel; mix @ order −1) → single-channel DIIS on F_eff with the
    // P_tot commutator → restricted diagonalization (one MO set, C_β = C_α) →
    // unrestricted split-occupation density whose energy reads the preserved
    // TRUE spin Focks (never F_eff).
    // Generalized (GHF) is a single-channel loop at spinor dimension 2n:
    // FockStage/DensityStage take the explicit spinor branch, while the
    // mix / diagonalize / converged stages run their restricted
    // (single-channel) path unchanged — on the 2n-dimensional members with
    // the spinor metric S⊗1₂ (the DIIS error FPS − SPF is then exactly the
    // spinor commutator).
    const bool rohf = (spin_ == SpinType::RestrictedOpen);
    const bool generalized = (spin_ == SpinType::Generalized);
    // RI (density fitting) is wired for the RHF/UHF and RKS/UKS Fock
    // builds only. ROHF routes through the unrestricted Fock build but is
    // deliberately refused until its RI path is validated; GHF contracts
    // the stored dense ERI directly; the SOSCF stepper builds its own
    // exact-ERI Fock internally and would silently desynchronize from a
    // fitted loop. No silent fallback — throw with the reason.
    if (ri != RI::None) {
        if (rohf)
            throw std::runtime_error(
                "MeanFieldMethod: RI (density fitting) is not wired for "
                "ROHF in this link — use ri = RI::None with ROHF.");
        if (generalized)
            throw std::runtime_error(
                "MeanFieldMethod: RI (density fitting) is not wired for "
                "GHF (generalized spinor Fock uses the dense ERI) — use "
                "ri = RI::None with GHF.");
        if (soscf)
            throw std::runtime_error(
                "MeanFieldMethod: RI (density fitting) cannot be combined "
                "with soscf — the CIAH stepper builds exact-ERI Fock "
                "matrices internally and would be inconsistent with the "
                "fitted SCF loop.");
    }
    if (coulomb == Coulomb::CFMM && ri != RI::None)
        throw std::runtime_error(
            "MeanFieldMethod: Coulomb::CFMM is mutually exclusive with "
            "RI::J/JK — pick one J route (CFMM multipole or density fitting).");
    if (coulomb == Coulomb::CFMM && generalized)
        throw std::runtime_error(
            "MeanFieldMethod: Coulomb::CFMM is not wired for GHF — use "
            "coulomb = Coulomb::Direct with GHF.");
    fock_stage_ = std::make_unique<FockStage>(
        restricted_, is_hf_, xc_type_, grid_level, n_radial, n_angular,
        hybrid_alpha_override, fixed_grid, integral_view_(), P_alpha_, P_beta_,
        F_alpha_, F_beta_, J_, e_xc_grid_, e_exact_exchange_, int_n_alpha_,
        int_n_beta_, ri, aux_basis, generalized, pcm_.get(), e_solvent_,
        e_pcm_fock_half_, coulomb, cfmm_order, cfmm_threshold);
    if (rohf) {
        rohf_couple_stage_ = std::make_unique<RohfCoupleStage>(
            S_, P_alpha_, P_beta_, F_alpha_, F_beta_, F_alpha_uncoupled_,
            F_beta_uncoupled_, P_total_);
    }
    if (soscf) {
        soscf_stepper_ = std::make_unique<internal::SOSCFStepper>(
            restricted_, integral_view_(), n_alpha_, n_beta_, Hcore_, v_nuc_,
            F_alpha_, F_beta_, P_alpha_, P_beta_, C_alpha_, C_beta_, eig_alpha_,
            eig_beta_, soscf_max_micro, soscf_trust);
    }
    mix_stage_ = std::make_unique<MixStage>(
        restricted_ || rohf || generalized, use_diis, diis_start_iter,
        diis_max_vectors, level_shift, mixer, mixer_switch_tol, soscf,
        soscf_start, S_, X_, rohf ? P_total_ : P_alpha_, P_beta_, iter_,
        e_total_, F_alpha_, F_beta_, err_alpha_, err_beta_, soscf_active_);
    diag_stage_ = std::make_unique<DiagonalizeStage>(
        restricted_ || rohf || generalized, F_alpha_, F_beta_, S_, X_,
        n_dropped_, C_alpha_, C_beta_, eig_alpha_, eig_beta_,
        soscf_stepper_.get(), soscf_active_);
    density_stage_ = std::make_unique<DensityStage>(
        restricted_, is_hf_, damping_factor, C_alpha_, C_beta_,
        rohf ? F_alpha_uncoupled_ : F_alpha_,
        rohf ? F_beta_uncoupled_ : F_beta_, Hcore_, J_, n_alpha_, n_beta_,
        n_basis_, diis_active_, e_xc_grid_, e_exact_exchange_, v_nuc_, P_alpha_,
        P_beta_, e_total_, e_one_, e_coulomb_, generalized, e_solvent_,
        e_pcm_fock_half_, S_, mom, mom_ref_C_alpha_, mom_ref_C_beta_);
    converged_stage_ = std::make_unique<ScfConverged>(
        tol, (conv_tol_grad > 0) ? conv_tol_grad : std::sqrt(tol),
        restricted_ || rohf || generalized, e_total_, err_alpha_, err_beta_);

    this->on(ATV::scf::Phase::fock).add_ref(*fock_stage_);
    if (rohf)
        this->on(ATV::scf::Phase::mix)
            .add_ref(*rohf_couple_stage_,
                     /*order=*/-1);
    this->on(ATV::scf::Phase::mix).add_ref(*mix_stage_);
    this->on(ATV::scf::Phase::diagonalize).add_ref(*diag_stage_);
    this->on(ATV::scf::Phase::density).add_ref(*density_stage_);
    this->add_criterion_ref(*converged_stage_);

    // Whole-molecule serial evaluator — no spatial decomposition.
    this->set_serial_on_replica();
    this->build(); // Driver::build — compile the Plan (un-hidden via using).
    driver_built_ = true;
    wired_self_ = this;
}

// ── Nested driver lifecycle ───────────────────────────────────────────────────

template <typename D>
void MeanFieldMethod<D>::build_on(cpu::Runtime& rt) {
    // Stage wiring needs geometry (internal::IntegralWorkspace dimensions); deferred to the
    // first drive_scf_. Parent Runtime does not own this method's Plan.
    (void)rt;
    // The one thing a nested build CAN check: Car-Parrinello shares the
    // nuclear clock, and only a `TimestepProvider` in the same loop can
    // publish it. `cpu::md::MD::build()` refuses the missing provider itself;
    // this catches the loop that pushes nothing at all (Evaluate / Minimize),
    // where `cpmd_dt_fs_ = 0` would otherwise propagate the electrons by zero
    // and silently produce a Born-Oppenheimer-looking trajectory with a frozen
    // wavefunction.
    if (cpmd && !(cpmd_dt_fs_ > 0.0))
        throw std::runtime_error(
            "MeanFieldMethod: cpmd requires an MD integrator publishing the "
            "timestep — no timestep was pushed into this method (nest it at "
            "md::Phase::force in a cpu::md::MD loop whose integrator "
            "implements TimestepProvider).");
}

template <typename D>
void MeanFieldMethod<D>::destroy_on(cpu::Runtime& rt) {
    (void)rt;
    reset_built_driver_();
}

template <typename D>
int MeanFieldMethod<D>::run_nested_erased(cpu::Context& ctx, cpu::Frame& f,
                                          int budget) {
    // The ASPC predictor is MD-only, and this is the only place that can see
    // otherwise. `mc::Transaction::rollback` restores a rejected trial's
    // geometry bitwise but cannot notify an evaluator, so the rejected density
    // stays in the history — and the predictor then extrapolates a
    // time-continuous history across an accept/reject jump and corrects it at
    // the loosened `corrector_tol_`. The returned energy becomes a function of
    // configurations the chain never visited, so Metropolis is applied to
    // history-dependent energies and detailed balance breaks silently. (Plain
    // last-density reuse is fine: a guess does not survive convergence.)
    //
    // FIRST, before any SCF work — including the CPMD cold seed and XL-BOMD's
    // engagement below — so a refused configuration costs no solve.
    if (use_predictor_ && ctx.mc.proposal != nullptr)
        throw std::runtime_error(
            "MeanFieldMethod: the ASPC predictor is MD-only — rejected MC "
            "moves poison the density history and break detailed balance; "
            "disable predictor() for sampling");
    // XL-BOMD engages here and nowhere else: the auxiliary density lives on
    // the nuclear clock, so the nested entry is both the only place it may
    // propagate and the natural place to refuse a second guess owner (and a
    // dissipation order Table I does not have) before any SCF is spent.
    engage_xl_bomd_();
    if (cpmd) {
        // Production Car-Parrinello step: no SCF at all — a fixed-density
        // evaluation plus one electronic velocity-Verlet advance.
        if (cpmd_alpha_ && cpmd_alpha_->ready()) {
            cpmd_step_(f);
            return 0;
        }
        // Cold seed. Validate BEFORE any SCF work so a refused configuration
        // costs a driver build, and so the CPMD gate (not a downstream
        // force-path gate of the same knob) is the one that speaks.
        validate_cpmd_config_();
        DriveOpts opts;
        opts.unwrap = true;
        opts.nested = true;
        opts.forces = true; // CP nuclei need Born-Oppenheimer forces at t = 0
        opts.write_energy = true;
        opts.accumulate =
            true; // in-plan: the driver zeroed; we are a co-author
        if (budget > 0)
            opts.max_iter = budget;
        drive_scf_(f, nullptr, opts);
        if (!this->converged())
            throw std::runtime_error(
                "MeanFieldMethod: the Car-Parrinello seed SCF did not "
                "converge ("
                + this->stop_reason()
                + ") — CP propagates FROM a stationary state, and seeding off "
                  "it injects fictitious kinetic energy that never leaves.");
        cpmd_seed_();
        // The electrons have not moved yet: publish zeros so a consumer of
        // `kEnergyElectronKinetic` (cpu::md::CpmdConserved) sees the key from
        // the first step rather than throwing on the seed.
        last_te_ev_ = 0.0;
        last_etherm_ev_ = 0.0;
        publish_electronic_energies(f, last_te_ev_, last_etherm_ev_);
        return iterations_;
    }
    DriveOpts opts;
    opts.unwrap = true;
    opts.nested = true;
    // The outer loop's per-phase force policy, ANDed down the Context by the
    // Runtime: an energy-only phase (MC's trial `energy`) skips the whole
    // gradient — for a QM method the dominant cost of the sweep.
    opts.forces = compute_force && ctx.exec.forces_requested;
    opts.write_energy = true;
    // THE in-plan entry: the hosting driver owns zeroing (ZeroForces @ -1000,
    // ZeroEnergy @ -999), so every write below is `+=` and a classical
    // potential co-bound in the same phase keeps its contribution. Overwriting
    // here is what made "electronic structure x molecular mechanics" silently
    // drop whichever source ran first.
    opts.accumulate = true;
    if (budget > 0)
        opts.max_iter = budget;
    drive_scf_(f, nullptr, opts);
    return iterations_;
}

// ── XL-BOMD engagement: refuse a second guess owner, seat the propagators ────

template <typename D>
void MeanFieldMethod<D>::engage_xl_bomd_() {
    if (!xl_bomd) {
        // Turned off (or never on): drop any history so a later re-enable
        // starts a fresh trajectory instead of splicing across the gap.
        xl_alpha_.reset();
        xl_beta_.reset();
        return;
    }
    // Competing guess owners. ASPC extrapolates a polynomial through past
    // densities and CP propagates the coefficients, while XL propagates an
    // auxiliary density with its own equation of motion — each answers "what
    // is the density now", so running two means two answers. `GPWEvaluator`
    // refuses the same pairs.
    if (use_predictor_)
        throw std::invalid_argument(
            "MeanFieldMethod: xl_bomd and the ASPC density predictor are "
            "mutually exclusive (ASPC extrapolates past densities; XL-BOMD "
            "propagates an auxiliary density). Drop predictor(...) or clear "
            "xl_bomd.");
    if (cpmd)
        throw std::invalid_argument(
            "MeanFieldMethod: xl_bomd and cpmd are mutually exclusive "
            "(Car-Parrinello propagates the orbital coefficients and never "
            "re-converges the SCF that XL-BOMD would guess for). Clear one.");

    // Seat the propagators (the constructor is the Table-I lookup, so an
    // unsupported dissipation order throws here — before the first SCF —
    // rather than after the first converged solve). A changed `xl_bomd_k`
    // re-seats: the history belongs to one recursion, not two.
    const bool dual_channel = (spin_ == SpinType::Unrestricted)
                              || (spin_ == SpinType::RestrictedOpen);
    if (!xl_alpha_ || xl_alpha_->order() != xl_bomd_k) {
        xl_alpha_.emplace(xl_bomd_k);
        if (dual_channel)
            xl_beta_.emplace(xl_bomd_k);
        else
            xl_beta_.reset();
    }
}

// ── THE funnel ────────────────────────────────────────────────────────────────

template <typename D>
void MeanFieldMethod<D>::drive_scf_(Frame& frame, const SCFInitialGuess* guess,
                                    const DriveOpts& opts) {
    // ECP ↔ x2c double-counting gate (per-element): relativistic ECPs bake
    // scalar relativity into their parameterization, so an atom that hits
    // the ECP table must not additionally be treated with x2c. All-electron
    // molecules (no table hit) pass — the gate is per-element, not global.
    if (x2c && !ecp.empty()) {
        const auto& table = EcpPotential::family_table(ecp);
        const int natm = static_cast<int>(frame.nlocal());
        const int* __restrict sp_type = frame.span(kAtom_type);
        for (int i = 0; i < natm; ++i) {
            const int Z = sp_type[i];
            if (Z >= 1 && Z < static_cast<int>(table.size())
                && !table[static_cast<std::size_t>(Z)].channels.empty()) {
                throw std::runtime_error(
                    "MeanFieldMethod::run: x2c=true combined with ecp='" + ecp
                    + "' would double-count scalar relativity for "
                      "element "
                    + EcpPotential::element_symbol(Z)
                    + " (Z=" + std::to_string(Z)
                    + "): the relativistic ECP already includes it. Disable "
                      "x2c or run this element all-electron.");
            }
        }
    }

    // ── THE GUESS LADDER ──────────────────────────────────────────────────
    //
    // One ordered resolution, highest rung wins, nothing below it runs:
    //
    //   1. the caller's explicit guess   — a caller who hands run() a density
    //      has a reason (a chosen SCF branch, a continuous finite-difference
    //      path), so no internal warm state may silently displace it;
    //   2. the XL-BOMD auxiliary density — a dynamical variable on the nuclear
    //      clock, available once its history is seeded;
    //   3. the ASPC extrapolation        — a polynomial through the recorded
    //      densities, available once its history is long enough;
    //   4. the previous run's converged density (plain warm start).
    //
    // A prediction that loses is also not *corrected*: `predicted` stays false,
    // so rung 1 takes the same plain full solve a cold method would take.
    SCFInitialGuess density_guess;
    const SCFInitialGuess* effective_guess = guess;
    bool predicted = false;

    // Rung 1. `apply_aspc == false` is the corrector/fallback recursion, which
    // already carries the guess this ladder resolved for it.
    bool resolved = (guess != nullptr) || !opts.apply_aspc;

    // Rung 2. Restricted / GHF carry one channel (the factor-2 total, resp.
    // spinor, density); UHF/UKS/ROHF/ROKS carry α and β.
    if (!resolved && xl_alpha_ && xl_alpha_->ready()) {
        density_guess.P_alpha = xl_alpha_->aux();
        density_guess.P_beta = (xl_beta_ && xl_beta_->ready())
                                   ? xl_beta_->aux()
                                   : xl_alpha_->aux();
        density_guess.has_alpha = true;
        density_guess.has_beta = true;
        effective_guess = &density_guess;
        resolved = true;
    }

    // Rung 3. Both channels are polled (predict has no side effect and the
    // pair is used or dropped together).
    if (!resolved && use_predictor_ && aspc_alpha_ && aspc_beta_) {
        xt::xtensor<double, 2> Pa_guess, Pb_guess;
        const bool have_alpha = aspc_alpha_->predict(Pa_guess);
        const bool have_beta = aspc_beta_->predict(Pb_guess);
        if (have_alpha && have_beta) {
            density_guess.P_alpha = std::move(Pa_guess);
            density_guess.P_beta = std::move(Pb_guess);
            density_guess.has_alpha = true;
            density_guess.has_beta = true;
            effective_guess = &density_guess;
            predicted = true;
            resolved = true;
        }
    }

    // Rung 4. The previous outer-step density (available with or without ASPC).
    if (!resolved && has_last_density_) {
        density_guess.P_alpha = last_P_alpha_;
        density_guess.P_beta = last_P_beta_;
        density_guess.has_alpha = true;
        density_guess.has_beta = true;
        effective_guess = &density_guess;
    }

    if (opts.apply_aspc && use_predictor_ && predicted) {
        // ASPC corrector: short SCF at loosened tol; fall back to full solve.
        DriveOpts copts = opts;
        copts.max_iter = corrector_steps_;
        copts.tol_override = corrector_tol_;
        copts.apply_aspc = false;
        // THE ONE path that runs two epilogues per entry. Under `accumulate`
        // (the in-plan entry) the corrector's contribution is *discarded* when
        // the fallback re-solves, and discarding an accumulation means undoing
        // it: leaving it in place would add this method twice to the
        // driver-owned accumulators (~2x the electronic energy and forces on
        // every ASPC step whose corrector misses tol). Snapshot the channels
        // the epilogue writes and restore them before the re-solve, so exactly
        // one contribution ever lands — and restore rather than subtract, so
        // the result is bit-identical to the plain full solve. The overwrite
        // entries need no guard: the second `=` erases the first.
        const bool guard_accumulators = opts.accumulate && fallback_full_scf_;
        int natm = 0;
        double e_before = 0.0;
        double e_electronic_before = 0.0;
        std::vector<AccReal> f_before;
        if (guard_accumulators) {
            // Park the resolution window: this guard's lookups are bookkeeping
            // and must not spend the hosting component's per-launch budget
            // (same pattern as the SCF loop below and PWEvaluator).
            const auto saved_window = frame.resolution_window();
            natm = static_cast<int>(frame.nlocal());
            e_before = frame.meta.get(kEnergy);
            e_electronic_before = frame.meta.get(kEnergyElectronic);
            if (opts.forces) {
                const AccReal* __restrict fx = frame.span(kAtom_fx);
                const AccReal* __restrict fy = frame.span(kAtom_fy);
                const AccReal* __restrict fz = frame.span(kAtom_fz);
                f_before.resize(static_cast<std::size_t>(3 * natm));
                for (int A = 0; A < natm; ++A) {
                    f_before[static_cast<std::size_t>(3 * A) + 0] = fx[A];
                    f_before[static_cast<std::size_t>(3 * A) + 1] = fy[A];
                    f_before[static_cast<std::size_t>(3 * A) + 2] = fz[A];
                }
            }
            frame.restore_resolution_window(saved_window);
        }
        drive_scf_core_(frame, effective_guess, copts);
        if (!this->converged() && fallback_full_scf_) {
            if (guard_accumulators) {
                const auto saved_window = frame.resolution_window();
                frame.meta.get(kEnergy) = e_before;
                frame.meta.get(kEnergyElectronic) = e_electronic_before;
                if (opts.forces) {
                    AccReal* __restrict fx = frame.span(kAtom_fx);
                    AccReal* __restrict fy = frame.span(kAtom_fy);
                    AccReal* __restrict fz = frame.span(kAtom_fz);
                    for (int A = 0; A < natm; ++A) {
                        fx[A] = f_before[static_cast<std::size_t>(3 * A) + 0];
                        fy[A] = f_before[static_cast<std::size_t>(3 * A) + 1];
                        fz[A] = f_before[static_cast<std::size_t>(3 * A) + 2];
                    }
                }
                frame.restore_resolution_window(saved_window);
            }
            DriveOpts full = opts;
            full.apply_aspc = false;
            drive_scf_core_(frame, effective_guess, full);
        }
    } else {
        DriveOpts core = opts;
        core.apply_aspc = false;
        drive_scf_core_(frame, effective_guess, core);
    }

    // Converged densities live on the plainly-named members; the restricted
    // alpha member carries the factor-2 total density (old result convention).
    const RealMatrix& P_alpha = P_alpha_;
    const RealMatrix& P_beta = restricted_ ? P_alpha_ : P_beta_;

    // XL-BOMD: exactly one auxiliary propagation per converged nested run —
    // the recursion is integrating the nuclear trajectory, so a standalone
    // single point (no nuclear step) must not enter it, and an unconverged
    // step has no density worth propagating (its residual would leak into
    // every later guess). Both cases match the GPW evaluator's treatment.
    if (opts.apply_aspc && opts.nested && xl_alpha_) {
        if (this->converged()) {
            xl_alpha_->advance(P_alpha);
            if (xl_beta_)
                xl_beta_->advance(P_beta);
            ++xl_advances_;
        } else {
            xl_alpha_->reset();
            if (xl_beta_)
                xl_beta_->reset();
        }
    }

    if (opts.apply_aspc && use_predictor_ && aspc_alpha_ && aspc_beta_) {
        if (predicted) {
            xt::xtensor<double, 2> Pa_corr, Pb_corr;
            aspc_alpha_->correct(P_alpha, density_guess.P_alpha, Pa_corr);
            aspc_beta_->correct(P_beta, density_guess.P_beta, Pb_corr);
            aspc_alpha_->record(Pa_corr);
            aspc_beta_->record(Pb_corr);
        } else {
            aspc_alpha_->record(P_alpha);
            aspc_beta_->record(P_beta);
        }
    }
    if (opts.apply_aspc) {
        has_last_density_ = true;
        last_P_alpha_ = P_alpha;
        last_P_beta_ = P_beta;
    }
}

// @brief Rebuild private integral fields at the current geometry and resolve
// occupation — see the header for why PositionGuard and ghost/x2c guards stay
// with the caller. Fill still uses a transitional temporary storage type.
template <typename D>
internal::IntegralView
MeanFieldMethod<D>::rebuild_integrals_(const Frame& frame, bool compute_pvp) {
    // Fill into a temporary, then move into method-owned **fields** (no bag kept).
    internal::IntegralWorkspace fill{
        frame, basis, charge, compute_eri, ecp, ghost_atoms, compute_pvp};
    if constexpr (requires { static_cast<const D*>(this)->periodic; }) {
        if (static_cast<const D*>(this)->periodic) {
            if (method_ != MeanFieldType::HF || spin_ != SpinType::Restricted)
                throw std::invalid_argument(
                    "periodic=true is Γ-point closed-shell RHF only — "
                    "use cpu::scf::RHF and clear periodic on KS/UHF");
            if (!frame.box.is_orthorhombic())
                throw std::invalid_argument(
                    "RHF.periodic=true requires an orthorhombic Frame.box");
            if (fill.spherical())
                throw std::invalid_argument(
                    "RHF.periodic=true v1 is Cartesian only (e.g. "
                    "BasisSet::STO3G)");
            const double L[3] = {
                units::convert(frame.box.length(0), units::Length::Angstrom,
                               units::Length::Bohr),
                units::convert(frame.box.length(1), units::Length::Angstrom,
                               units::Length::Bohr),
                units::convert(frame.box.length(2), units::Length::Angstrom,
                               units::Length::Bohr),
            };
            const auto spec = internal::lattice_spec(
                L, internal::shell_pair_rcut(fill.shells));
            if (spec.has_extra()) {
                const auto blocks = blocks_from_shells(fill.shells);
                const auto n = static_cast<std::size_t>(fill.n_basis);
                fill.ERI = xt::zeros<double>({n, n, n, n});
                internal::fill_eri_lattice(blocks, spec, fill.ERI);
                internal::apply_gamma_images(fill, spec);
            }
        }
    }
    ints_S_ = std::move(fill.S);
    ints_T_ = std::move(fill.T);
    ints_V_ = std::move(fill.V);
    ints_Hcore_ = std::move(fill.Hcore);
    ints_W_ = std::move(fill.W);
    ints_ERI_ = std::move(fill.ERI);
    ints_V_nuc_ = fill.V_nuc;
    ints_n_basis_ = fill.n_basis;
    ints_n_basis_cart_ = fill.n_basis_cart;
    ints_n_electrons_ = fill.n_electrons;
    ints_ecp_ = std::move(fill.ecp);
    ints_has_ecp_ = fill.has_ecp;
    ints_ecp_centers_ = std::move(fill.ecp_centers);
    ints_embed_charges_.clear();
    ints_embed_self_energy_ = true;
    ints_external_field_ = {{0.0, 0.0, 0.0}};
    ints_solvent_eps_ = 0.0;
    ints_solvent_radii_scale_ = fill.solvent_radii_scale;
    ints_solvent_x_ = fill.solvent_x;
    ints_solvent_model_ = fill.solvent_model;
    ints_atoms_ = std::move(fill.atoms);
    ints_shell_data_ = std::move(fill.shell_data);
    *ints_shells_sp_ = std::move(fill.shells);
    for (std::size_t i = 0; i < ints_shells_sp_->size(); ++i) {
        auto& sh = (*ints_shells_sp_)[i];
        const int np = sh.n_prim;
        sh.alpha = ints_shell_data_[i].data();
        sh.coeff = ints_shell_data_[i].data() + np;
    }
    *ints_c2s_sp_ = std::move(fill.c2s);
    resolve_occupation(ints_n_electrons_);
    return integral_view_();
}

template <typename D>
void MeanFieldMethod<D>::drive_scf_core_(Frame& frame,
                                         const SCFInitialGuess* guess,
                                         const DriveOpts& opts) {
    // ── Prologue: per-run assembly ────────────────────────────────────────────
    PositionGuard guard(frame, opts.unwrap);

    restricted_ = (spin_ == SpinType::Restricted);
    is_hf_ = (method_ == MeanFieldType::HF);
    const bool restricted = restricted_;
    const bool is_hf = is_hf_;

    // Ghost atoms are HF-only in this link: KS grids key atomic radii off Z,
    // and a ghost Z=0 would silently pick the wrong radius. Refuse — no
    // fallback (counterpoise for KS is a follow-up chain link).
    if (!ghost_atoms.empty() && !is_hf)
        throw std::invalid_argument(
            "MeanFieldMethod::run: ghost_atoms is non-empty but the method is "
            "Kohn-Sham (RKS/UKS) — ghost / counterpoise support is pure HF "
            "only in this link. Use RHF/UHF, or clear ghost_atoms.");

    const internal::IntegralView ints =
        rebuild_integrals_(frame, /*compute_pvp=*/x2c);
    // x2c-1e: replace the non-relativistic core Hamiltonian with the
    // one-step exact-two-component h^X2C. S / ERI / Fock loop are untouched
    // (two-electron integrals stay non-relativistic by definition of
    // x2c-1e). W was assembled only when x2c=true.
    if (x2c) {
        // RealMatrix is column-major; internal::IntegralWorkspace stores row-major
        // xtensor — copy into RealMatrix so the workspace binds cleanly.
        const RealMatrix S = ints_S_;
        const RealMatrix T = ints_T_;
        const RealMatrix V = ints_V_;
        const RealMatrix W = ints_W_;
        const internal::X2cWorkspace x2c_ws(S, T, V, W,
                                            internal::kLightSpeedAU);
        ints_Hcore_ = x2c_ws.hcore();
    }

    // SOSCF is HF-only in this link: the CIAH Hessian products carry no XC
    // second-order kernel, so a Kohn-Sham run would silently Newton-step the
    // wrong surface. Refuse — no fallback.
    if (soscf && !is_hf)
        throw std::invalid_argument(
            "MeanFieldMethod::run: soscf=true is supported for HF (RHF/UHF) "
            "only — the Kohn-Sham orbital Hessian needs the XC second-order "
            "kernel, which this SOSCF link does not include. Disable soscf "
            "or run RHF/UHF.");

    // GHF-unsupported knobs — explicit refusal instead of a silently wrong
    // Restricted/Unrestricted branch (no fallback). Analytic forces are
    // supported (spinor W + unsymmetrized four-block exchange); see
    // ScfGradient and accumulate_two_electron_gradient.
    const bool generalized = (spin_ == SpinType::Generalized);
    if (generalized) {
        if (soscf)
            throw std::invalid_argument(
                "MeanFieldMethod::run: soscf=true is not supported for GHF "
                "— the CIAH orbital Hessian has no spinor rotation blocks. "
                "Disable soscf.");
        if (std::abs(level_shift) > internal::kLevelShiftSkipThreshold)
            throw std::invalid_argument(
                "MeanFieldMethod::run: level_shift is not supported for GHF "
                "— the restricted shift projector 0.5*P is not the spinor "
                "occupied projector. Set level_shift = 0.");
        if (mixer != MixerKind::CDIIS)
            throw std::invalid_argument(
                "MeanFieldMethod::run: EDIIS/ADIIS mixers are not validated "
                "for the GHF spinor loop — use MixerKind::CDIIS.");
        if (use_predictor_)
            throw std::invalid_argument(
                "MeanFieldMethod::run: the ASPC density predictor is not "
                "supported for GHF in this link.");
        if (init_guess != InitGuess::Hcore)
            throw std::invalid_argument(
                "MeanFieldMethod::run: GHF supports InitGuess::Hcore only — "
                "the MINAO/Hueckel/SAP guesses are spin-balanced, which "
                "makes the collinearity-breaking spin rotation a no-op "
                "(silent UHF-saddle trap). Keep the GHF default.");
        if (!compute_eri)
            throw std::invalid_argument(
                "MeanFieldMethod::run: GHF requires compute_eri = true — "
                "the spinor exchange K[P^ab] of the non-symmetric "
                "spin-off-diagonal block needs the stored dense ERI tensor "
                "(no direct-SCF spinor path in this link).");
    }

    // x2c-1e analytic forces are not derived (picture-change of the
    // nuclear-gradient operator + dX/dR of the decoupling matrix). Refuse
    // rather than return a non-relativistic force under an x2c energy.
    if (opts.forces && x2c)
        throw std::runtime_error(
            "MeanFieldMethod::forces: analytic forces are not implemented "
            "for x2c=true — the h^X2C nuclear gradient (picture-change + "
            "decoupling-matrix response) is a follow-up chain link. Disable "
            "x2c or use finite differences.");

    // ROHF-unsupported knobs — explicit refusal instead of a silently wrong
    // Restricted/Unrestricted branch (no fallback).
    const bool rohf = (spin_ == SpinType::RestrictedOpen);
    if (rohf) {
        if (soscf)
            throw std::invalid_argument(
                "MeanFieldMethod::run: soscf=true is not supported for ROHF "
                "— the CIAH orbital Hessian does not include the Roothaan "
                "coupling terms. Disable soscf.");
        if (std::abs(level_shift) > internal::kLevelShiftSkipThreshold)
            throw std::invalid_argument(
                "MeanFieldMethod::run: level_shift is not supported for ROHF "
                "— the restricted shift projector 0.5*P_tot is not the "
                "open-shell occupied projector. Set level_shift = 0.");
    }
    if (mom != MomStyle::Off) {
        if (spin_ == SpinType::Generalized || rohf)
            throw std::invalid_argument(
                "MeanFieldMethod::run: mom != Off is not supported for ROHF/"
                "GHF in this link — use RHF/UHF/RKS/UKS with set_mom_reference "
                "or UHF::delta_scf");
        if (mom_ref_C_alpha_.size() == 0)
            throw std::invalid_argument(
                "MeanFieldMethod::run: mom != Off requires set_mom_reference "
                "(or UHF::delta_scf) before run — empty MOM reference");
        if (static_cast<int>(mom_ref_C_alpha_.shape()[1]) != n_alpha_
            || (!restricted_
                && static_cast<int>(mom_ref_C_beta_.shape()[1]) != n_beta_))
            throw std::invalid_argument(
                "MeanFieldMethod::run: MOM reference occupation count does "
                "not match n_alpha/n_beta (got C_ref_a cols="
                + std::to_string(mom_ref_C_alpha_.shape()[1])
                + ", n_alpha=" + std::to_string(n_alpha_) + ")");
        if (static_cast<int>(mom_ref_C_alpha_.shape()[0]) != ints.n_basis)
            throw std::invalid_argument(
                "MeanFieldMethod::run: MOM reference AO dimension "
                + std::to_string(mom_ref_C_alpha_.shape()[0]) + " != n_basis "
                + std::to_string(ints.n_basis));
    }
    const int n = ints.n_basis;
    const int n_alpha = n_alpha_;
    const int n_beta = n_beta_;

    // Occupation validity guard (no-fallback discipline).
    if (n_alpha < 0 || n_beta < 0 || n_alpha + n_beta != ints.n_electrons
        || (restricted && n_alpha != n_beta)) {
        throw std::invalid_argument(
            "MeanFieldMethod::run: inconsistent occupation configuration "
            "(n_electrons="
            + std::to_string(ints.n_electrons) + ", n_alpha="
            + std::to_string(n_alpha) + ", n_beta=" + std::to_string(n_beta)
            + "): requires n_alpha >= 0, n_beta >= 0, n_alpha + n_beta == "
              "n_electrons, and n_alpha == n_beta when Restricted. Check the "
              "molecular charge / spin multiplicity.");
    }

    // Environment: point charges, external field, and C-PCM all have force
    // paths in ScfGradient::compute (embed/field = semi-numerical 1e +
    // classical; C-PCM = vacuum HF terms on the solvent density + fixed-P
    // finite difference of ΔG_el over the rebuilt cavity).
    const bool field_active = std::abs(external_field[0]) > 0.0
                              || std::abs(external_field[1]) > 0.0
                              || std::abs(external_field[2]) > 0.0;
    const bool solvent_active = solvent_eps > 0.0;

    // Apply environment to the n-dimensional Hcore / classical energy
    // *before* any GHF spinor embed so both spin blocks share the same
    // one-electron environment (Frame-pure: env is never parked on Frame).
    RealMatrix H = ints.Hcore;
    double v_nuc = ints.V_nuc;
    if (!point_charges.empty()) {
        const internal::PointChargeEmbedding emb(point_charges);
        RealMatrix Venv = emb.core_matrix(ints.shells);
        if (ints.spherical()) {
            const auto ct = xt::transpose(ints.c2s);
            Venv = xt::linalg::dot(ct, xt::linalg::dot(Venv, ints.c2s));
        }
        H += Venv;
        v_nuc += emb.nuclear_energy(ints.atoms);
        if (embed_self_energy)
            v_nuc += emb.self_energy();
        // Retain charges + self-energy gate for the analytic gradient.
        ints_embed_charges_ = point_charges;
        ints_embed_self_energy_ = embed_self_energy;
    }
    if (field_active) {
        const internal::ExternalField field(external_field, ints.atoms);
        RealMatrix Vf = field.core_matrix(ints.shells);
        if (ints.spherical()) {
            const auto ct = xt::transpose(ints.c2s);
            Vf = xt::linalg::dot(ct, xt::linalg::dot(Vf, ints.c2s));
        }
        H += Vf;
        v_nuc += field.nuclear_energy(ints.atoms);
        // Retain the field on the pack so the analytic gradient can rebuild
        // dV_field/dR + the nuclear field-energy derivative.
        ints_external_field_ = external_field;
    }

    // Publish the environment-augmented Hcore back onto the private pack so
    // FockStage / HamiltonianBuilder (which read `integrals.Hcore`, not the
    // method member) build F = H_env + G. Energy uses Hcore_ + v_nuc_ below;
    // if only the member were updated, ½Tr[(H_env+F_vac)P] would undercount
    // the field/embedding by exactly ½.
    ints_Hcore_ = H;
    ints_V_nuc_ = v_nuc;

    // C-PCM: private reaction-field workspace. Rebuilt every run (geometry
    // may have moved). solvent_eps == 0 keeps pcm_ null so FockStage is a
    // pure vacuum path. FockStage captures pcm_.get() at wire time, so any
    // rebuild of pcm_ (or a toggle on/off) must re-wire the stages.
    e_solvent_ = 0;
    e_pcm_fock_half_ = 0;
    const bool had_pcm = (pcm_ != nullptr);
    if (solvent_active) {
        if (generalized)
            throw std::invalid_argument(
                "MeanFieldMethod::run: solvent_eps > 0 is not supported for "
                "GHF in this link — clear solvent_eps or use RHF/UHF/RKS/UKS.");
        pcm_ = std::make_unique<internal::Pcm>(
            ints.atoms, solvent_eps, solvent_radii_scale, solvent_x,
            /*lebedev_n=*/302, solvent_model);
        if (ints.spherical())
            pcm_->prepare(ints.shells, ints.c2s);
        else
            pcm_->prepare(ints.shells);
        // Retain solvent parameters so ScfGradient can rebuild the cavity at
        // displaced geometries for ∂ΔG_el/∂R at fixed density.
        ints_solvent_eps_ = solvent_eps;
        ints_solvent_radii_scale_ = solvent_radii_scale;
        ints_solvent_x_ = solvent_x;
        ints_solvent_model_ = solvent_model;
    } else {
        pcm_.reset();
        ints_solvent_eps_ = 0.0;
        ints_solvent_radii_scale_ = solvent_radii_scale;
        ints_solvent_x_ = solvent_x;
        ints_solvent_model_ = solvent_model;
    }
    // Rebind whenever the solvent workspace is present (pointer always new)
    // or when it was dropped after a previous solvent run.
    if (driver_built_ && (solvent_active || had_pcm))
        reset_built_driver_();

    // Wire stages after pcm_ is ready so FockStage's non-owning pointer is
    // live for the whole SCF loop.
    ensure_driver_built_();

    // n-dimensional Hcore (with environment) kept for the GHF n-dim aufbau
    // seed below; the loop members may then be spinor-embedded.
    const RealMatrix& H_ao = ints_Hcore_;

    // Generalized (GHF): the loop members live at spinor dimension 2n. The
    // spin-independent one-electron operators are embedded block-diagonally
    // (S⊗1₂, h⊗1₂ in the blocked [α-rows; β-rows] layout); the AO integrals
    // themselves stay n-dimensional inside the FockStage spinor branch.
    if (generalized) {
        S_ = internal::DensityBuilder::spinor_embed(ints.S);
        Hcore_ = internal::DensityBuilder::spinor_embed(H_ao);
    } else {
        S_ = ints.S;
        Hcore_ = H_ao;
    }
    n_basis_ = n;
    v_nuc_ = v_nuc;
    diis_pairs_ = 0;
    e_xc_grid_ = 0;
    e_exact_exchange_ = 0;
    int_n_alpha_ = 0;
    int_n_beta_ = 0;

    // Orthogonalization matrix X + linear-dependence handling —
    // canonical-orth path only when columns drop.
    internal::EigenSolver eigen;
    internal::DensityBuilder density;
    n_dropped_ = 0;
    if (lindep_threshold > 0) {
        linalg::CanonicalOrth ortho(S_, lindep_threshold);
        n_dropped_ = ortho.n_dropped;
        X_ = (ortho.n_dropped > 0) ? std::move(ortho.X)
                                   : std::move(ortho.lowdin);
    } else {
        X_ = linalg::build_orthogonalization_matrix(S_);
    }

    auto diagonalize = [&](const RealMatrix& F) {
        return (n_dropped_ > 0) ? eigen.step_orthogonal(F, X_)
                                : eigen.step(F, S_);
    };

    if (generalized) {
        // GHF initial guess: the noncollinear-magnetism standard "charge
        // density + per-site moment directions" construction (the shape of
        // a plane-wave code's MAGMOM/SAXIS guess). The spin-scalar charge
        // half of an n-dimensional aufbau reference is embedded unrotated
        // (½P_c⊗1₂ — rotating charge would leak spurious spin-off-diagonal
        // weight), and each AO μ is decorated with a full-strength local
        // moment of magnitude d_μ = clamp(P_c(μ,μ), 0, 2−P_c(μ,μ)) pointing
        // along (sin θ_μ, 0, cos θ_μ) in the real-algebra x-z spin plane,
        // θ_μ = guess_spin_rotation·μ. θ = 0 keeps the guess collinear —
        // DIIS then converges onto a collinear UHF stationary point (a
        // saddle on spin-frustrated systems); this trap is test-pinned.
        const int N = n_alpha; // resolve_occupation put the total N here
        const int n_a = (N + 1) / 2;
        const int n_b = N / 2;
        const bool valid_spinor_guess =
            guess && guess->has_alpha && guess->P_alpha.dimension() == 2
            && static_cast<int>(guess->P_alpha.shape()[0]) == 2 * n
            && static_cast<int>(guess->P_alpha.shape()[1]) == 2 * n;
        RealMatrix P0;
        if (valid_spinor_guess) {
            P0 = 0.5 * (guess->P_alpha + xt::transpose(guess->P_alpha));
        } else {
            // n-dim aufbau charge density (own n-dim orthogonalization —
            // the member X_ is spinor).
            RealMatrix Xn;
            if (lindep_threshold > 0) {
                linalg::CanonicalOrth ortho_n(ints.S, lindep_threshold);
                Xn = (ortho_n.n_dropped > 0) ? std::move(ortho_n.X)
                                             : std::move(ortho_n.lowdin);
            } else {
                Xn = linalg::build_orthogonalization_matrix(ints.S);
            }
            auto [eig_n, C_n] = eigen.step_orthogonal(H_ao, Xn);
            (void)eig_n;
            const RealMatrix P_c =
                density.step(C_n, n_a, n) + density.step(C_n, n_b, n);
            P0 = xt::zeros<double>({2 * n, 2 * n});
            xt::view(P0, xt::range(0, n), xt::range(0, n)) = 0.5 * P_c;
            xt::view(P0, xt::range(n, 2 * n), xt::range(n, 2 * n)) = 0.5 * P_c;

            double theta = 0.0;
            if constexpr (std::is_same_v<D, GHF>)
                theta = static_cast<D*>(this)->guess_spin_rotation;
            for (int mu = 0; mu < n; ++mu) {
                const double d =
                    std::min(std::max(P_c(mu, mu), 0.0), 2.0 - P_c(mu, mu));
                const double th = theta * mu;
                const double mz = 0.5 * d * std::cos(th);
                const double mx = 0.5 * d * std::sin(th);
                P0(mu, mu) += mz;
                P0(n + mu, n + mu) -= mz;
                P0(mu, n + mu) += mx;
                P0(n + mu, mu) += mx;
            }
        }
        P_alpha_ = std::move(P0);
        P_beta_ = P_alpha_;
        // Placeholders only: the diagonalize stage overwrites C/ε before the
        // density stage ever reads them (fock → mix → diagonalize → density).
        C_alpha_ = xt::zeros<double>({2 * n, 2 * n});
        C_beta_ = C_alpha_;
        eig_alpha_ = xt::zeros<double>({2 * n});
        eig_beta_ = eig_alpha_;
    } else {
        // Initial guess, reusing diagonalize + density once.
        auto [eig0, C0] = diagonalize(Hcore_);

        RealMatrix P_alpha, P_beta;
        const bool valid_guess_alpha =
            guess && guess->has_alpha && guess->P_alpha.dimension() == 2
            && static_cast<int>(guess->P_alpha.shape()[0]) == n
            && static_cast<int>(guess->P_alpha.shape()[1]) == n;
        const bool valid_guess_beta =
            guess && guess->has_beta && guess->P_beta.dimension() == 2
            && static_cast<int>(guess->P_beta.shape()[0]) == n
            && static_cast<int>(guess->P_beta.shape()[1]) == n;
        const bool want_minao = (init_guess == InitGuess::MINAO)
                                && !valid_guess_alpha && minao_supported(ints);
        RealMatrix P_minao;
        if (want_minao)
            P_minao = build_minao_density(ints);

        // Hückel / SAP: diagonalize a guess Fock instead of Hcore, then flow
        // through the same orbital-based density branches below (open shell
        // keeps the existing split + HOMO-LUMO mixing rule — both guesses are
        // spin-symmetric; see the guess headers for the averaging limitation).
        const bool want_hueckel =
            (init_guess == InitGuess::Hueckel) && !valid_guess_alpha;
        const bool want_sap =
            (init_guess == InitGuess::SAP) && !valid_guess_alpha;
        if ((want_hueckel || want_sap) && ints.has_ecp)
            throw std::runtime_error(
                "MeanFieldMethod::run: the Hueckel/SAP initial guesses are not "
                "available with an ECP (the pack stores valence-reduced Z, so "
                "the atomic tables would resolve the wrong element). Use "
                "InitGuess::MINAO or InitGuess::Hcore.");
        if (want_hueckel) {
            internal::HueckelGuess hueckel(ints.shells, ints.S, ints.atoms);
            std::tie(eig0, C0) = diagonalize(hueckel.F);
        } else if (want_sap) {
            const MolecularGrid sap_grid(ints.atoms.data(),
                                         static_cast<int>(ints.atoms.size()),
                                         grid_level, n_radial, n_angular);
            internal::SapGuess sap(ints.shells, ints.T, ints.atoms, sap_grid,
                                   ints.c2s);
            std::tie(eig0, C0) = diagonalize(sap.F);
        }

        if (restricted && valid_guess_alpha) {
            P_alpha = 0.5 * (guess->P_alpha + xt::transpose(guess->P_alpha));
            P_beta = P_alpha;
        } else if (!restricted && valid_guess_alpha && valid_guess_beta) {
            P_alpha = 0.5 * (guess->P_alpha + xt::transpose(guess->P_alpha));
            P_beta = 0.5 * (guess->P_beta + xt::transpose(guess->P_beta));
        } else if (want_minao && restricted) {
            P_alpha = P_minao;
            P_beta = P_alpha;
        } else if (want_minao) {
            P_alpha = 0.5 * P_minao;
            P_beta = 0.5 * P_minao;
            const bool is_ks = (method_ == MeanFieldType::KS);
            // ROHF never wants the UHF symmetry-breaking guess perturbation: the
            // coupled F_eff restores one MO set anyway, so start from the plain
            // Aufbau split (PySCF-matching).
            const bool skip_mixing = (is_ks && (n_alpha == n_beta)) || rohf;
            if (method_ == MeanFieldType::HF && n_alpha > 0 && n_beta > 0
                && !skip_mixing && n_alpha != n_beta) {
                RealMatrix C_alpha_init = C0;
                apply_homo_lumo_mixing(C_alpha_init, n_alpha, n);
                P_alpha = density.step(C_alpha_init, n_alpha, n);
                P_beta = density.step(C0, n_beta, n);
            }
        } else if (restricted) {
            P_alpha = density.step(C0, n_alpha, n, double(2));
            P_beta = P_alpha;
        } else {
            RealMatrix C_alpha_init = C0;
            const bool is_ks = (method_ == MeanFieldType::KS);
            // See the MINAO branch: ROHF skips the UHF symmetry-breaking guess.
            const bool skip_mixing = (is_ks && (n_alpha == n_beta)) || rohf;
            if (n_alpha > 0 && n_beta > 0 && !skip_mixing)
                apply_homo_lumo_mixing(C_alpha_init, n_alpha, n);
            P_alpha = density.step(C_alpha_init, n_alpha, n);
            P_beta = density.step(C0, n_beta, n);
        }

        P_alpha_ = std::move(P_alpha);
        P_beta_ = std::move(P_beta);
        C_alpha_ = C0;
        C_beta_ = C0;
        eig_alpha_ = eig0;
        eig_beta_ = eig0;
    } // end !generalized guess flow

    mix_stage_->reset();
    converged_stage_->reset();
    soscf_active_ = false; // per-run SOSCF latch + stepper state
    if (soscf_stepper_)
        soscf_stepper_->reset();
    // Retarget the criterion for this run: a loosened threshold for the ASPC
    // corrector (opts.tol_override), else the method's own tol — restoring the
    // old corrector_loop's tolerance behavior without a second criterion.
    const double eff_tol = (opts.tol_override > 0) ? opts.tol_override : tol;
    const double eff_grad =
        (conv_tol_grad > 0) ? conv_tol_grad : std::sqrt(eff_tol);
    converged_stage_->set_tolerances(eff_tol, eff_grad);

    // ── Loop: fock → mix → diagonalize → density → check ──────────────────────
    const int budget = (opts.max_iter > 0) ? opts.max_iter : max_iter;
    const bool shift_active =
        std::abs(level_shift) > internal::kLevelShiftSkipThreshold;

    const auto saved_window = frame.resolution_window();
    setup_pass(frame);

    int executed = 0;
    for (iter_ = 0; iter_ < budget; ++iter_) {
        // diis_active is (diis_->size() >= 2) AFTER the MixStage's step this
        // iteration. DIISMixer::step pushes one
        // pair then trims to max_vectors, so size == min(pushes, max_vectors).
        if (use_diis && iter_ >= diis_start_iter) {
            diis_pairs_ = std::min(diis_pairs_ + 1, diis_max_vectors);
            diis_active_ = (diis_pairs_ >= 2);
        } else {
            diis_active_ = false;
        }

        mutable_outer_status().reset();
        step_once(iter_);
        ++executed;
        if (poll_criteria_(iter_) != ATV::Verdict::Continue)
            break;
    }

    teardown_pass();
    frame.restore_resolution_window(saved_window);

    const bool did_converge = this->converged();

    // ── Epilogue: de-shift, wavefunction publish, energy / forces ────────────────
    // De-shifted orbital energies: rebuild one UNSHIFTED Fock from the
    // converged density and diagonalize it. The SOSCF finish shares this
    // block: after CIAH rotations the orbitals are only semicanonical, so a
    // final diagonalization of the Fock of the converged density restores
    // the canonical ε (and canonical C) the published determinant carries.
    if (did_converge && (shift_active || soscf_active_)) {
        internal::HamiltonianBuilder deshift;
        // Same ri/aux as the loop's Fock build: the de-shifted orbital
        // energies must diagonalize the SAME (possibly fitted) Fock the
        // SCF converged, or ε would carry the fit error as an artifact.
        deshift.prepare(ints, is_hf, grid_level, n_radial, n_angular,
                        fixed_grid, ri, aux_basis, coulomb, cfmm_order,
                        cfmm_threshold);
        internal::FockResult fock =
            deshift.step(ints, restricted, is_hf, xc_type_,
                         hybrid_alpha_override, P_alpha_, P_beta_);
        std::tie(eig_alpha_, C_alpha_) = diagonalize(fock.F_alpha);
        if (restricted) {
            eig_beta_ = eig_alpha_;
            C_beta_ = C_alpha_;
        } else {
            std::tie(eig_beta_, C_beta_) = diagonalize(fock.F_beta);
        }
    }

    iterations_ = executed; // loop state lives on the Driver (converged()).

    // Publish the wavefunction: assign into the stable shared slot (the
    // handle-before-run contract). Integer Aufbau occupations per spin.
    const int n_mo = static_cast<int>(C_alpha_.shape()[1]);
    RealVector f_alpha = xt::zeros<double>({n_mo});
    for (int i = 0; i < n_alpha && i < n_mo; ++i)
        f_alpha(i) = 1.0;
    if (generalized) {
        // Spinor determinant: one channel, N electrons aufbau over the 2n
        // spinors (n_alpha carries the total N for the generalized form).
        *det_sp_ = SlaterDeterminant(SlaterDeterminant::Generalized{}, C_alpha_,
                                     eig_alpha_, f_alpha, n_alpha);
    } else if (restricted) {
        *det_sp_ = SlaterDeterminant(C_alpha_, eig_alpha_, f_alpha, n_alpha);
    } else {
        RealVector f_beta = xt::zeros<double>({n_mo});
        for (int i = 0; i < n_beta && i < n_mo; ++i)
            f_beta(i) = 1.0;
        *det_sp_ = SlaterDeterminant(C_alpha_, eig_alpha_, f_alpha, n_alpha,
                                     C_beta_, eig_beta_, f_beta, n_beta);
        det_sp_->set_spin_contamination(
            compute_spin_contamination(C_alpha_, C_beta_, S_, n_alpha, n_beta));
    }

    *has_result_sp_ = true;

    // Geometry-only DFT-D3(BJ) / DFT-D4 additive correction (after SCF; not in
    // Fock). Every gradient this class publishes carries the matching
    // −∂E_disp/∂R for the SAME flag pair — `ScfGradient` below for the mean
    // field, `add_dispersion_forces` in mp2_forces/ccsd_forces — all four
    // routed through `internal/dispersion.h` so the choice is made once.
    e_disp_ = internal::dispersion_energy(frame, d3bj, d4);
    e_total_ += e_disp_;

    if (opts.forces) {
        if (ri != RI::None)
            throw std::runtime_error(
                "MeanFieldMethod: analytic forces are not wired for RI "
                "(density fitting) — the gradient of the fitted energy "
                "needs the 2-/3-center derivative integrals (a later "
                "link). Use ri = RI::None for forces.");
        // The dispersion flags carry into the gradient: `e_total_` above
        // already includes E_disp, so the force must include −∂E_disp/∂R.
        internal::ScfGradient gradient{method_,
                                       spin_,
                                       xc_type_,
                                       grid_level,
                                       n_radial,
                                       n_angular,
                                       xc_grad_project_translation,
                                       d3bj,
                                       d4};
        // Fresh view after env (embed/field/solvent) was written onto storage —
        // IntegralView scalars (V_nuc, embed_self_energy, …) are by value and
        // the prologue view is stale for those; refs alone are not enough.
        last_grad_ =
            gradient.compute(frame, integral_view_(), *det_sp_, e_total_);
        // Republish embedding-site forces for the public accessor.
        embed_site_forces_.clear();
        if (last_grad_.embedding_forces.size() > 0) {
            const int nc =
                static_cast<int>(last_grad_.embedding_forces.shape(0));
            embed_site_forces_.resize(static_cast<std::size_t>(nc));
            for (int c = 0; c < nc; ++c) {
                embed_site_forces_[static_cast<std::size_t>(c)] = {
                    last_grad_.embedding_forces(c, 0),
                    last_grad_.embedding_forces(c, 1),
                    last_grad_.embedding_forces(c, 2)};
            }
        }
        const int natm = static_cast<int>(frame.nlocal());
        AccReal* __restrict fx = frame.span(kAtom_fx);
        AccReal* __restrict fy = frame.span(kAtom_fy);
        AccReal* __restrict fz = frame.span(kAtom_fz);
        // `accumulate` (the in-plan entry) vs overwrite (the standalone
        // entries) — see DriveOpts::accumulate. `kEnergyElectronic` carries
        // this method's own share and is written with the SAME operator as
        // `kEnergy`, so the per-term key never drifts from the total.
        if (opts.accumulate) {
            for (int A = 0; A < natm; ++A) {
                fx[A] += last_grad_.forces(A, 0);
                fy[A] += last_grad_.forces(A, 1);
                fz[A] += last_grad_.forces(A, 2);
            }
            frame.meta.get(kEnergy) += last_grad_.energy;
            frame.meta.get(kEnergyElectronic) += last_grad_.energy;
        } else {
            for (int A = 0; A < natm; ++A) {
                fx[A] = last_grad_.forces(A, 0);
                fy[A] = last_grad_.forces(A, 1);
                fz[A] = last_grad_.forces(A, 2);
            }
            frame.meta.get(kEnergy) = last_grad_.energy;
            frame.meta.get(kEnergyElectronic) = last_grad_.energy;
        }
    } else if (opts.write_energy) {
        const double e_ev = e_total_ * units::kHartreeInEv;
        if (opts.accumulate) {
            frame.meta.get(kEnergy) += e_ev;
            frame.meta.get(kEnergyElectronic) += e_ev;
        } else {
            frame.meta.get(kEnergy) = e_ev;
            frame.meta.get(kEnergyElectronic) = e_ev;
        }
    }
}

// ── Car–Parrinello (cpmd = true) ──────────────────────────────────────────────

template <typename D>
void MeanFieldMethod<D>::set_timestep(double dt_fs) {
    if (!(dt_fs > 0.0) || !std::isfinite(dt_fs))
        throw std::invalid_argument(
            "MeanFieldMethod::set_timestep: nuclear timestep must be finite "
            "and > 0 (fs), got "
            + std::to_string(dt_fs));
    cpmd_dt_fs_ = dt_fs;
}

template <typename D>
void MeanFieldMethod<D>::validate_cpmd_config_() const {
    // Spin shapes that are not a pair of independently propagated coefficient
    // matrices. ROHF/ROKS share one set of spatial orbitals between the
    // channels (the extended Lagrangian is a different constrained problem);
    // GHF propagates two-component spinors (the RATTLE metric is the spinor
    // overlap). Neither is "RHF with more columns".
    if (spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd is not supported for ROHF/ROKS — the "
            "restricted-open channels share one set of spatial orbitals, so "
            "there are no independent alpha/beta coefficients to propagate. "
            "Use RHF/RKS (occ = 2) or UHF/UKS (dual channel, occ = 1).");
    if (spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd is not supported for GHF — Car-Parrinello "
            "on a two-component spinor manifold is new physics (the "
            "constraint metric is the spinor overlap), not RHF with twice the "
            "columns. Use RHF/RKS or UHF/UKS.");

    // Competing density propagators: ASPC extrapolates the density across
    // nuclear steps while CP propagates the coefficients — running both means
    // two answers to "what is P now". The GPW twin refuses the same pair.
    if (use_predictor_)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd and the ASPC density predictor are "
            "mutually exclusive (ASPC extrapolates the density; CP propagates "
            "the coefficients). Drop predictor(...) or clear cpmd.");
    if (xl_bomd)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd and xl_bomd are mutually exclusive "
            "(XL-BOMD propagates an auxiliary density to guess an SCF that a "
            "CP production step never runs). Clear one.");

    // Fock builds / one-electron operators the CP force does not cover.
    if (ri != RI::None)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd requires ri = RI::None — the analytic "
            "gradient of a density-fitted energy needs the 2-/3-center "
            "derivative integrals, which ScfGradient does not carry.");
    if (x2c)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd is not supported with x2c = true — the "
            "h^X2C nuclear gradient (picture change + decoupling-matrix "
            "response) is not derived, so the CP nuclei would feel a "
            "non-relativistic force under a relativistic energy.");
    if (solvent_eps > 0.0)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd is not supported with a continuum solvent "
            "— the reaction field is a fixed-point of the SCF density, which "
            "a propagated density never solves. Set solvent_eps = 0.");
    if (mom != MomStyle::Off)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd and mom != Off are mutually exclusive — "
            "maximum-overlap occupation re-selects the occupied subspace at "
            "every diagonalization, and a CP production step never "
            "diagonalizes.");
    if (soscf)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd and soscf are mutually exclusive — SOSCF "
            "is a second-order accelerator for the SCF loop that a CP "
            "production step deliberately does not run.");
    if (!point_charges.empty())
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd with QM/MM point charges is not "
            "productized — the embedding force on the MM sites and the "
            "electron-MM Pulay response at a non-stationary density are the "
            "whole QM/MM x CPMD integration, not a flag. Clear "
            "point_charges.");
    if (external_field[0] != 0.0 || external_field[1] != 0.0
        || external_field[2] != 0.0)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd with a uniform external field is not "
            "productized — clear external_field.");

    // Energy terms the CP functional does not carry. `ScfGradient` now has
    // analytic D3(BJ)/D4 legs (BOMD is conservative with them), but the CP
    // step evaluates `mean_field_energy_at_density`, which folds in no E_disp
    // — so under CP the dispersion would be a force with no energy, the exact
    // mirror of the BOMD defect this refusal was written for. Lifting it means
    // adding E_disp to the CP total and to the conserved-energy bookkeeping,
    // not deleting these lines.
    if (d3bj || d4)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd with d3bj/d4 is refused — the "
            "Car-Parrinello energy functional carries no E_disp, so a "
            "dispersion force would be work done against an energy the "
            "conserved quantity never sees. Clear d3bj/d4.");

    // Propagator parameters (validated here so a bad mu is reported before a
    // full seed SCF is spent).
    if (!(cpmd_mu > 0.0) || !std::isfinite(cpmd_mu))
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd_mu must be finite and > 0 (a.u. mass)");
    if (cpmd_thermostat) {
        if (!(cpmd_ekin_target_ha > 0.0) || !std::isfinite(cpmd_ekin_target_ha))
            throw std::invalid_argument(
                "MeanFieldMethod: cpmd_thermostat requires "
                "cpmd_ekin_target_ha finite and > 0 (Hartree)");
        if (!(cpmd_tau_e > 0.0) || !std::isfinite(cpmd_tau_e))
            throw std::invalid_argument(
                "MeanFieldMethod: cpmd_thermostat requires cpmd_tau_e finite "
                "and > 0 (fs)");
    }
    if (!compute_force)
        throw std::invalid_argument(
            "MeanFieldMethod: cpmd requires compute_force = true — the "
            "nuclei are integrated from the CP force, so a force-free CP "
            "step would move nothing and conserve nothing.");
}

template <typename D>
void MeanFieldMethod<D>::cpmd_seed_() {
    // A near-singular overlap makes the holonomic constraint CᵀSC = I
    // ill-conditioned, so the propagated coefficients drift off the manifold.
    // Discovered while driving (the drop count is known only after the seed
    // SCF built X), hence runtime_error rather than invalid_argument.
    if (n_dropped_ > 0)
        throw std::runtime_error(
            "MeanFieldMethod: cpmd cannot seed from a wavefunction with "
            + std::to_string(n_dropped_)
            + " dropped overlap direction(s) (lindep_threshold = "
            + std::to_string(lindep_threshold)
            + ") — the RATTLE metric CᵀSC = I is ill-conditioned there. "
              "Lower lindep_threshold or use a less redundant basis.");
    if (n_alpha_ < 1)
        throw std::runtime_error(
            "MeanFieldMethod: cpmd seed requires at least one occupied alpha "
            "orbital to propagate.");
    const int n = static_cast<int>(C_alpha_.shape()[0]);
    if (n < 1 || static_cast<int>(C_alpha_.shape()[1]) < n_alpha_)
        throw std::runtime_error(
            "MeanFieldMethod: cpmd seed: the converged alpha MO set has "
            "fewer columns than occupied orbitals.");

    // Occupied columns only. They are S-orthonormal by construction on both
    // the Löwdin and the canonical-orthogonalization paths (CᵀSC = I is what
    // the generalized eigensolver returns), which is exactly the seed
    // contract of `electronic::Cpmd::seed`.
    const auto occupied = [](const RealMatrix& C, int n_occ, int n_basis) {
        RealMatrix out = xt::zeros<double>({n_basis, n_occ});
        for (int mu = 0; mu < n_basis; ++mu)
            for (int i = 0; i < n_occ; ++i)
                out(mu, i) = C(mu, i);
        return out;
    };

    cpmd_alpha_.emplace(cpmd_mu);
    cpmd_alpha_->thermostat = cpmd_thermostat;
    cpmd_alpha_->ekin_target_ha = cpmd_ekin_target_ha;
    cpmd_alpha_->tau_e_fs = cpmd_tau_e;
    cpmd_alpha_->seed(occupied(C_alpha_, n_alpha_, n));

    // UHF/UKS: the beta channel is independent (occ = 1 each). A doublet with
    // no beta electrons simply has no beta propagator.
    if (spin_ == SpinType::Unrestricted && n_beta_ > 0) {
        if (static_cast<int>(C_beta_.shape()[1]) < n_beta_)
            throw std::runtime_error(
                "MeanFieldMethod: cpmd seed: the converged beta MO set has "
                "fewer columns than occupied beta orbitals.");
        cpmd_beta_.emplace(cpmd_mu);
        cpmd_beta_->thermostat = cpmd_thermostat;
        cpmd_beta_->ekin_target_ha = cpmd_ekin_target_ha;
        cpmd_beta_->tau_e_fs = cpmd_tau_e;
        cpmd_beta_->seed(occupied(C_beta_, n_beta_, n));
    } else {
        cpmd_beta_.reset();
    }
}

template <typename D>
void MeanFieldMethod<D>::cpmd_step_(Frame& frame) {
    // A production Car-Parrinello step is NOT the stage Plan: DIIS,
    // diagonalization and the density rebuild are exactly what CP skips, and
    // `DriveOpts.max_iter = 0` cannot express "build one Fock at a given
    // density" (the loop body would never run, so no Fock would be built).
    // Drive `internal::HamiltonianBuilder` directly — the precedent is the
    // de-shift epilogue in drive_scf_core_.
    PositionGuard guard(frame, /*active=*/true);

    restricted_ = (spin_ == SpinType::Restricted);
    is_hf_ = (method_ == MeanFieldType::HF);
    const bool restricted = restricted_;
    const bool is_hf = is_hf_;

    // 1. Integrals at the new geometry (ERI is direct; CP never needs the
    //    dense tensor, and the guards already refused every knob that would).
    const internal::IntegralView ints =
        rebuild_integrals_(frame, /*compute_pvp=*/false);
    const int n = ints.n_basis;
    S_ = ints.S;
    Hcore_ = ints.Hcore;
    v_nuc_ = ints.V_nuc;
    n_basis_ = n;

    // 2. Löwdin-transport the coefficients under the NEW metric (a geometry
    //    change is not a dynamics step) and project the electronic velocity
    //    back onto the constraint manifold.
    cpmd_alpha_->align_to_overlap(S_);
    const bool has_beta = (cpmd_beta_ && cpmd_beta_->ready());
    if (has_beta)
        cpmd_beta_->align_to_overlap(S_);

    // 3. Density from the propagated coefficients. Restricted folds occ = 2
    //    into the alpha channel (the loop's convention: P_alpha carries the
    //    factor-2 closed-shell density and P_beta aliases it).
    const RealMatrix& C_a = cpmd_alpha_->C();
    const int n_occ_a = static_cast<int>(C_a.shape()[1]);
    RealMatrix P_a =
        linalg::build_density_matrix(C_a, n_occ_a, n, restricted ? 2.0 : 1.0);
    RealMatrix P_b;
    if (restricted) {
        P_b = P_a;
    } else if (has_beta) {
        const RealMatrix& C_b = cpmd_beta_->C();
        P_b = linalg::build_density_matrix(
            C_b, static_cast<int>(C_b.shape()[1]), n, 1.0);
    } else {
        P_b = xt::zeros<double>({n, n});
    }

    // 4. One Fock/KS build at that fixed density.
    internal::HamiltonianBuilder hb;
    hb.prepare(ints, is_hf, grid_level, n_radial, n_angular, fixed_grid, ri,
               aux_basis, coulomb, cfmm_order, cfmm_threshold);
    const internal::FockResult fock = hb.step(ints, restricted, is_hf, xc_type_,
                                              hybrid_alpha_override, P_a, P_b);
    F_alpha_ = fock.F_alpha;
    F_beta_ = fock.F_beta;
    J_ = fock.J;
    e_xc_grid_ = fock.E_xc_grid;
    e_exact_exchange_ = fock.E_exact_exchange;
    int_n_alpha_ = fock.int_n_alpha;
    int_n_beta_ = fock.int_n_beta;
    // Solvent / dispersion are refused under cpmd; keep the members honest.
    e_solvent_ = 0;
    e_pcm_fock_half_ = 0;
    e_disp_ = 0;

    // 5. The same four-branch functional the SCF loop evaluates, at a density
    //    that is both the Fock's input and this step's output.
    const internal::MeanFieldEnergy e = internal::mean_field_energy_at_density(
        restricted, is_hf, Hcore_, F_alpha_, F_beta_, J_, P_a, P_b, P_a, P_b,
        e_xc_grid_, e_exact_exchange_, v_nuc_, e_solvent_, e_pcm_fock_half_);
    e_total_ = e.total;
    e_one_ = e.one_electron;
    e_coulomb_ = e.coulomb;
    P_alpha_ = P_a;
    P_beta_ = P_b;

    // 6. Nuclear forces at the propagated (non-stationary) state: the Pulay
    //    weight is the full Lagrange-multiplier form, not sum_i eps_i c_i c_i^T.
    //    Accumulate with `+=`: MD zeroes forces and energy at pre_force, so
    //    this method is one contributor among however many the loop binds.
    // No dispersion legs here (`false, false`): the Car-Parrinello energy
    // functional evaluated at step 5 — `mean_field_energy_at_density` — carries
    // no E_disp (e_disp_ is zeroed above), so its gradient must not carry one
    // either. `validate_cpmd_config_` rejects d3bj/d4 before this point, so
    // the flags are false anyway; the literal is what keeps force and energy
    // describing the same surface if that refusal is ever lifted without also
    // folding E_disp into the CP total.
    internal::ScfGradient gradient{method_,
                                   spin_,
                                   xc_type_,
                                   grid_level,
                                   n_radial,
                                   n_angular,
                                   xc_grad_project_translation,
                                   /*d3bj=*/false,
                                   /*d4=*/false};
    const RealMatrix empty;
    last_grad_ = gradient.compute_at_density(
        frame, ints, C_a, F_alpha_,
        restricted || !has_beta ? empty : cpmd_beta_->C(),
        restricted ? empty : F_beta_, e_total_);
    const int natm = static_cast<int>(frame.nlocal());
    AccReal* __restrict fx = frame.span(kAtom_fx);
    AccReal* __restrict fy = frame.span(kAtom_fy);
    AccReal* __restrict fz = frame.span(kAtom_fz);
    for (int A = 0; A < natm; ++A) {
        fx[A] += last_grad_.forces(A, 0);
        fy[A] += last_grad_.forces(A, 1);
        fz[A] += last_grad_.forces(A, 2);
    }
    frame.meta.get(kEnergy) += last_grad_.energy;
    // Same value, same operator, into the electronic per-term share: the CP
    // step is only ever reached from the in-plan entry, so it accumulates.
    frame.meta.get(kEnergyElectronic) += last_grad_.energy;

    // 7. Advance the electrons one velocity-Verlet + RATTLE step on the
    //    nuclear clock, then publish the adiabaticity signal. Closed shell is
    //    one channel at occ = 2; UHF/UKS are two channels at occ = 1 each and
    //    T_e is their sum.
    cpmd_alpha_->advance(F_alpha_, S_, cpmd_dt_fs_, restricted ? 2.0 : 1.0);
    double te_ha = cpmd_alpha_->kinetic_ha();
    double etherm_ha = cpmd_alpha_->thermostat_energy_ha();
    if (has_beta) {
        cpmd_beta_->advance(F_beta_, S_, cpmd_dt_fs_, /*occ=*/1.0);
        te_ha += cpmd_beta_->kinetic_ha();
        etherm_ha += cpmd_beta_->thermostat_energy_ha();
    }
    last_te_ev_ = te_ha * units::kHartreeInEv;
    last_etherm_ev_ = etherm_ha * units::kHartreeInEv;
    publish_electronic_energies(frame, last_te_ev_, last_etherm_ev_);

    // The defining CP signature: zero SCF iterations, forever after the seed.
    iterations_ = 0;
}

// ── Entry points ──────────────────────────────────────────────────────────────

template <typename D>
void MeanFieldMethod<D>::run(Frame& frame, const SCFInitialGuess* guess) {
    refuse_standalone_cpmd(cpmd, "MeanFieldMethod::run");
    DriveOpts opts;
    opts.unwrap = true;
    opts.write_energy = true; // Frame-out: the mutating entry always writes.
    opts.forces = false;
    drive_scf_(frame, guess, opts);
}

template <typename D>
void MeanFieldMethod<D>::run(const Frame& frame, const SCFInitialGuess* guess) {
    refuse_standalone_cpmd(cpmd, "MeanFieldMethod::run");
    // SCF is read-only w.r.t. the frame: the stages compute on private member
    // matrices, never frame fields; this entry unwraps nothing and writes
    // nothing back. The const_cast is therefore never a mutation of the
    // caller's object (the backend only binds the pointer and touches the
    // const-qualified resolution window).
    DriveOpts opts;
    opts.unwrap = false;
    opts.write_energy = false;
    opts.forces = false;
    drive_scf_(const_cast<Frame&>(frame), guess, opts);
}

template <typename D>
GradientResult MeanFieldMethod<D>::forces(Frame& frame,
                                          const SCFInitialGuess* guess) {
    refuse_standalone_cpmd(cpmd, "MeanFieldMethod::forces");
    DriveOpts opts;
    opts.unwrap = false;
    opts.write_energy = false;
    opts.forces = true;
    drive_scf_(frame, guess, opts);
    return last_grad_;
}

template <typename D>
MP2Result MeanFieldMethod<D>::mp2(Frame& frame, const SCFInitialGuess* guess) {
    if (spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "MeanFieldMethod::mp2: a GHF (spinor) reference is not supported "
            "— the MP2 pack has no spinor transformation. Use RHF or UHF.");
    const bool prev = compute_eri;
    // Canonical MP2 needs a dense ERI; RI-MP2 builds 3-center integrals only.
    if (ri == RI::None)
        compute_eri = true;
    run(frame, guess);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::mp2: SCF reference did not converge — refusing "
            "to compute a correlation correction on a non-stationary "
            "determinant (no fallback)");
    if ((spin_ == SpinType::Unrestricted || spin_ == SpinType::RestrictedOpen)
        && frozen != 0) {
        throw std::invalid_argument(
            "MeanFieldMethod::mp2: frozen-core UMP2/ROMP2 is not supported — "
            "set frozen=0, or use closed-shell RHF.");
    }
    if (ri != RI::None) {
        if (spin_ != SpinType::Restricted)
            throw std::invalid_argument(
                "MeanFieldMethod::mp2: RI-MP2 is closed-shell only — set "
                "ri=RI::None for UMP2/ROMP2, or use RHF.");
        return ri_mp2_energy(integral_view_(), *det_sp_, e_total_, aux_basis,
                             frozen, mp2_os_scale, mp2_ss_scale);
    }
    if (spin_ == SpinType::RestrictedOpen) {
        // PySCF ROMP2: UMP2 on ROHF orbitals with semi-canonical spin-Fock
        // diagonal denominators (F_alpha_uncoupled_ / F_beta_uncoupled_).
        return romp2_energy(integral_view_(), *det_sp_, F_alpha_uncoupled_,
                            F_beta_uncoupled_, e_total_, mp2_os_scale,
                            mp2_ss_scale);
    }
    if (spin_ == SpinType::Unrestricted)
        return ump2_energy(integral_view_(), *det_sp_, e_total_, mp2_os_scale,
                           mp2_ss_scale);
    return mp2_energy(integral_view_(), *det_sp_, e_total_, frozen,
                      mp2_os_scale, mp2_ss_scale);
}

template <typename D>
GradientResult MeanFieldMethod<D>::mp2_forces(Frame& frame,
                                              const SCFInitialGuess* guess) {
    if (spin_ != SpinType::Restricted && spin_ != SpinType::Unrestricted)
        throw std::invalid_argument(
            "MeanFieldMethod::mp2_forces: RMP2 (RHF) or UMP2 (UHF) analytic "
            "gradient only — ROHF/ROMP2 gradients are a follow-up");
    if (ri != RI::None)
        throw std::invalid_argument(
            "MeanFieldMethod::mp2_forces: RI-MP2 gradients are not "
            "implemented — set ri=RI::None");
    if (frozen != 0)
        throw std::invalid_argument(
            "MeanFieldMethod::mp2_forces: frozen-core MP2 gradients are not "
            "implemented — set frozen=0");
    if (std::fabs(mp2_os_scale - 1.0) > 0.0
        || std::fabs(mp2_ss_scale - 1.0) > 0.0)
        throw std::invalid_argument(
            "MeanFieldMethod::mp2_forces: spin-component-scaled MP2 gradients "
            "are not implemented — keep mp2_os_scale=mp2_ss_scale=1");

    const bool prev = compute_eri;
    compute_eri = true;
    run(frame, guess);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::mp2_forces: SCF reference did not converge — "
            "refusing gradient on a non-stationary determinant (no fallback)");

    GradientResult gr;
    if (spin_ == SpinType::Unrestricted) {
        const MP2Result corr =
            ump2_energy(integral_view_(), *det_sp_, e_total_, /*os_scale=*/1.0,
                        /*ss_scale=*/1.0);
        gr = internal::ump2_analytic_forces(frame, integral_view_(), *det_sp_,
                                            e_total_, corr.e_corr);
    } else {
        const MP2Result corr =
            mp2_energy(integral_view_(), *det_sp_, e_total_, /*n_frozen=*/0,
                       /*os_scale=*/1.0, /*ss_scale=*/1.0);
        gr = internal::mp2_analytic_forces(frame, integral_view_(), *det_sp_,
                                           e_total_, corr.e_corr);
    }
    // `e_total_` (and therefore `gr.energy`) already carries E_disp; the MP2
    // gradient routines know nothing about dispersion, so the leg is added
    // here. Same switch as the energy — see internal/dispersion.h.
    internal::add_dispersion_forces(frame, d3bj, d4, gr);
    last_grad_ = gr;

    // Publish force / energy on Frame (same contract as forces()).
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
    return gr;
}

template <typename D>
MP3Result MeanFieldMethod<D>::mp3(Frame& frame, const SCFInitialGuess* guess) {
    if (spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MeanFieldMethod::mp3: closed-shell RMP3 only — use RHF "
            "(open-shell MP3 is a follow-up).");
    if (ri != RI::None)
        throw std::invalid_argument(
            "MeanFieldMethod::mp3: RI-MP3 is not implemented — set "
            "ri=RI::None");
    if (std::fabs(mp2_os_scale - 1.0) > 0.0
        || std::fabs(mp2_ss_scale - 1.0) > 0.0)
        throw std::invalid_argument(
            "MeanFieldMethod::mp3: spin-component-scaled MP3 is not "
            "implemented — keep mp2_os_scale=mp2_ss_scale=1");

    const bool prev = compute_eri;
    compute_eri = true;
    run(frame, guess);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::mp3: SCF reference did not converge — refusing "
            "to compute a correlation correction on a non-stationary "
            "determinant (no fallback)");
    return mp3_energy(integral_view_(), *det_sp_, e_total_, frozen);
}

template <typename D>
MP4Result MeanFieldMethod<D>::mp4(Frame& frame, const SCFInitialGuess* guess) {
    if (spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MeanFieldMethod::mp4: closed-shell RMP4 only — use RHF "
            "(open-shell MP4 is a follow-up).");
    if (ri != RI::None)
        throw std::invalid_argument(
            "MeanFieldMethod::mp4: RI-MP4 is not implemented — set "
            "ri=RI::None");
    if (std::fabs(mp2_os_scale - 1.0) > 0.0
        || std::fabs(mp2_ss_scale - 1.0) > 0.0)
        throw std::invalid_argument(
            "MeanFieldMethod::mp4: spin-component-scaled MP4 is not "
            "implemented — keep mp2_os_scale=mp2_ss_scale=1");

    const bool prev = compute_eri;
    compute_eri = true;
    run(frame, guess);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::mp4: SCF reference did not converge — refusing "
            "to compute a correlation correction on a non-stationary "
            "determinant (no fallback)");
    return mp4_energy(integral_view_(), *det_sp_, e_total_, frozen);
}

template <typename D>
FCIResult MeanFieldMethod<D>::fci(Frame& frame) {
    if (spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "MeanFieldMethod::fci: an ROHF reference is not supported — the "
            "FCI pack requires a closed-shell reference. Use RHF.");
    if (spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "MeanFieldMethod::fci: a GHF (spinor) reference is not supported "
            "— the FCI pack requires a closed-shell reference. Use RHF.");
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::fci: SCF reference did not converge — refusing "
            "to compute a correlation correction on a non-stationary "
            "determinant (no fallback)");
    return fci_energy(integral_view_(), *det_sp_, e_total_);
}

template <typename D>
CASCIResult MeanFieldMethod<D>::casci(Frame& frame, int ncas, int nelecas) {
    if (spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "MeanFieldMethod::casci: an ROHF reference is not supported — "
            "closed-shell CASCI requires RHF.");
    if (spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "MeanFieldMethod::casci: a GHF (spinor) reference is not "
            "supported — closed-shell CASCI requires RHF.");
    if (spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MeanFieldMethod::casci: closed-shell RHF reference only — UHF "
            "CASCI is not implemented (no fallback). Use RHF.");
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::casci: SCF reference did not converge — refusing "
            "to compute a correlation correction on a non-stationary "
            "determinant (no fallback)");
    return casci_energy(integral_view_(), *det_sp_, e_total_, ncas, nelecas);
}

template <typename D>
CASSCFResult MeanFieldMethod<D>::casscf(Frame& frame, int ncas, int nelecas) {
    if (spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "MeanFieldMethod::casscf: an ROHF reference is not supported — "
            "closed-shell CASSCF requires RHF.");
    if (spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "MeanFieldMethod::casscf: a GHF (spinor) reference is not "
            "supported — closed-shell CASSCF requires RHF.");
    if (spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MeanFieldMethod::casscf: closed-shell RHF reference only — UHF "
            "CASSCF is not implemented (no fallback). Use RHF.");
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument("MeanFieldMethod::casscf: SCF reference "
                                    "did not converge — refusing "
                                    "to optimize orbitals on a non-stationary "
                                    "determinant (no fallback)");
    // Orbital gradient tol tracks the SCF energy tolerance (Ha).
    return casscf_energy(integral_view_(), *det_sp_, e_total_, ncas, nelecas,
                         /*orb_tol=*/std::max(tol, 1e-8), /*max_macro=*/50);
}

template <typename D>
NEVPT2Result MeanFieldMethod<D>::nevpt2(Frame& frame, int ncas, int nelecas) {
    if (spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "MeanFieldMethod::nevpt2: an ROHF reference is not supported — "
            "closed-shell SC-NEVPT2 requires RHF.");
    if (spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "MeanFieldMethod::nevpt2: a GHF (spinor) reference is not "
            "supported — closed-shell SC-NEVPT2 requires RHF.");
    if (spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MeanFieldMethod::nevpt2: closed-shell RHF reference only — UHF "
            "NEVPT2 is not implemented (no fallback). Use RHF.");
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument("MeanFieldMethod::nevpt2: SCF reference "
                                    "did not converge — refusing "
                                    "to run CASSCF/NEVPT2 on a non-stationary "
                                    "determinant (no fallback)");
    return nevpt2_energy(integral_view_(), *det_sp_, e_total_, ncas, nelecas,
                         /*orb_tol=*/std::max(tol, 1e-8), /*max_macro=*/50);
}

template <typename D>
CCSDResult MeanFieldMethod<D>::ccsd(Frame& frame) {
    if (spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MeanFieldMethod::ccsd: closed-shell RHF reference only — UHF / "
            "ROHF / GHF CCSD is not implemented (no fallback). Use RHF.");
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::ccsd: SCF reference did not converge — refusing "
            "to compute a correlation correction on a non-stationary "
            "determinant (no fallback)");
    return ccsd_energy(integral_view_(), *det_sp_, e_total_, /*tol=*/1e-10,
                       /*max_iter=*/100, /*use_diis=*/cc_diis);
}

template <typename D>
GradientResult MeanFieldMethod<D>::ccsd_forces(Frame& frame,
                                               const SCFInitialGuess* guess) {
    if (spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MeanFieldMethod::ccsd_forces: closed-shell RHF reference only — "
            "UCCSD gradients are not implemented (no fallback). Use RHF.");
    if (ri != RI::None)
        throw std::invalid_argument(
            "MeanFieldMethod::ccsd_forces: RI-CCSD gradients are not "
            "implemented — set ri=RI::None");
    if (frozen != 0)
        throw std::invalid_argument(
            "MeanFieldMethod::ccsd_forces: frozen-core CCSD gradients are not "
            "implemented — set frozen=0");

    const bool prev = compute_eri;
    compute_eri = true;
    run(frame, guess);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::ccsd_forces: SCF reference did not converge — "
            "refusing gradient on a non-stationary determinant (no fallback)");

    const CCSDResult corr = ccsd_energy(integral_view_(), *det_sp_, e_total_);
    GradientResult gr = internal::ccsd_analytic_forces(
        frame, integral_view_(), *det_sp_, e_total_, corr.e_corr);
    // Same reason as mp2_forces: the reference energy folded in E_disp, so the
    // published force must fold in −∂E_disp/∂R (internal/dispersion.h).
    internal::add_dispersion_forces(frame, d3bj, d4, gr);
    last_grad_ = gr;

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
    return gr;
}

template <typename D>
CC2Result MeanFieldMethod<D>::cc2(Frame& frame) {
    if (spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MeanFieldMethod::cc2: closed-shell RHF reference only — UHF / "
            "ROHF / GHF CC2 is not implemented (no fallback). Use RHF.");
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::cc2: SCF reference did not converge — refusing "
            "to compute a correlation correction on a non-stationary "
            "determinant (no fallback)");
    return cc2_energy(integral_view_(), *det_sp_, e_total_);
}

template <typename D>
CCSDTResult MeanFieldMethod<D>::ccsd_t(Frame& frame) {
    if (spin_ != SpinType::Restricted)
        throw std::invalid_argument(
            "MeanFieldMethod::ccsd_t: closed-shell RHF reference only — UHF / "
            "ROHF / GHF CCSD(T) is not implemented (no fallback). Use RHF.");
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::ccsd_t: SCF reference did not converge — "
            "refusing "
            "to compute a correlation correction on a non-stationary "
            "determinant (no fallback)");
    return ccsd_t_energy(integral_view_(), *det_sp_, e_total_);
}

// ── MOM/IMOM reference install ────────────────────────────────────────────────

namespace {

RealMatrix gather_occupied_columns(const SlaterDeterminant& det,
                                   SlaterDeterminant::Spin spin) {
    const RealMatrix& C = det.coefficients(spin);
    const RealVector& f = det.occupations(spin);
    const int n = static_cast<int>(C.shape()[0]);
    const int n_mo = static_cast<int>(C.shape()[1]);
    int n_occ = 0;
    for (int i = 0; i < n_mo; ++i)
        if (f(i) > 0.5)
            ++n_occ;
    if (n_occ == 0)
        throw std::invalid_argument(
            "MeanFieldMethod::set_mom_reference: spin channel has no occupied "
            "orbitals (f_i > 0.5)");
    RealMatrix C_occ = xt::zeros<double>({n, n_occ});
    int col = 0;
    for (int i = 0; i < n_mo; ++i) {
        if (f(i) <= 0.5)
            continue;
        xt::view(C_occ, xt::all(), col) = xt::view(C, xt::all(), i);
        ++col;
    }
    return C_occ;
}

} // namespace

template <typename D>
void MeanFieldMethod<D>::set_mom_reference(const SlaterDeterminant& det) {
    using Spin = SlaterDeterminant::Spin;
    if (det.generalized())
        throw std::invalid_argument("MeanFieldMethod::set_mom_reference: "
                                    "generalized (GHF) determinants "
                                    "are not supported for MOM/IMOM in this "
                                    "link — use RHF/UHF/RKS/UKS");
    mom_ref_C_alpha_ = gather_occupied_columns(det, Spin::alpha);
    if (det.restricted())
        mom_ref_C_beta_ = mom_ref_C_alpha_;
    else
        mom_ref_C_beta_ = gather_occupied_columns(det, Spin::beta);
}

template <typename D>
vib::HessianResult MeanFieldMethod<D>::central_difference_nuclear_hessian(
    Frame& frame, const std::function<GradientResult(Frame&)>& forces_at) {
    const int natm = static_cast<int>(frame.nlocal());
    const int n3 = 3 * natm;
    // Central FD of the caller's analytic forces. Match the offline PySCF pin
    // step (h = 1e-4 Bohr); convert forces eV/Å → Ha/Bohr for the FD, then
    // scale the Hessian to eV/Å². A fresh solver per ±h displacement (owned by
    // forces_at) avoids DIIS/MO state bleed.
    //   g = −F;  H_ha = ∂g/∂R [Ha/Bohr²];  H_ev = H_ha * scale.
    const double h_bohr = 1e-4;
    const double h_ang = h_bohr * units::kBohrInAngstrom;
    const double f_to_ha_bohr = units::kBohrInAngstrom / units::kHartreeInEv;
    const double hess_to_ev_ang2 =
        units::kHartreeInEv / (units::kBohrInAngstrom * units::kBohrInAngstrom);

    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto coord = [&](int A, int d) -> StorageReal& {
        if (d == 0)
            return x[A];
        if (d == 1)
            return y[A];
        return z[A];
    };

    std::vector<double> r0(static_cast<std::size_t>(n3));
    for (int A = 0; A < natm; ++A)
        for (int d = 0; d < 3; ++d)
            r0[static_cast<std::size_t>(3 * A + d)] =
                static_cast<double>(coord(A, d));

    xt::xtensor<double, 2> H = xt::zeros<double>({n3, n3});

    for (int A = 0; A < natm; ++A) {
        for (int d = 0; d < 3; ++d) {
            const int col = 3 * A + d;
            for (int B = 0; B < natm; ++B)
                for (int e = 0; e < 3; ++e)
                    coord(B, e) = static_cast<StorageReal>(
                        r0[static_cast<std::size_t>(3 * B + e)]);
            coord(A, d) = static_cast<StorageReal>(
                r0[static_cast<std::size_t>(col)] + h_ang);
            const GradientResult gp = forces_at(frame);

            coord(A, d) = static_cast<StorageReal>(
                r0[static_cast<std::size_t>(col)] - h_ang);
            const GradientResult gm = forces_at(frame);

            // g = −F; H = ∂g/∂R = −∂F/∂R.
            for (int B = 0; B < natm; ++B)
                for (int e = 0; e < 3; ++e) {
                    const int row = 3 * B + e;
                    const double Fp =
                        static_cast<double>(gp.forces(B, e)) * f_to_ha_bohr;
                    const double Fm =
                        static_cast<double>(gm.forces(B, e)) * f_to_ha_bohr;
                    const double dF_dR = (Fp - Fm) / (2.0 * h_bohr);
                    H(row, col) = (-dF_dR) * hess_to_ev_ang2;
                }
        }
    }

    // Restore geometry to the reference r0.
    for (int A = 0; A < natm; ++A)
        for (int d = 0; d < 3; ++d)
            coord(A, d) = static_cast<StorageReal>(
                r0[static_cast<std::size_t>(3 * A + d)]);

    // Symmetrize (FD noise).
    for (int i = 0; i < n3; ++i)
        for (int j = 0; j < i; ++j) {
            const double s = 0.5 * (H(i, j) + H(j, i));
            H(i, j) = s;
            H(j, i) = s;
        }

    vib::HessianResult out;
    out.n_atoms = natm;
    out.hessian = std::move(H);
    return out;
}

template <typename D>
void MeanFieldMethod<D>::check_hessian_preconditions(
    std::string_view method_name, const internal::IntegralView& pack) const {
    if (!converged())
        throw std::runtime_error(std::string(method_name)
                                 + ": SCF did not converge — no reference "
                                   "wavefunction to differentiate ("
                                 + stop_reason() + ")");
    if (pack.spherical())
        throw std::runtime_error(
            std::string(method_name)
            + ": analytic second derivatives are not implemented "
              "for spherical-harmonic basis sets (def2 family) yet — the "
              "derivative c2s transform is a follow-up chain link. Use a "
              "Cartesian-convention basis or the finite-difference "
              "vib::Hessian driver.");
    if (pack.has_ecp)
        throw std::runtime_error(
            std::string(method_name)
            + ": analytic second derivatives are not implemented "
              "for ECP packs — d²U/dR² of the semi-local potential is a "
              "follow-up chain link. Use the finite-difference vib::Hessian "
              "driver.");
}

// ── QmEngine / EmbeddingEngine (esmm-qmmm-02) ─────────────────────────

template <typename D>
std::string MeanFieldMethod<D>::qm_engine_name() const {
    return static_cast<const D*>(this)->name();
}

template <typename D>
void MeanFieldMethod<D>::prepare(const cpu::BuildContext&) {
    // Integrals need geometry at evaluate time; nothing to allocate cold.
}

template <typename D>
void MeanFieldMethod<D>::evaluate(const cpu::LaunchContext& qm_ctx) {
    DriveOpts opts;
    opts.write_energy = true;
    opts.forces = qm_ctx.forces_requested;
    opts.unwrap = true;
    // Standalone-style overwrite of energy/forces on the private QM frame
    // (couplings pre-zero the channels).
    opts.accumulate = false;
    drive_scf_(qm_ctx.frame, /*guess=*/nullptr, opts);

    // Mulliken charges [e] onto the private QM frame when a density exists.
    if (det_sp_ && ints_n_basis_ > 0) {
        const auto charges =
            cpu::mulliken_charge(integral_view_(), det_sp_->density_total());
        StorageReal* __restrict q = qm_ctx.frame.span(kAtom_charge);
        const int n = static_cast<int>(qm_ctx.frame.nlocal());
        const int nc = static_cast<int>(charges.size());
        for (int i = 0; i < n && i < nc; ++i)
            q[i] =
                static_cast<StorageReal>(charges[static_cast<std::size_t>(i)]);
    }
}

template <typename D>
void MeanFieldMethod<D>::reset_history() {
    aspc_alpha_.reset();
    aspc_beta_.reset();
    has_last_density_ = false;
    use_predictor_ = false;
    // XL / CPMD state is owned by AIMD wrappers; clear any residual seed flags
    // that would warm-start a second evaluate after reset.
    xl_alpha_.reset();
    xl_beta_.reset();
    cpmd_alpha_.reset();
    cpmd_beta_.reset();
}

// explicit instantiations
template class MeanFieldMethod<RHF>;
template class MeanFieldMethod<ROHF>;
template class MeanFieldMethod<UHF>;
template class MeanFieldMethod<GHF>;
template class MeanFieldMethod<RKS>;
template class MeanFieldMethod<UKS>;
template class MeanFieldMethod<ROKS>;

} // namespace scf
} // namespace cpu
} // namespace ATV
