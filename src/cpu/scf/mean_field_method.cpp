/// @file cpu/scf/mean_field_method.cpp
/// @brief `MeanFieldMethod<D>` template implementation — Frame API; a
///        CPUDriver<scf::Phase> loop over the sub-spec-02 stages, funneled
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
#include "atomiverse/cpu/linalg/eigensolve.h" // CanonicalOrth, build_orthogonalization_matrix
#include "atomiverse/cpu/scf/aspc_predictor.h"
#include "atomiverse/cpu/scf/ecp_potential.h"
#include "atomiverse/cpu/scf/internal/external_field.h"
#include "atomiverse/cpu/scf/internal/fci_pack.h"
#include "atomiverse/cpu/scf/internal/gradient.h"
#include "atomiverse/cpu/scf/internal/hamiltonian_builder.h"
#include "atomiverse/cpu/scf/internal/hueckel_guess.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/pcm.h"
#include "atomiverse/cpu/scf/internal/point_charges.h"
#include "atomiverse/cpu/scf/internal/sap_guess.h"
#include "atomiverse/cpu/scf/internal/level_shift.h" // kLevelShiftSkipThreshold
#include "atomiverse/cpu/scf/internal/minao.h"
#include "atomiverse/cpu/scf/internal/mp2_pack.h"
#include "atomiverse/cpu/scf/internal/x2c.h"
#include "atomiverse/cpu/scf/scf_guess.h"     // apply_homo_lumo_mixing
#include "atomiverse/cpu/scf/spin_analysis.h" // compute_spin_contamination

namespace ATV {
namespace cpu {
namespace scf {

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
    if (frame_.simbox.length(0) <= 0.0 || frame_.simbox.length(1) <= 0.0
        || frame_.simbox.length(2) <= 0.0)
        return;
    StorageReal* __restrict sp_x = frame_.span(kAtom_x);
    StorageReal* __restrict sp_y = frame_.span(kAtom_y);
    StorageReal* __restrict sp_z = frame_.span(kAtom_z);
    const double x0 = sp_x[0], y0 = sp_y[0], z0 = sp_z[0];
    for (int i = 1; i < n; ++i) {
        double dx = sp_x[i] - x0, dy = sp_y[i] - y0, dz = sp_z[i] - z0;
        frame_.simbox.min_image(dx, dy, dz);
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
    aspc_alpha_ = std::make_unique<ASPCPredictor>(order);
    aspc_beta_ = std::make_unique<ASPCPredictor>(order);
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
    fock_stage_ = std::make_unique<FockStage>(
        restricted_, is_hf_, xc_type_, grid_level, n_radial, n_angular,
        hybrid_alpha_override, fixed_grid, *integrals_sp_, P_alpha_, P_beta_,
        F_alpha_, F_beta_, J_, e_xc_grid_, e_exact_exchange_, int_n_alpha_,
        int_n_beta_, ri, aux_basis, generalized, pcm_.get(), e_solvent_,
        e_pcm_fock_half_);
    if (rohf) {
        rohf_couple_stage_ = std::make_unique<RohfCoupleStage>(
            S_, P_alpha_, P_beta_, F_alpha_, F_beta_, F_alpha_uncoupled_,
            F_beta_uncoupled_, P_total_);
    }
    if (soscf) {
        soscf_stepper_ = std::make_unique<internal::SOSCFStepper>(
            restricted_, *integrals_sp_, n_alpha_, n_beta_, Hcore_, v_nuc_,
            F_alpha_, F_beta_, P_alpha_, P_beta_, C_alpha_, C_beta_,
            eig_alpha_, eig_beta_, soscf_max_micro, soscf_trust);
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
        n_basis_, diis_active_, e_xc_grid_, e_exact_exchange_, v_nuc_,
        P_alpha_, P_beta_, e_total_, e_one_, e_coulomb_, generalized,
        e_solvent_, e_pcm_fock_half_);
    converged_stage_ = std::make_unique<ScfConverged>(
        tol, (conv_tol_grad > 0) ? conv_tol_grad : std::sqrt(tol),
        restricted_ || rohf || generalized, e_total_, err_alpha_, err_beta_);

    this->on(ATV::scf::Phase::fock).add_ref(*fock_stage_);
    if (rohf)
        this->on(ATV::scf::Phase::mix).add_ref(*rohf_couple_stage_,
                                               /*order=*/-1);
    this->on(ATV::scf::Phase::mix).add_ref(*mix_stage_);
    this->on(ATV::scf::Phase::diagonalize).add_ref(*diag_stage_);
    this->on(ATV::scf::Phase::density).add_ref(*density_stage_);
    this->add_criterion_ref(*converged_stage_);

    // Whole-molecule serial evaluator — no spatial decomposition.
    this->set_serial_on_replica();
    this->build(); // CPUDriver::build — compile the Plan (un-hidden via using).
    driver_built_ = true;
    wired_self_ = this;
}

// ── Nested driver lifecycle ───────────────────────────────────────────────────

template <typename D>
void MeanFieldMethod<D>::build_on(cpu::Backend& rt) {
    // Stage wiring needs geometry (IntegralPack dimensions); deferred to the
    // first drive_scf_. Parent Runtime does not own this method's Plan.
    (void)rt;
}

template <typename D>
void MeanFieldMethod<D>::destroy_on(cpu::Backend& rt) {
    (void)rt;
    reset_built_driver_();
}

template <typename D>
int MeanFieldMethod<D>::run_nested_erased(cpu::Context& ctx, cpu::Frame& f,
                                          int budget) {
    (void)ctx;
    DriveOpts opts;
    opts.unwrap = true;
    opts.forces = compute_force;
    opts.write_energy = true;
    if (budget > 0)
        opts.max_iter = budget;
    drive_scf_(f, nullptr, opts);
    return iterations_;
}

// ── THE funnel ────────────────────────────────────────────────────────────────

template <typename D>
void MeanFieldMethod<D>::drive_scf_(Frame& frame,
                                    const SCFInitialGuess* guess,
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
                    "MeanFieldMethod::run: x2c=true combined with ecp='"
                    + ecp + "' would double-count scalar relativity for "
                            "element "
                    + EcpPotential::element_symbol(Z) + " (Z="
                    + std::to_string(Z)
                    + "): the relativistic ECP already includes it. Disable "
                      "x2c or run this element all-electron.");
            }
        }
    }

    // ASPC predict / correct / record — gated only on use_predictor_ (and
    // opts.apply_aspc so the corrector/fallback recursion does not re-enter).
    SCFInitialGuess density_guess;
    const SCFInitialGuess* effective_guess = guess;
    bool predicted = false;

    if (opts.apply_aspc && use_predictor_ && aspc_alpha_ && aspc_beta_) {
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
        } else if (has_last_density_ && !guess) {
            density_guess.P_alpha = last_P_alpha_;
            density_guess.P_beta = last_P_beta_;
            density_guess.has_alpha = true;
            density_guess.has_beta = true;
            effective_guess = &density_guess;
        }
    } else if (opts.apply_aspc && has_last_density_ && !guess) {
        // Warm-start from the previous outer-step density (even without ASPC).
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
        drive_scf_core_(frame, effective_guess, copts);
        if (!this->converged() && fallback_full_scf_) {
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

    *integrals_sp_ = IntegralPack{frame, basis, charge, compute_eri, ecp,
                                  ghost_atoms, /*compute_pvp=*/x2c};
    // x2c-1e: replace the non-relativistic core Hamiltonian with the
    // one-step exact-two-component h^X2C. S / ERI / Fock loop are untouched
    // (two-electron integrals stay non-relativistic by definition of
    // x2c-1e). W was assembled only when x2c=true.
    if (x2c) {
        // RealMatrix is column-major; IntegralPack stores row-major
        // xtensor — copy into RealMatrix so the workspace binds cleanly.
        const RealMatrix S = integrals_sp_->S;
        const RealMatrix T = integrals_sp_->T;
        const RealMatrix V = integrals_sp_->V;
        const RealMatrix W = integrals_sp_->W;
        const internal::X2cWorkspace x2c_ws(S, T, V, W,
                                           internal::kLightSpeedAU);
        integrals_sp_->Hcore = x2c_ws.hcore();
    }
    const IntegralPack& ints = *integrals_sp_;
    resolve_occupation(ints.n_electrons);

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
    // Restricted/Unrestricted branch (no fallback).
    const bool generalized = (spin_ == SpinType::Generalized);
    if (generalized) {
        if (opts.forces)
            throw std::invalid_argument(
                "MeanFieldMethod::forces: GHF analytic forces are not "
                "implemented — the spinor gradient assembly is a follow-up "
                "chain link. Use finite differences or UHF.");
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
        if (opts.forces)
            throw std::invalid_argument(
                "MeanFieldMethod::forces: ROHF analytic forces are not "
                "implemented — the gradient Lagrangian needs the Roothaan "
                "coupling terms (follow-up chain link). Use finite "
                "differences or UHF.");
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

    // Environment (point charges / external field / C-PCM) — refuse silent
    // wrong forces when any is active (analytic gradients of environment
    // terms are follow-up links).
    const bool field_active =
        std::abs(external_field[0]) > 0.0 || std::abs(external_field[1]) > 0.0
        || std::abs(external_field[2]) > 0.0;
    const bool solvent_active = solvent_eps > 0.0;
    const bool env_active =
        !point_charges.empty() || field_active || solvent_active;
    if (opts.forces && env_active)
        throw std::runtime_error(
            "MeanFieldMethod::forces: analytic forces are not implemented "
            "with point_charges, external_field, or solvent_eps > 0 — the "
            "environment / C-PCM gradient is a follow-up chain link. Clear "
            "the environment members, or use finite differences.");

    // Apply environment to the n-dimensional Hcore / classical energy
    // *before* any GHF spinor embed so both spin blocks share the same
    // one-electron environment (Frame-pure: env is never parked on Frame).
    RealMatrix H = ints.Hcore;
    double v_nuc = ints.V_nuc;
    if (!point_charges.empty()) {
        std::vector<internal::PointChargeSite> sites;
        sites.reserve(point_charges.size());
        for (const auto& c : point_charges)
            sites.push_back({c.x, c.y, c.z, c.q});
        const internal::PointChargeEmbedding emb(sites);
        RealMatrix Venv = emb.core_matrix(ints.shells);
        if (ints.spherical()) {
            const auto ct = xt::transpose(ints.c2s);
            Venv = xt::linalg::dot(ct, xt::linalg::dot(Venv, ints.c2s));
        }
        H += Venv;
        v_nuc += emb.nuclear_energy(ints.atoms) + emb.self_energy();
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
    }

    // Publish the environment-augmented Hcore back onto the private pack so
    // FockStage / HamiltonianBuilder (which read `integrals.Hcore`, not the
    // method member) build F = H_env + G. Energy uses Hcore_ + v_nuc_ below;
    // if only the member were updated, ½Tr[(H_env+F_vac)P] would undercount
    // the field/embedding by exactly ½.
    integrals_sp_->Hcore = H;
    integrals_sp_->V_nuc = v_nuc;

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
            /*lebedev_n=*/302);
        if (ints.spherical())
            pcm_->prepare(ints.shells, ints.c2s);
        else
            pcm_->prepare(ints.shells);
    } else {
        pcm_.reset();
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
    const RealMatrix& H_ao = integrals_sp_->Hcore;

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
            const RealMatrix P_c = density.step(C_n, n_a, n)
                                   + density.step(C_n, n_b, n);
            P0 = xt::zeros<double>({2 * n, 2 * n});
            xt::view(P0, xt::range(0, n), xt::range(0, n)) = 0.5 * P_c;
            xt::view(P0, xt::range(n, 2 * n), xt::range(n, 2 * n)) =
                0.5 * P_c;

            double theta = 0.0;
            if constexpr (std::is_same_v<D, GHF>)
                theta = static_cast<D*>(this)->guess_spin_rotation;
            for (int mu = 0; mu < n; ++mu) {
                const double d = std::min(
                    std::max(P_c(mu, mu), 0.0), 2.0 - P_c(mu, mu));
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
    const bool want_sap = (init_guess == InitGuess::SAP) && !valid_guess_alpha;
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
        const MolecularGrid sap_grid = build_molecular_grid(
            ints.atoms.data(), static_cast<int>(ints.atoms.size()), grid_level,
            n_radial, n_angular);
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
                        fixed_grid, ri, aux_basis);
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
        *det_sp_ = SlaterDeterminant(SlaterDeterminant::Generalized{},
                                     C_alpha_, eig_alpha_, f_alpha, n_alpha);
    } else if (restricted) {
        *det_sp_ = SlaterDeterminant(C_alpha_, eig_alpha_, f_alpha, n_alpha);
    } else {
        RealVector f_beta = xt::zeros<double>({n_mo});
        for (int i = 0; i < n_beta && i < n_mo; ++i)
            f_beta(i) = 1.0;
        *det_sp_ =
            SlaterDeterminant(C_alpha_, eig_alpha_, f_alpha, n_alpha, C_beta_,
                              eig_beta_, f_beta, n_beta);
        det_sp_->set_spin_contamination(compute_spin_contamination(
            C_alpha_, C_beta_, S_, n_alpha, n_beta));
    }

    *has_result_sp_ = true;

    if (opts.forces) {
        if (ri != RI::None)
            throw std::runtime_error(
                "MeanFieldMethod: analytic forces are not wired for RI "
                "(density fitting) — the gradient of the fitted energy "
                "needs the 2-/3-center derivative integrals (a later "
                "link). Use ri = RI::None for forces.");
        internal::ScfGradient gradient{method_,
                                       spin_,
                                       xc_type_,
                                       grid_level,
                                       n_radial,
                                       n_angular,
                                       force_native_two_electron_deriv,
                                       xc_grad_project_translation};
        last_grad_ = gradient.compute(frame, ints, *det_sp_, e_total_);
        const int natm = static_cast<int>(frame.nlocal());
        AccReal* __restrict fx = frame.span(kAtom_fx);
        AccReal* __restrict fy = frame.span(kAtom_fy);
        AccReal* __restrict fz = frame.span(kAtom_fz);
        for (int A = 0; A < natm; ++A) {
            fx[A] = last_grad_.forces(A, 0);
            fy[A] = last_grad_.forces(A, 1);
            fz[A] = last_grad_.forces(A, 2);
        }
        frame.meta.get(kEnergy) = last_grad_.energy;
    } else if (opts.write_energy) {
        frame.meta.get(kEnergy) = e_total_ * units::kHartreeInEv;
    }
}

// ── Entry points ──────────────────────────────────────────────────────────────

template <typename D>
void MeanFieldMethod<D>::run(Frame& frame, const SCFInitialGuess* guess) {
    DriveOpts opts;
    opts.unwrap = true;
    opts.write_energy = true; // Frame-out: the mutating entry always writes.
    opts.forces = false;
    drive_scf_(frame, guess, opts);
}

template <typename D>
void MeanFieldMethod<D>::run(const Frame& frame,
                             const SCFInitialGuess* guess) {
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
GradientResult MeanFieldMethod<D>::forces(Frame& frame) {
    DriveOpts opts;
    opts.unwrap = false;
    opts.write_energy = false;
    opts.forces = true;
    drive_scf_(frame, nullptr, opts);
    return last_grad_;
}

template <typename D>
MP2Result MeanFieldMethod<D>::mp2(Frame& frame) {
    if (spin_ == SpinType::RestrictedOpen)
        throw std::invalid_argument(
            "MeanFieldMethod::mp2: an ROHF reference is not supported — the "
            "closed-shell MP2 pack would silently treat the single MO set as "
            "a doubly-occupied one. Use RHF (closed shell) — ROHF-MP2 is a "
            "follow-up chain link.");
    if (spin_ == SpinType::Generalized)
        throw std::invalid_argument(
            "MeanFieldMethod::mp2: a GHF (spinor) reference is not supported "
            "— the closed-shell MP2 pack has no spinor transformation. Use "
            "RHF.");
    const bool prev = compute_eri;
    compute_eri = true;
    run(frame);
    compute_eri = prev;
    if (!this->converged())
        throw std::invalid_argument(
            "MeanFieldMethod::mp2: SCF reference did not converge — refusing "
            "to compute a correlation correction on a non-stationary "
            "determinant (no fallback)");
    return mp2_energy(*integrals_sp_, *det_sp_, e_total_);
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
    return fci_energy(*integrals_sp_, *det_sp_, e_total_);
}

// explicit instantiations
template class MeanFieldMethod<RHF>;
template class MeanFieldMethod<ROHF>;
template class MeanFieldMethod<UHF>;
template class MeanFieldMethod<GHF>;
template class MeanFieldMethod<RKS>;
template class MeanFieldMethod<UKS>;

} // namespace scf
} // namespace cpu
} // namespace ATV
