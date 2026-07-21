/// @file cpu/scf/stages/mix_stage.cpp
/// @brief MixStage::launch — DIIS error + Fock-space extrapolation + level
///        shift, byte-for-byte the shipping SCF loop body (steps 2 + 2b).

#include "atomiverse/cpu/scf/stages/mix_stage.h"

#include <algorithm>
#include <cmath>

#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xreducer.hpp>

#include "atomiverse/cpu/scf/internal/level_shift.h"

namespace ATV {
namespace cpu {
namespace scf {

void MixStage::launch(const cpu::LaunchContext& ctx) {
    if (!ctx.at(ATV::scf::Phase::mix))
        return;

    // Step 2: DIIS error from the UNSHIFTED Fock (scf_solver.cpp:246-249).
    err_alpha_ = internal::diis_error(F_alpha_, P_alpha_, S_, X_);
    if (!restricted_)
        err_beta_ = internal::diis_error(F_beta_, P_beta_, S_, X_);

    // SOSCF escalation latch: once max|DIIS error| falls below soscf_start
    // (or from the earliest eligible iteration when soscf_start <= 0), the
    // CIAH Newton step owns the orbital update — the DIIS error above is
    // still built every iteration (the convergence criterion reads it), but
    // extrapolation and the level shift are skipped for the rest of the run.
    //
    // iter >= 1 gate: at iteration 0 a density-only initial guess (MINAO /
    // caller-provided P) coexists with COEFFICIENTS from the prologue's
    // Hcore diagonalization — C and P describe different states, and a
    // Newton rotation of that C is meaningless (observed: MINAO + SOSCF
    // diverged by ~7 Hartree). One Roothaan diagonalize+density round first
    // synchronizes C with the guess; SOSCF may take over from iteration 1.
    if (soscf_ && !soscf_active_ && iter_ >= 1) {
        double err_inf = xt::amax(xt::abs(err_alpha_))();
        if (!restricted_)
            err_inf = std::max(err_inf, xt::amax(xt::abs(err_beta_))());
        if (soscf_start_ <= 0.0 || err_inf < soscf_start_)
            soscf_active_ = true;
    }
    if (soscf_active_)
        return;

    // Far-field escalation latch: once max|DIIS error| falls below the
    // switch tolerance, an EDIIS/ADIIS run permanently hands over to CDIIS
    // (EDIIS/ADIIS stall near convergence — far-field mixers only).
    bool cdiis_now = (mixer_kind_ == MixerKind::CDIIS) || switched_;
    if (!cdiis_now) {
        double err_max = xt::amax(xt::abs(err_alpha_))();
        if (!restricted_)
            err_max = std::max(err_max, xt::amax(xt::abs(err_beta_))());
        if (err_max < mixer_switch_tol_) {
            switched_ = true;
            cdiis_now = true;
        }
    }

    // Step 2: mixer extrapolation in Fock space. The CDIIS branch is
    // byte-for-byte the shipping loop (scf_solver.cpp:251-263); EDIIS/ADIIS
    // extrapolate from their own (F, D[, E]) histories and push the
    // UNSHIFTED Fock exactly like CDIIS does.
    if (use_diis_ && iter_ >= diis_start_iter_) {
        if (cdiis_now) {
            if (restricted_) {
                F_alpha_ = diis_.step(F_alpha_, err_alpha_);
                F_beta_ = F_alpha_;
            } else {
                auto [Fa_diis, Fb_diis] =
                    diis_.step(F_alpha_, err_alpha_, F_beta_, err_beta_);
                F_alpha_ = Fa_diis;
                F_beta_ = Fb_diis;
            }
        } else if (mixer_kind_ == MixerKind::EDIIS_CDIIS) {
            if (restricted_) {
                F_alpha_ = ediis_.step(F_alpha_, P_alpha_, energy_);
                F_beta_ = F_alpha_;
            } else {
                auto [Fa_e, Fb_e] = ediis_.step(F_alpha_, P_alpha_, F_beta_,
                                                P_beta_, energy_);
                F_alpha_ = Fa_e;
                F_beta_ = Fb_e;
            }
        } else { // MixerKind::ADIIS_CDIIS
            if (restricted_) {
                F_alpha_ = adiis_.step(F_alpha_, P_alpha_);
                F_beta_ = F_alpha_;
            } else {
                auto [Fa_a, Fb_a] =
                    adiis_.step(F_alpha_, P_alpha_, F_beta_, P_beta_);
                F_alpha_ = Fa_a;
                F_beta_ = Fb_a;
            }
        }
    }

    // Step 2b: Saunders-Hillier level shift, strictly after DIIS (which built
    // its error and extrapolated the unshifted Fock) and strictly before
    // diagonalization (scf_solver.cpp:270-282). Restricted D_occ = 0.5*P
    // (occupancy-1 convention); unrestricted P_sigma is already S-idempotent.
    const bool shift_active =
        std::abs(level_shift_) > internal::kLevelShiftSkipThreshold;
    if (shift_active) {
        if (restricted_) {
            RealMatrix D_occ = 0.5 * P_alpha_;
            F_alpha_ =
                internal::apply_level_shift(F_alpha_, S_, D_occ, level_shift_);
            F_beta_ = F_alpha_;
        } else {
            F_alpha_ = internal::apply_level_shift(F_alpha_, S_, P_alpha_,
                                                   level_shift_);
            F_beta_ =
                internal::apply_level_shift(F_beta_, S_, P_beta_, level_shift_);
        }
    }
}

} // namespace scf
} // namespace cpu
} // namespace ATV
