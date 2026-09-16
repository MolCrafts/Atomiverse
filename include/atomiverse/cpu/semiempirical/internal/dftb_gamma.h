/// @file cpu/semiempirical/internal/dftb_gamma.h
/// @brief DFTB+ Elstner short-range γ (SCC charge kernel) — private.
///
/// Family oracle is DFTB+ 24.1. Full γ is assembled as:
/// \f[ \gamma_{AB}(R) = \frac{1}{R} - S(\tau_A,\tau_B,R),\quad
///     \tau = \frac{16}{5} U = 3.2\,U \f]
/// with on-site \f$\gamma_{AA}=U_A\f$. The short-range piece \f$S\f$ is the
/// published Elstner form (Elstner et al., *Phys. Rev. B* **58**, 7260 (1998),
/// DOI 10.1103/PhysRevB.58.7260, with the review-article sign convention);
/// DFTB+ calls the same expression `expGamma`. Written from the paper, not
/// transcribed from DFTB+ — that code is LGPL-3.0 and cannot be absorbed by
/// this GPL-2.0-only tree. DFTB+ is the numerical oracle here, never the
/// source.
///
/// **Not** Ohno–Klopman \f$1/\sqrt{R^2+\tau^2}\f$ — that form gave H₂O
/// E² ≈ 0.65 eV vs DFTB+ 0.496 eV at the same Δq (see
/// `.claude/notes/dftb-vs-dftbplus-cleanup.md`).
///
/// Maths are evaluated in Hartree/Bohr (DFTB+ native) then converted to the
/// public eV/Å surface used by MioSketch paths.
#pragma once

#include <algorithm>
#include <cmath>

#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {
namespace dftb_gamma {

/// DFTB+ `accuracy.F90` pins (a.u.).
inline constexpr double kTolSameDist = 1.0e-5;
inline constexpr double kMinHubDiff = 0.3125e-5; // 0.3125 * 1e-5 Ha

/// Short-range piece `expGamma` [Ha] — DFTB+ `shortgammafuncs.F90`.
/// Full off-site γ = 1/R − expGamma; on-site is handled as U outside.
inline double exp_gamma_au(double rab, double Ua, double Ub) {
    const double tauA = 3.2 * Ua;
    const double tauB = 3.2 * Ub;
    if (rab < kTolSameDist) {
        if (std::abs(Ua - Ub) < kMinHubDiff)
            return -0.5 * (Ua + Ub);
        return -0.5
               * ((tauA * tauB) / (tauA + tauB)
                  + (tauA * tauB) * (tauA * tauB)
                        / ((tauA + tauB) * (tauA + tauB) * (tauA + tauB)));
    }
    if (std::abs(Ua - Ub) < kMinHubDiff) {
        const double tauMean = 0.5 * (tauA + tauB);
        return std::exp(-tauMean * rab)
               * (1.0 / rab + 0.6875 * tauMean
                  + 0.1875 * rab * tauMean * tauMean
                  + 0.02083333333333333333 * rab * rab * tauMean * tauMean
                        * tauMean);
    }
    auto sub = [](double r, double t1, double t2) {
        const double t1s = t1 * t1;
        const double t2s = t2 * t2;
        const double d = t1s - t2s;
        const double d2 = d * d;
        const double d3 = d2 * d;
        return std::exp(-t1 * r)
               * ((0.5 * t2s * t2s * t1 / d2)
                  - (t2s * t2s * t2s - 3.0 * t2s * t2s * t1s) / (r * d3));
    };
    return sub(rab, tauA, tauB) + sub(rab, tauB, tauA);
}

/// d(expGamma)/dR [Ha/Bohr] — DFTB+ `expGammaPrime`.
inline double exp_gamma_prime_au(double rab, double Ua, double Ub) {
    if (rab < kTolSameDist)
        return 0.0;
    const double tauA = 3.2 * Ua;
    const double tauB = 3.2 * Ub;
    if (std::abs(Ua - Ub) < kMinHubDiff) {
        const double tauMean = 0.5 * (tauA + tauB);
        const double s =
            1.0 / rab + 0.6875 * tauMean + 0.1875 * rab * tauMean * tauMean
            + 0.02083333333333333333 * rab * rab * tauMean * tauMean * tauMean;
        const double sp =
            -1.0 / (rab * rab) + 0.1875 * tauMean * tauMean
            + 2.0 * 0.02083333333333333333 * rab * tauMean * tauMean * tauMean;
        return -tauMean * std::exp(-tauMean * rab) * s
               + std::exp(-tauMean * rab) * sp;
    }
    auto sub_prime = [](double r, double t1, double t2) {
        const double t1s = t1 * t1;
        const double t2s = t2 * t2;
        const double d = t1s - t2s;
        const double d2 = d * d;
        const double d3 = d2 * d;
        const double f = (0.5 * t2s * t2s * t1 / d2)
                         - (t2s * t2s * t2s - 3.0 * t2s * t2s * t1s) / (r * d3);
        const double fp =
            (t2s * t2s * t2s - 3.0 * t2s * t2s * t1s) / (r * r * d3);
        return -t1 * std::exp(-t1 * r) * f + std::exp(-t1 * r) * fp;
    };
    return sub_prime(rab, tauA, tauB) + sub_prime(rab, tauB, tauA);
}

/// Full γ_AB [Ha] at R [Bohr], U [Ha]. On-site → U_A (caller a==b).
inline double gamma_au(double r_bohr, double ua_ha, double ub_ha) {
    if (r_bohr < kTolSameDist)
        return ua_ha; // identical-atom onsite; hetero R≈0 rare
    return 1.0 / r_bohr - exp_gamma_au(r_bohr, ua_ha, ub_ha);
}

/// ∂γ/∂R [Ha/Bohr].
inline double dgamma_dr_au(double r_bohr, double ua_ha, double ub_ha) {
    if (r_bohr < kTolSameDist)
        return 0.0;
    return -1.0 / (r_bohr * r_bohr) - exp_gamma_prime_au(r_bohr, ua_ha, ub_ha);
}

/// Full γ_AB [eV] at R [Å], U [eV].
inline double gamma_eV(double r_A, double ua_eV, double ub_eV) {
    if (r_A < 1e-12)
        return ua_eV;
    const double r = r_A * units::kAngstromInBohr;
    const double ua = ua_eV * units::kEvInHartree;
    const double ub = ub_eV * units::kEvInHartree;
    return gamma_au(r, ua, ub) * units::kHartreeInEv;
}

/// ∂γ/∂R [eV/Å].
inline double dgamma_dr_eV(double r_A, double ua_eV, double ub_eV) {
    if (r_A < 1e-12)
        return 0.0;
    const double r = r_A * units::kAngstromInBohr;
    const double ua = ua_eV * units::kEvInHartree;
    const double ub = ub_eV * units::kEvInHartree;
    // dγ_eV/dR_Å = (dγ_Ha/dR_Bohr) * (dR_Bohr/dR_Å) * (eV/Ha)
    return dgamma_dr_au(r, ua, ub) * units::kAngstromInBohr
           * units::kHartreeInEv;
}

/// Stable step [Ha] for real central-difference ∂/∂U_A on the hetero branch.
///
/// The hetero Elstner form has poles in (τ_A²−τ_B²)^{-3}; steps ≲ 10^{-5} Ha
/// blow up when U_A≈U_B. Relative step ≳ 10^{-3} Ha keeps the sample points
/// off the pole.
inline double hubbard_fd_step_ha(double ua_ha) {
    return std::max(2.0e-3 * std::max(ua_ha, 1.0e-3), 5.0e-4);
}

/// ∂γ_same/∂U at equal Hubbard U (both arguments move together) [a.u.].
/// Used via symmetry: ∂γ/∂U_A |_{U_A=U_B} = ½ ∂γ_same/∂U.
inline double dgamma_same_dU_au(double r_bohr, double u_ha) {
    if (r_bohr < kTolSameDist)
        return 1.0;
    const double h = hubbard_fd_step_ha(u_ha);
    // Equal-U path only (both args shift).
    const double gp = gamma_au(r_bohr, u_ha + h, u_ha + h);
    const double gm = gamma_au(r_bohr, u_ha - h, u_ha - h);
    return (gp - gm) / (2.0 * h);
}

inline double d2gamma_same_dU_dr_au(double r_bohr, double u_ha) {
    if (r_bohr < kTolSameDist)
        return 0.0;
    const double h = hubbard_fd_step_ha(u_ha);
    const double dgp = dgamma_dr_au(r_bohr, u_ha + h, u_ha + h);
    const double dgm = dgamma_dr_au(r_bohr, u_ha - h, u_ha - h);
    return (dgp - dgm) / (2.0 * h);
}

/// ∂γ/∂U_A [dimensionless in a.u.] with U_B held fixed.
inline double dgamma_dUa_au(double r_bohr, double ua_ha, double ub_ha) {
    if (r_bohr < kTolSameDist)
        return 1.0; // γ_AA = U_A
    if (std::abs(ua_ha - ub_ha) < kMinHubDiff)
        // Symmetric equal-U surface: each argument owns half the slope.
        return 0.5 * dgamma_same_dU_au(r_bohr, 0.5 * (ua_ha + ub_ha));
    const double h = hubbard_fd_step_ha(ua_ha);
    const double gp = gamma_au(r_bohr, ua_ha + h, ub_ha);
    const double gm = gamma_au(r_bohr, ua_ha - h, ub_ha);
    return (gp - gm) / (2.0 * h);
}

/// ∂²γ/(∂U_A ∂R) [1/Bohr] with U_B fixed.
inline double d2gamma_dUa_dr_au(double r_bohr, double ua_ha, double ub_ha) {
    if (r_bohr < kTolSameDist)
        return 0.0;
    if (std::abs(ua_ha - ub_ha) < kMinHubDiff)
        return 0.5 * d2gamma_same_dU_dr_au(r_bohr, 0.5 * (ua_ha + ub_ha));
    const double h = hubbard_fd_step_ha(ua_ha);
    const double dgp = dgamma_dr_au(r_bohr, ua_ha + h, ub_ha);
    const double dgm = dgamma_dr_au(r_bohr, ua_ha - h, ub_ha);
    return (dgp - dgm) / (2.0 * h);
}

/// Γ_AB = ∂γ/∂q_A = (∂γ/∂U_A) U^d_A [eV/e].
inline double gamma_charge_eV(double r_A, double ua_eV, double ub_eV,
                              double uda_eV) {
    if (r_A < 1e-12)
        return uda_eV;
    const double r = r_A * units::kAngstromInBohr;
    const double ua = ua_eV * units::kEvInHartree;
    const double ub = ub_eV * units::kEvInHartree;
    // ∂γ_eV/∂U_eV = ∂γ_Ha/∂U_Ha
    return dgamma_dUa_au(r, ua, ub) * uda_eV;
}

/// ∂Γ_AB/∂R = ∂²γ/(∂q_A ∂R) [eV/(e·Å)].
inline double gamma_charge_dr_eV(double r_A, double ua_eV, double ub_eV,
                                 double uda_eV) {
    if (r_A < 1e-12)
        return 0.0;
    const double r = r_A * units::kAngstromInBohr;
    const double ua = ua_eV * units::kEvInHartree;
    const double ub = ub_eV * units::kEvInHartree;
    // ∂²γ_eV/(∂U_eV ∂R_Å) = ∂²γ_Ha/(∂U_Ha ∂R_Bohr) * (dR_Bohr/dR_Å)
    return d2gamma_dUa_dr_au(r, ua, ub) * units::kAngstromInBohr * uda_eV;
}

} // namespace dftb_gamma
} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
