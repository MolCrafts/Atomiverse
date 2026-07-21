/// @file cpu/scf/xc_functional.cpp
/// @brief LSDA (Slater + VWN5), PBE GGA, and VWN-RPA exchange-correlation
///        functionals. All formulas use atomic units (Hartree, Bohr).
///
/// LSDA: Dirac-Slater exchange + VWN5 correlation (Vosko-Wilk-Nusair
/// 1980, parameterization V). PBE: Perdew-Burke-Ernzerhof GGA. VWN-RPA:
/// RPA-parametrized VWN LDA correlation (libxc id 8), the LDA correlation
/// component of B3LYP. The B88 / LYP GGA building blocks live in the
/// sibling translation unit xc_gga.cpp.

#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cpu/scf/xc_internal.h"

#include <algorithm>
#include <cmath>

namespace ATV {
namespace cpu {
namespace scf {

// ── Constants ────────────────────────────────────────────────────────────

static constexpr double kPi = 3.14159265358979323846;
static constexpr double kRhoMin = 1e-12; // total density cutoff
static constexpr double kSpinDensMin =
    1e-15; // per-spin density floor (matches libxc dens_threshold)
static constexpr double kSigmaMin = 1e-20; // gradient cutoff for GGA

// Slater exchange constant: C_x = (3/4)(3/π)^{1/3}
static constexpr double kCx =
    0.7385587663820224; // (3.0/4.0) * std::cbrt(3.0/kPi)

// ── VWN5 parameters (Hartree convention, matching libxc) ─────────────────
// A stored as A_Ha = A_Ry/2 (libxc maple/vwn.mpl). VWN5Params is declared
// in xc_internal.h (shared with vwn_rpa_correlation).

// Paramagnetic: A_Ry = 0.0621814 -> A_Ha = 0.0310907
static constexpr VWN5Params kVWN5_P = {0.0310907, -0.10498, 3.72744, 12.9352};
// Ferromagnetic: A_Ry = 0.0310907 -> A_Ha = 0.01554535
static constexpr VWN5Params kVWN5_F = {0.01554535, -0.32500, 7.06042, 18.0578};
// Spin stiffness alpha_c: A_Ry = -1/(3*pi^2) -> A_Ha = -1/(6*pi^2)
static constexpr VWN5Params kVWN5_A = {-1.0 / (6.0 * kPi * kPi), -0.0047584,
                                       1.13107, 13.0045};

// ── VWN-RPA parameters (libxc maple/vwn.mpl, RPA columns) ────────────────
// XC_LDA_C_VWN_RPA (id 8): same Pade form as VWN5, RPA (A, x0, b, c).
static constexpr VWN5Params kVWN_RPA_P = {0.0310907, -0.409286, 13.0720,
                                          42.7198};
static constexpr VWN5Params kVWN_RPA_F = {0.01554535, -0.743294, 20.1231,
                                          101.578};

// ── VWN5 Pade approximant F(x) ──────────────────────────────────────────

// F(x) = A { ln(x²/X(x)) + 2b/Q arctan(Q/(2x+b))
//          - bx₀/X(x₀) [ln((x-x₀)²/X(x)) + 2(b+2x₀)/Q arctan(Q/(2x+b))] }
// where X(x) = x² + bx + c,  Q = √(4c - b²)
double vwn5_F(double x, const VWN5Params& p) {
    double X = x * x + p.b * x + p.c;
    double X0 = p.x0 * p.x0 + p.b * p.x0 + p.c;
    double Q = std::sqrt(4.0 * p.c - p.b * p.b);
    double atanQ = std::atan2(Q, 2.0 * x + p.b);

    double term1 = std::log(x * x / X);
    double term2 = 2.0 * p.b / Q * atanQ;
    double term3 = -p.b * p.x0 / X0
                   * (std::log((x - p.x0) * (x - p.x0) / X)
                      + 2.0 * (p.b + 2.0 * p.x0) / Q * atanQ);

    return p.A * (term1 + term2 + term3);
}

// Derivative dF/dx for VWN5
double vwn5_dFdx(double x, const VWN5Params& p) {
    double X = x * x + p.b * x + p.c;
    double X0 = p.x0 * p.x0 + p.b * p.x0 + p.c;
    double Q = std::sqrt(4.0 * p.c - p.b * p.b);

    // d/dx [ln(x²/X)] = 2/x - (2x+b)/X
    double dterm1 = 2.0 / x - (2.0 * x + p.b) / X;

    // d/dx [2b/Q arctan(Q/(2x+b))] = -2b/(X) * Q/(Q² + (2x+b)²) * 2
    // Simplify: = -4b / (Q * X) ... wait, let's be precise.
    // arctan'(u) = 1/(1+u²), u = Q/(2x+b), u' = -2Q/(2x+b)²
    // d/dx = 2b/Q * 1/(1 + Q²/(2x+b)²) * (-2Q/(2x+b)²)
    //       = 2b/Q * (2x+b)² / ((2x+b)² + Q²) * (-2Q/(2x+b)²)
    //       = 2b * (-2) / ((2x+b)² + Q²)
    // Note: (2x+b)² + Q² = 4x² + 4bx + b² + 4c - b² = 4(x²+bx+c) = 4X
    double dterm2 = -4.0 * p.b / (4.0 * X); // = -b/X

    // d/dx [-bx₀/X₀ * {ln((x-x₀)²/X) + 2(b+2x₀)/Q arctan(...)}]
    double fac = -p.b * p.x0 / X0;
    double dlog = 2.0 / (x - p.x0) - (2.0 * x + p.b) / X;
    double datan = -4.0 * (p.b + 2.0 * p.x0) / (4.0 * X); // same simplification
    double dterm3 = fac * (dlog + datan);

    return p.A * (dterm1 + dterm2 + dterm3);
}

// ── Spin interpolation function f(ζ) ────────────────────────────────────

// f(ζ) = [(1+ζ)^{4/3} + (1-ζ)^{4/3} - 2] / [2(2^{1/3} - 1)]
static const double kFppZero = 4.0 / (9.0 * (std::pow(2.0, 1.0 / 3.0) - 1.0));
// f''(0) = 4 / (9 * (2^{1/3} - 1))

static double spin_f(double zeta) {
    double denom = 2.0 * (std::pow(2.0, 1.0 / 3.0) - 1.0);
    return (std::pow(1.0 + zeta, 4.0 / 3.0) + std::pow(1.0 - zeta, 4.0 / 3.0)
            - 2.0)
           / denom;
}

// df/dζ
static double spin_df(double zeta) {
    double denom = 2.0 * (std::pow(2.0, 1.0 / 3.0) - 1.0);
    return (4.0 / 3.0)
           * (std::pow(1.0 + zeta, 1.0 / 3.0) - std::pow(1.0 - zeta, 1.0 / 3.0))
           / denom;
}

// ── Slater exchange ─────────────────────────────────────────────────────

// Spin-polarized Slater exchange:
//   ε_x = -2^{1/3} C_x [ρ_α^{4/3} + ρ_β^{4/3}] / ρ
//   v_x^σ = -(3/π)^{1/3} 2^{1/3} ρ_σ^{1/3}  (= -(6/π)^{1/3} ρ_σ^{1/3} * (4/3) ... )
//
// Actually, the standard form:
//   E_x = -C_x_spin * ∫ [ρ_α^{4/3} + ρ_β^{4/3}] dτ
//   where C_x_spin = 2^{1/3} * C_x = 2^{1/3} * (3/4)(3/π)^{1/3}
//
//   Energy density (per volume): f_x = -C_x_spin * [ρ_α^{4/3} + ρ_β^{4/3}]
//   ε_x = f_x / ρ  (energy per electron, but we track f_x = exc * rho)
//
//   v_x^α = ∂f_x/∂ρ_α = -(4/3) C_x_spin ρ_α^{1/3}
//   v_x^β = ∂f_x/∂ρ_β = -(4/3) C_x_spin ρ_β^{1/3}

static void slater_exchange(double rho_a, double rho_b, double& exc,
                            double& vx_a, double& vx_b) {
    double Cx_spin = std::pow(2.0, 1.0 / 3.0) * kCx;

    double ra43 = std::pow(rho_a, 4.0 / 3.0);
    double rb43 = std::pow(rho_b, 4.0 / 3.0);

    // f_x = -Cx_spin * (ρ_α^{4/3} + ρ_β^{4/3})  (energy per volume)
    double fx = -Cx_spin * (ra43 + rb43);
    double rho = rho_a + rho_b;
    exc = fx / rho; // energy per electron

    vx_a = -(4.0 / 3.0) * Cx_spin * std::pow(rho_a, 1.0 / 3.0);
    vx_b = -(4.0 / 3.0) * Cx_spin * std::pow(rho_b, 1.0 / 3.0);
}

// ── PW92 correlation (LSDA base for PBE) ────────────────────────────────
//
// Perdew & Wang, Phys. Rev. B 45, 13244 (1992).
// Used by libxc as the LDA correlation base for PBE (XC_GGA_C_PBE) per
// the PBE paper (Perdew-Burke-Ernzerhof, PRL 77, 3865 (1996)). VWN5 is
// NOT correct for PBE -- VWN5 and PW92 disagree by ~1e-4 Ha at typical
// densities, which propagates into a uniform PBE bias if mixed.
//
// PW92 form (same spin interpolation as VWN5 but with G(r_s) from the
// PW92 paper Table I, replacing VWN5's Pade ansatz):
//   ε_c(r_s, ζ) = ε_c^P(r_s)
//               + α_c(r_s) f(ζ)/f''(0) (1 - ζ^4)
//               + [ε_c^F(r_s) - ε_c^P(r_s)] f(ζ) ζ^4
//
//   G(r_s; A, α1, β1, β2, β3, β4, p) =
//       -2A(1 + α1 r_s) ln{1 + 1/[2A(β1 √r_s + β2 r_s + β3 r_s^{3/2} + β4 r_s^{p+1})]}
//
// libxc source studied: libxc/maple/lda_c_pw.mpl and libxc/src/work_lda.c.

struct PW92Params {
    double A, alpha1, beta1, beta2, beta3, beta4, p;
};

// Paramagnetic (Table I, ζ = 0)
static constexpr PW92Params kPW92_P = {0.0310907, 0.21370, 7.5957, 3.5876,
                                       1.6382,    0.49294, 1.0};
// Ferromagnetic (Table I, ζ = 1)
static constexpr PW92Params kPW92_F = {0.01554535, 0.20548, 14.1189, 6.1977,
                                       3.3662,     0.62517, 1.0};
// Spin-stiffness -α_c (Table I, "α_c" row, stored as positive G; negated when used)
static constexpr PW92Params kPW92_A = {0.0168869, 0.11125, 10.357, 3.6231,
                                       0.88026,   0.49671, 1.0};

// Returns G(r_s) and dG/dr_s for PW92 parameter set p.
static void pw92_G(double rs, const PW92Params& p, double& G, double& dG_drs) {
    double sqrt_rs = std::sqrt(rs);
    double rs_3_2 = rs * sqrt_rs;
    double rs_p1 = std::pow(rs, p.p + 1.0);
    double rs_p = std::pow(rs, p.p);

    // q1 = 2A(β1 √rs + β2 rs + β3 rs^{3/2} + β4 rs^{p+1})
    double q1 = 2.0 * p.A
                * (p.beta1 * sqrt_rs + p.beta2 * rs + p.beta3 * rs_3_2
                   + p.beta4 * rs_p1);
    double dq1_drs =
        2.0 * p.A
        * (0.5 * p.beta1 / sqrt_rs + p.beta2 + 1.5 * p.beta3 * sqrt_rs
           + (p.p + 1.0) * p.beta4 * rs_p);

    // q2 = ln(1 + 1/q1)
    double q2 = std::log(1.0 + 1.0 / q1);
    // dq2/drs = -1/q1² * dq1/drs / (1 + 1/q1) = -dq1/drs / (q1 * (q1 + 1))
    double dq2_drs = -dq1_drs / (q1 * (q1 + 1.0));

    // G = -2A(1 + α1 rs) q2
    double prefactor = 1.0 + p.alpha1 * rs;
    G = -2.0 * p.A * prefactor * q2;
    dG_drs = -2.0 * p.A * (p.alpha1 * q2 + prefactor * dq2_drs);
}

static void pw92_correlation(double rho_a, double rho_b, double& ec,
                             double& vc_a, double& vc_b) {
    rho_a = std::max(rho_a, kSpinDensMin);
    rho_b = std::max(rho_b, kSpinDensMin);
    double rho = rho_a + rho_b;
    double rs = std::pow(3.0 / (4.0 * kPi * rho), 1.0 / 3.0);
    double zeta = (rho_a - rho_b) / rho;

    double ecP, ecF, ac, decP_drs, decF_drs, dac_drs;
    pw92_G(rs, kPW92_P, ecP, decP_drs);
    pw92_G(rs, kPW92_F, ecF, decF_drs);
    pw92_G(rs, kPW92_A, ac, dac_drs);
    // The stiffness row's G function gives -α_c; negate to get α_c.
    ac = -ac;
    dac_drs = -dac_drs;

    double fz = spin_f(zeta);
    double z4 = zeta * zeta * zeta * zeta;

    // ε_c
    ec = ecP + ac * fz / kFppZero * (1.0 - z4) + (ecF - ecP) * fz * z4;

    // dε_c/drs
    double dec_drs = decP_drs + dac_drs * fz / kFppZero * (1.0 - z4)
                     + (decF_drs - decP_drs) * fz * z4;

    // dε_c/dζ
    double dfz = spin_df(zeta);
    double dz4 = 4.0 * zeta * zeta * zeta;
    double dec_dz = ac / kFppZero * (dfz * (1.0 - z4) - fz * dz4)
                    + (ecF - ecP) * (dfz * z4 + fz * dz4);

    // v_c^σ = ε_c - (rs/3) dε_c/drs - (ζ ∓ 1) dε_c/dζ
    // (using -(ζ-1) for α, -(ζ+1) for β -- canonical sign convention)
    double rs3_dec = (rs / 3.0) * dec_drs;
    vc_a = ec - rs3_dec - (zeta - 1.0) * dec_dz;
    vc_b = ec - rs3_dec - (zeta + 1.0) * dec_dz;
}

// ── VWN5 correlation ────────────────────────────────────────────────────

// ε_c(r_s, ζ) = ε_c^P(r_s) + α_c(r_s) f(ζ)/f''(0) (1 - ζ⁴)
//             + [ε_c^F(r_s) - ε_c^P(r_s)] f(ζ) ζ⁴
//
// where r_s = (3/(4πρ))^{1/3}, ζ = (ρ_α - ρ_β)/ρ

static void vwn5_correlation(double rho_a, double rho_b, double& ec,
                             double& vc_a, double& vc_b) {
    // Per-spin density floor matching libxc work_lda clamping.
    // Prevents exact zeta = +/-1 while keeping exchange unaffected.
    rho_a = std::max(rho_a, kSpinDensMin);
    rho_b = std::max(rho_b, kSpinDensMin);

    double rho = rho_a + rho_b;
    double rs = std::pow(3.0 / (4.0 * kPi * rho), 1.0 / 3.0);
    double x = std::sqrt(rs);
    double zeta = (rho_a - rho_b) / rho;

    // Evaluate the three VWN5 functions
    double ecP = vwn5_F(x, kVWN5_P);
    double ecF = vwn5_F(x, kVWN5_F);
    double ac = vwn5_F(x, kVWN5_A);

    double fz = spin_f(zeta);
    double z4 = zeta * zeta * zeta * zeta;

    // ε_c = ε_c^P + α_c f(ζ)/f''(0) (1 - ζ⁴) + [ε_c^F - ε_c^P] f(ζ) ζ⁴
    ec = ecP + ac * fz / kFppZero * (1.0 - z4) + (ecF - ecP) * fz * z4;

    // For the potential v_c^σ = ε_c - (r_s/3) dε_c/dr_s + (∓ζ - 1)(dε_c/dζ)
    // where ∓ = - for α, + for β (i.e., -(ζ-1) for α, -(ζ+1) for β... let's be precise)
    //
    // v_c^α = ε_c - (r_s/3) ∂ε_c/∂r_s - (ζ - 1) ∂ε_c/∂ζ
    // v_c^β = ε_c - (r_s/3) ∂ε_c/∂r_s - (ζ + 1) ∂ε_c/∂ζ
    //
    // Since x = √r_s, ∂/∂r_s = (1/(2x)) ∂/∂x
    // So (r_s/3) ∂ε_c/∂r_s = (x²/3)(1/(2x)) ∂ε_c/∂x = x/(6) ∂ε_c/∂x

    double decP_dx = vwn5_dFdx(x, kVWN5_P);
    double decF_dx = vwn5_dFdx(x, kVWN5_F);
    double dac_dx = vwn5_dFdx(x, kVWN5_A);

    double dec_dx = decP_dx + dac_dx * fz / kFppZero * (1.0 - z4)
                    + (decF_dx - decP_dx) * fz * z4;

    double rs3_dec = x / 6.0 * dec_dx;

    // ∂ε_c/∂ζ
    double dfz = spin_df(zeta);
    double dz4 = 4.0 * zeta * zeta * zeta;

    double dec_dz = ac / kFppZero * (dfz * (1.0 - z4) - fz * dz4)
                    + (ecF - ecP) * (dfz * z4 + fz * dz4);

    vc_a = ec - rs3_dec - (zeta - 1.0) * dec_dz;
    vc_b = ec - rs3_dec - (zeta + 1.0) * dec_dz;
}

// ── VWN-RPA correlation ─────────────────────────────────────────────────

/// @brief RPA-parametrized VWN LDA correlation (libxc XC_LDA_C_VWN_RPA,
///        id 8), spin-polarized.
///
/// Reuses the VWN5 Pade machinery (`vwn5_F` / `vwn5_dFdx`) with the RPA
/// parameter columns and a two-channel spin blend
/// eps_c = F_P(1 - f(zeta)) + F_F f(zeta); unlike VWN5 the RPA energy
/// uses no spin-stiffness channel.
///
/// @param rho_a Alpha spin density [electrons/Bohr^3].
/// @param rho_b Beta spin density [electrons/Bohr^3].
/// @param[out] ec   Correlation energy density per electron [Hartree].
/// @param[out] vc_a d(rho*ec)/drho_alpha, alpha potential [Hartree].
/// @param[out] vc_b d(rho*ec)/drho_beta, beta potential [Hartree].
void vwn_rpa_correlation(double rho_a, double rho_b, double& ec, double& vc_a,
                         double& vc_b) {
    // Per-spin density floor matching libxc work_lda clamping.
    rho_a = std::max(rho_a, kSpinDensMin);
    rho_b = std::max(rho_b, kSpinDensMin);

    double rho = rho_a + rho_b;
    double x = std::sqrt(std::pow(3.0 / (4.0 * kPi * rho), 1.0 / 3.0));
    double zeta = (rho_a - rho_b) / rho;

    // RPA paramagnetic / ferromagnetic Pade channels.
    double F_P = vwn5_F(x, kVWN_RPA_P);
    double F_F = vwn5_F(x, kVWN_RPA_F);
    double fz = spin_f(zeta);
    ec = F_P * (1.0 - fz) + F_F * fz; // eps_c = F_P(1-f) + F_F f

    // d eps_c / dx, then (rs/3) d eps_c/drs = (x/6) d eps_c/dx.
    double dec_dx =
        vwn5_dFdx(x, kVWN_RPA_P) * (1.0 - fz) + vwn5_dFdx(x, kVWN_RPA_F) * fz;
    double rs3_dec = x / 6.0 * dec_dx;

    // d eps_c / d zeta = (F_F - F_P) f'(zeta).
    double dec_dz = (F_F - F_P) * spin_df(zeta);

    vc_a = ec - rs3_dec - (zeta - 1.0) * dec_dz;
    vc_b = ec - rs3_dec - (zeta + 1.0) * dec_dz;
}

// ── PBE exchange ────────────────────────────────────────────────────────

// PBE exchange enhancement factor:
//   F_x(s) = 1 + κ - κ / (1 + μ s² / κ)
//   s = |∇ρ_σ| / (2 (6π²)^{1/3} ρ_σ^{4/3})
//   κ = 0.804,  μ = 0.21951
//
// The total exchange per spin channel:
//   E_x^PBE = E_x^LSDA * (1 + enhancement from F_x)
//
// Implementation: compute per-spin exchange with enhancement.

static constexpr double kPBE_kappa = 0.804;
static constexpr double kPBE_mu = 0.2195149727645171; // beta*pi^2/3

// Compute PBE exchange for a single spin channel.
// Returns: (energy density contribution, v_rho, v_sigma) for this spin.
static void pbe_exchange_spin(double rho_s, double sigma_ss, double& ex_s,
                              double& vrho_s, double& vsigma_ss) {
    if (rho_s < kRhoMin * 0.5) {
        ex_s = 0;
        vrho_s = 0;
        vsigma_ss = 0;
        return;
    }

    double Cx_spin = std::pow(2.0, 1.0 / 3.0) * kCx;

    // LSDA exchange for this spin (energy per volume = -Cx_spin * ρ_σ^{4/3})
    double rs43 = std::pow(rho_s, 4.0 / 3.0);
    double ex_lda = -Cx_spin * rs43; // per volume

    // Reduced gradient s
    double kf =
        std::pow(6.0 * kPi * kPi, 1.0 / 3.0) * std::pow(rho_s, 1.0 / 3.0);
    double grad = std::sqrt(std::max(sigma_ss, 0.0));
    double s = grad / (2.0 * kf * rho_s);
    double s2 = s * s;

    // Enhancement factor F_x(s)
    double denom = 1.0 + kPBE_mu * s2 / kPBE_kappa;
    double Fx = 1.0 + kPBE_kappa - kPBE_kappa / denom;

    // Energy density (per volume) for this spin
    ex_s = ex_lda * Fx;

    // dF_x/ds² = μ / (1 + μs²/κ)²
    double dFx_ds2 = kPBE_mu / (denom * denom);

    // ds²/dρ_σ: s² = σ / (4 k_f² ρ²), k_f = (6π²)^{1/3} ρ^{1/3}
    // s² = σ / (4 (6π²)^{2/3} ρ^{2/3} ρ²) = σ / (4 (6π²)^{2/3} ρ^{8/3})
    // ds²/dρ = -(8/3) s² / ρ
    double ds2_drho = -(8.0 / 3.0) * s2 / rho_s;

    // ds²/dσ = 1 / (4 (6π²)^{2/3} ρ^{8/3})
    double kf2_fac =
        4.0 * std::pow(6.0 * kPi * kPi, 2.0 / 3.0) * std::pow(rho_s, 8.0 / 3.0);
    double ds2_dsigma = 1.0 / kf2_fac;

    // v_rho = d(ex_lda * Fx)/dρ = (d ex_lda/dρ) Fx + ex_lda * dFx/dρ
    double dex_lda_drho = -(4.0 / 3.0) * Cx_spin * std::pow(rho_s, 1.0 / 3.0);
    vrho_s = dex_lda_drho * Fx + ex_lda * dFx_ds2 * ds2_drho;

    // v_sigma = ex_lda * dFx/dσ
    vsigma_ss = ex_lda * dFx_ds2 * ds2_dsigma;
}

// ── PBE correlation ─────────────────────────────────────────────────────

// PBE correlation gradient correction:
//   H(r_s, ζ, t) = γ φ³ ln{1 + (β/γ) t² [(1 + At²)/(1 + At² + A²t⁴)]}
//   A = (β/γ) / [exp(-ε_c^unif/(γφ³)) - 1]
//   t = |∇ρ| / (2 φ k_s ρ)
//   φ(ζ) = [(1+ζ)^{2/3} + (1-ζ)^{2/3}] / 2
//   k_s = √(4k_F/π),  k_F = (3π²ρ)^{1/3}
//   β = 0.066725,  γ = (1 - ln2)/π² ≈ 0.031091

static constexpr double kPBE_beta = 0.06672455060314922;
static constexpr double kPBE_gamma = 0.031090690869654895034; // (1 - ln2)/π²

static double phi_pbe(double zeta) {
    return 0.5
           * (std::pow(1.0 + zeta, 2.0 / 3.0)
              + std::pow(1.0 - zeta, 2.0 / 3.0));
}

static void pbe_correlation(double rho_a, double rho_b, double ec_lda,
                            double vc_lda_a, double vc_lda_b, double sigma_aa,
                            double sigma_ab, double sigma_bb, double& ec_pbe,
                            double& vc_a, double& vc_b, double& vsigma_aa,
                            double& vsigma_ab, double& vsigma_bb) {
    double rho = rho_a + rho_b;
    double zeta = (rho_a - rho_b) / rho;
    zeta = std::max(-1.0 + 1e-14, std::min(1.0 - 1e-14, zeta));

    double phi = phi_pbe(zeta);
    double phi3 = phi * phi * phi;

    // k_F and k_s
    double kF = std::pow(3.0 * kPi * kPi * rho, 1.0 / 3.0);
    double ks = std::sqrt(4.0 * kF / kPi);

    // t = |∇ρ| / (2 φ k_s ρ)
    double sigma_total = sigma_aa + 2.0 * sigma_ab + sigma_bb;
    double grad_rho = std::sqrt(std::max(sigma_total, 0.0));
    double t = grad_rho / (2.0 * phi * ks * rho);
    double t2 = t * t;

    // A coefficient
    double exp_arg = -ec_lda / (kPBE_gamma * phi3);
    double exp_val = std::exp(exp_arg);
    double A = (kPBE_beta / kPBE_gamma) / (exp_val - 1.0);

    // H(r_s, ζ, t)
    double At2 = A * t2;
    double numer = 1.0 + At2;
    double denom = 1.0 + At2 + A * A * t2 * t2;
    double ratio = (kPBE_beta / kPBE_gamma) * t2 * numer / denom;
    double H = kPBE_gamma * phi3 * std::log(1.0 + ratio);

    ec_pbe = ec_lda + H;

    // For the derivatives, we use the chain rule.
    // dH/dt² = γφ³ * 1/(1+ratio) * d(ratio)/dt²
    // This is complex; for now we use numerical-like analytic forms.

    // d(ratio)/dt²:
    // ratio = (β/γ) t² (1+At²) / (1+At²+A²t⁴)
    // Let u = At², then ratio = (β/γ) t² (1+u) / (1+u+u²)
    // d(ratio)/dt² = (β/γ) * [(1+u)/(1+u+u²) + t² * d/dt² {(1+u)/(1+u+u²)}]
    // where du/dt² = A
    double u = At2;
    double one_u = 1.0 + u;
    double one_u_u2 = 1.0 + u + u * u;
    double d_frac_du =
        (1.0 * one_u_u2 - one_u * (1.0 + 2.0 * u)) / (one_u_u2 * one_u_u2);
    // = (1 + u + u² - (1+u)(1+2u)) / (1+u+u²)²
    // = (1 + u + u² - 1 - 2u - u - 2u²) / ...
    // = (-2u - u²) / (1+u+u²)²
    // = -u(2+u) / (1+u+u²)²

    double dratio_dt2 =
        (kPBE_beta / kPBE_gamma) * (one_u / one_u_u2 + t2 * A * d_frac_du);

    double dH_dt2 = kPBE_gamma * phi3 / (1.0 + ratio) * dratio_dt2;

    // dt²/dσ_total: t² = σ_total / (4 φ² k_s² ρ²)
    double dt2_dsigma = 1.0 / (4.0 * phi * phi * ks * ks * rho * rho);

    // dt²/dρ = -(7/3) t²/ρ  (t ~ ρ^{-7/6}, t² ~ ρ^{-7/3})
    // t² = σ/(4φ²ks²ρ²), ks² = 4kF/π, kF ~ ρ^{1/3}
    // t² ~ σ / (φ² ρ^{1/3} ρ²) = σ / (φ² ρ^{7/3})
    // dt²/dρ = -(7/3) t²/ρ  (at fixed σ and ζ)

    // For v_sigma: H depends on σ_total only through t²
    double dH_dsigma_total = dH_dt2 * dt2_dsigma;

    // σ_total = σ_αα + 2σ_αβ + σ_ββ
    vsigma_aa = rho * dH_dsigma_total;
    vsigma_ab = 2.0 * rho * dH_dsigma_total;
    vsigma_bb = rho * dH_dsigma_total;

    // For v_rho: more complex — H depends on ρ through t, k_s, A, ec_lda, phi
    // Use the identity for uniform scaling:
    // v_c^σ = (ec_lda + H) - (r_s/3) d(ec_lda+H)/dr_s ∓ (ζ∓1) d(ec_lda+H)/dζ
    // But this gets very involved. A simpler approach for correctness:
    //
    // v^σ_c = vc_lda_σ + H + dH/dρ_σ (at fixed σ's)
    //
    // dH/dρ_α = (∂H/∂ρ)|_ζ * (∂ρ/∂ρ_α) + (∂H/∂ζ)|_ρ * (∂ζ/∂ρ_α)
    //         + (∂H/∂ec_lda) * (∂ec_lda/∂ρ_α)
    //
    // This is still complex. For a robust first implementation:
    // v_c^α = vc_lda_a + H - (σ_total / rho) * dH_dsigma_total
    //         + correction terms...
    //
    // Actually, the simplest correct form for the KS potential contribution:
    // The PBE correlation to the Fock matrix is handled via the grid integration.
    // The key outputs for the grid assembly are:
    //   vrho_α, vrho_β: partial derivatives of f_c w.r.t. ρ_α, ρ_β
    //   vsigma_αα, vsigma_αβ, vsigma_ββ: partial derivatives w.r.t. σ's
    //
    // where f_c = ρ * ε_c(ρ_α, ρ_β, σ's).
    //
    // For the rho derivatives, we need d(ρ·(ec_lda + H))/dρ_α.
    // = (ec_lda + H) + ρ * d(ec_lda + H)/dρ_α
    // The LDA part gives vc_lda_a = d(ρ·ec_lda)/dρ_α.
    // So we need: vc_lda_a + H + ρ * dH/dρ_α.
    //
    // For numerical stability and correctness, compute dH/drho_a and dH/drho_b
    // using direct per-spin derivatives.  We avoid computing dphi/dzeta, which
    // diverges at full polarization (zeta = +/-1), by computing dphi/drho_sigma
    // directly with a libxc-compatible zeta threshold.
    //
    // phi = [(1+zeta)^{2/3} + (1-zeta)^{2/3}]/2
    // Define za = 1+zeta = 2*rho_a/rho, zb = 1-zeta = 2*rho_b/rho.
    // When za or zb < zeta_threshold, that term's derivative is clamped to 0.

    static constexpr double kZetaThreshold =
        2.2204460492503131e-16; // DBL_EPSILON

    double za = 1.0 + zeta;
    double zb = 1.0 - zeta;

    // dphi/drho_sigma: direct computation avoiding dphi/dzeta singularity.
    // dzeta/drho_a = (1-zeta)/rho = zb/rho
    // dzeta/drho_b = -(1+zeta)/rho = -za/rho
    //
    // dphi/dzeta = (1/3)*[za^{-1/3} - zb^{-1/3}] / ... but we don't compute this.
    // Instead: dphi/drho_sigma = sum of per-term derivatives
    //   d/drho_a [(1+zeta)^{2/3}] / 2 = (2/3)(1+zeta)^{-1/3} * (dzeta/drho_a) / 2
    //                                   = (1/3)*za^{-1/3} * zb/rho  [if za > threshold]
    //   d/drho_a [(1-zeta)^{2/3}] / 2 = (2/3)(1-zeta)^{-1/3} * (-dzeta/drho_a) / 2
    //                                   = -(1/3)*zb^{-1/3} * zb/rho = -(1/3)*zb^{2/3}/rho  [if zb > threshold]
    //
    // Note: the zb^{-1/3} * zb = zb^{2/3} form is safe even as zb -> 0, but
    // for consistency with libxc we still apply the threshold.

    double dphi_drho_a, dphi_drho_b;
    {
        double term_a_da =
            (za > kZetaThreshold)
                ? (1.0 / 3.0) * std::pow(za, -1.0 / 3.0) * zb / rho
                : 0.0;
        double term_b_da = (zb > kZetaThreshold)
                               ? -(1.0 / 3.0) * std::pow(zb, 2.0 / 3.0) / rho
                               : 0.0;
        dphi_drho_a = term_a_da + term_b_da;

        // Correction: re-derive term_b_da properly.
        // d/drho_a [(1-zeta)^{2/3}] = (2/3)(1-zeta)^{-1/3} * d(1-zeta)/drho_a
        //   d(1-zeta)/drho_a = -dzeta/drho_a = -zb/rho
        // So d/drho_a [(1-zeta)^{2/3}] = (2/3)*zb^{-1/3}*(-zb/rho) = -(2/3)*zb^{2/3}/rho
        // Then /2 for phi: -(1/3)*zb^{2/3}/rho  -- this matches above.

        double term_a_db = (za > kZetaThreshold)
                               ? -(1.0 / 3.0) * std::pow(za, 2.0 / 3.0) / rho
                               : 0.0;
        double term_b_db =
            (zb > kZetaThreshold)
                ? (1.0 / 3.0) * std::pow(zb, -1.0 / 3.0) * za / rho
                : 0.0;
        dphi_drho_b = term_a_db + term_b_db;
    }

    // dt^2/drho_a, dt^2/drho_b: t^2 = sigma/(4*phi^2*ks^2*rho^2)
    // t^2 depends on rho_sigma through rho, ks (via kF ~ rho^{1/3}), and phi (via zeta).
    // At fixed sigma and zeta: dt^2/drho = -(7/3)*t^2/rho
    // Zeta dependence:         dt^2/dphi = -2*t^2/phi (since t^2 ~ 1/phi^2)
    double dt2_drho = -(7.0 / 3.0) * t2 / rho;
    double dt2_dphi = -2.0 * t2 / phi;

    // dA/drho_sigma: A depends on ec_lda (through vc_lda) and phi.
    // A = (beta/gamma) / (exp(-ec_lda/(gamma*phi^3)) - 1)
    double dA_dfactor = A * A * exp_val / (kPBE_beta / kPBE_gamma);
    double dA_dec_lda = dA_dfactor / (kPBE_gamma * phi3);
    double dA_dphi = dA_dfactor * (-3.0 * ec_lda / (kPBE_gamma * phi3 * phi));

    // dec_lda/drho_a = (vc_lda_a - ec_lda) / rho
    // (from: d(rho*ec)/drho_a = ec + rho*dec/drho_a = vc_lda_a)
    double dec_lda_drho_a = (vc_lda_a - ec_lda) / rho;
    double dec_lda_drho_b = (vc_lda_b - ec_lda) / rho;

    // dH/dA: from the ratio expression
    double d_frac_dA = -t2 * u * (2.0 + u) / (one_u_u2 * one_u_u2);
    double dratio_dA = (kPBE_beta / kPBE_gamma) * t2 * d_frac_dA;
    double dH_dA = kPBE_gamma * phi3 / (1.0 + ratio) * dratio_dA;

    // dH/dphi: H = gamma * phi^3 * ln(1+ratio)
    // Explicit phi^3 prefactor:
    double dH_dphi_explicit =
        3.0 * kPBE_gamma * phi * phi * std::log(1.0 + ratio);
    // Implicit phi dependence through t^2 (t^2 ~ 1/phi^2):
    double dH_dphi_implicit = dH_dt2 * dt2_dphi;
    // Implicit phi dependence through A:
    double dH_dphi_A = dH_dA * dA_dphi;
    double dH_dphi_total = dH_dphi_explicit + dH_dphi_implicit + dH_dphi_A;

    // Assemble dH/drho_sigma:
    // H depends on rho_sigma through: rho (-> kF, ks, t), zeta (-> phi), ec_lda (-> A).
    // Use dt2_drho (constant-phi part) to avoid double-counting the phi path,
    // which is fully handled by dH_dphi_total * dphi_drho_sigma.
    double dH_drho_a = dH_dt2 * dt2_drho + dH_dA * (dA_dec_lda * dec_lda_drho_a)
                       + dH_dphi_total * dphi_drho_a;

    double dH_drho_b = dH_dt2 * dt2_drho + dH_dA * (dA_dec_lda * dec_lda_drho_b)
                       + dH_dphi_total * dphi_drho_b;

    // Final: v_c^sigma = d(rho*(ec_lda + H))/drho_sigma = vc_lda_sigma + H + rho * dH/drho_sigma
    vc_a = vc_lda_a + H + rho * dH_drho_a;
    vc_b = vc_lda_b + H + rho * dH_drho_b;
}

// ── Hybrid metadata ─────────────────────────────────────────────────────

double hybrid_exchange_fraction(XCType type) {
    switch (type) {
    case XCType::LSDA:
        return 0.0;
    case XCType::PBE:
        return 0.0;
    case XCType::B3LYP:
        return 0.20; // libxc XC_HYB_GGA_XC_B3LYP
    case XCType::PBE0:
        return 0.25; // libxc XC_HYB_GGA_XC_PBEH
    }
    return 0.0;
}

bool is_gga(XCType type) {
    switch (type) {
    case XCType::LSDA:
        return false;
    case XCType::PBE:
        return true;
    case XCType::B3LYP:
        return true;
    case XCType::PBE0:
        return true;
    }
    return false;
}

// ── Semi-local hybrid component evaluators ──────────────────────────────
//
// B3LYP semi-local mix (absolute coefficients, libxc id 402):
//   0.08*Slater-X + 0.72*B88-X + 0.19*VWN_RPA-C + 0.81*LYP-C
// The 0.20 exact-HF-exchange admixture is applied by the Fock builder, not
// here. The semi-local coefficients are absolute -- there is no
// (1-alpha_hf) rescaling.
//
// B88-X is the FULL Becke-88 GGA exchange (it includes its own LDA Slater
// base plus the gradient enhancement); the 0.08*Slater term is the extra
// pure-LDA exchange of the B3LYP recipe, not a double count.

namespace {

constexpr double kB3LYP_cSlater = 0.08; ///< absolute Slater-X coefficient.
constexpr double kB3LYP_cB88 = 0.72;    ///< absolute B88-X coefficient.
constexpr double kB3LYP_cVWN = 0.19;    ///< absolute VWN_RPA-C coefficient.
constexpr double kB3LYP_cLYP = 0.81;    ///< absolute LYP-C coefficient.
constexpr double kPBE0_cX = 0.75;       ///< PBE0 semi-local PBE exchange scale.

// Evaluate the B3LYP semi-local energy density / potential at one grid point.
// Outputs follow the XCOutput convention: exc is the semi-local energy
// density per electron; vrho_* and vsigma_* are df/d(rho|sigma).
void b3lyp_semilocal(double ra, double rb, double sigma_aa, double sigma_ab,
                     double sigma_bb, XCOutput& out) {
    const double rho = ra + rb;

    // 0.08 * Slater exchange (per-electron exc, per-electron potentials).
    double ex_s, vx_s_a, vx_s_b;
    slater_exchange(ra, rb, ex_s, vx_s_a, vx_s_b);

    // 0.72 * B88 exchange (per-spin, per-volume energy + potentials).
    double sa = std::max(sigma_aa, 0.0);
    double sb = std::max(sigma_bb, 0.0);
    double exb_a, vrho_b_a, vsig_b_aa;
    double exb_b, vrho_b_b, vsig_b_bb;
    b88_exchange_spin(ra, sa, exb_a, vrho_b_a, vsig_b_aa);
    b88_exchange_spin(rb, sb, exb_b, vrho_b_b, vsig_b_bb);
    const double ex_b88 = (exb_a + exb_b) / rho; // per-electron

    // 0.19 * VWN-RPA LDA correlation (per-electron exc, per-electron pot).
    double ec_vwn, vc_vwn_a, vc_vwn_b;
    vwn_rpa_correlation(ra, rb, ec_vwn, vc_vwn_a, vc_vwn_b);

    // 0.81 * LYP GGA correlation (per-electron exc, per-electron pot).
    double s_ave = 0.5 * (sa + sb);
    double sab = std::min(std::max(sigma_ab, -s_ave), +s_ave);
    double ec_lyp, vc_lyp_a, vc_lyp_b, vsig_l_aa, vsig_l_ab, vsig_l_bb;
    lyp_correlation(ra, rb, sa, sab, sb, ec_lyp, vc_lyp_a, vc_lyp_b, vsig_l_aa,
                    vsig_l_ab, vsig_l_bb);

    out.exc = kB3LYP_cSlater * ex_s + kB3LYP_cB88 * ex_b88
              + kB3LYP_cVWN * ec_vwn + kB3LYP_cLYP * ec_lyp;
    out.vrho_alpha = kB3LYP_cSlater * vx_s_a + kB3LYP_cB88 * vrho_b_a
                     + kB3LYP_cVWN * vc_vwn_a + kB3LYP_cLYP * vc_lyp_a;
    out.vrho_beta = kB3LYP_cSlater * vx_s_b + kB3LYP_cB88 * vrho_b_b
                    + kB3LYP_cVWN * vc_vwn_b + kB3LYP_cLYP * vc_lyp_b;
    out.vsigma_aa = kB3LYP_cB88 * vsig_b_aa + kB3LYP_cLYP * vsig_l_aa;
    out.vsigma_ab = kB3LYP_cLYP * vsig_l_ab;
    out.vsigma_bb = kB3LYP_cB88 * vsig_b_bb + kB3LYP_cLYP * vsig_l_bb;
}

// Evaluate the PBE0 semi-local energy density / potential at one grid point.
// PBE0 reuses the native PBE engine: the PBE exchange contribution is scaled
// by 0.75; PBE correlation stays at 1.0.
void pbe0_semilocal(double ra, double rb, double sigma_aa, double sigma_ab,
                    double sigma_bb, XCOutput& out) {
    const double rho = ra + rb;

    double sa = std::max(sigma_aa, 0.0);
    double sb = std::max(sigma_bb, 0.0);
    double s_ave = 0.5 * (sa + sb);
    double sab = std::min(std::max(sigma_ab, -s_ave), +s_ave);

    // PBE exchange (per-spin, per-volume).
    double ex_pbe_a, vrho_x_a, vsig_x_aa;
    double ex_pbe_b, vrho_x_b, vsig_x_bb;
    pbe_exchange_spin(ra, sa, ex_pbe_a, vrho_x_a, vsig_x_aa);
    pbe_exchange_spin(rb, sb, ex_pbe_b, vrho_x_b, vsig_x_bb);
    const double ex_pbe = (ex_pbe_a + ex_pbe_b) / rho; // per-electron

    // PBE correlation: PW92 LDA base + gradient correction.
    double ec_lda, vc_lda_a, vc_lda_b;
    pw92_correlation(ra, rb, ec_lda, vc_lda_a, vc_lda_b);
    double ec_pbe, vc_pbe_a, vc_pbe_b, vsig_c_aa, vsig_c_ab, vsig_c_bb;
    pbe_correlation(ra, rb, ec_lda, vc_lda_a, vc_lda_b, sa, sab, sb, ec_pbe,
                    vc_pbe_a, vc_pbe_b, vsig_c_aa, vsig_c_ab, vsig_c_bb);

    // 0.75 * PBE-X + 1.00 * PBE-C.
    out.exc = kPBE0_cX * ex_pbe + ec_pbe;
    out.vrho_alpha = kPBE0_cX * vrho_x_a + vc_pbe_a;
    out.vrho_beta = kPBE0_cX * vrho_x_b + vc_pbe_b;
    out.vsigma_aa = kPBE0_cX * vsig_x_aa + vsig_c_aa;
    out.vsigma_ab = vsig_c_ab;
    out.vsigma_bb = kPBE0_cX * vsig_x_bb + vsig_c_bb;
}

} // namespace

// ── Public API ──────────────────────────────────────────────────────────

XCOutput evaluate_xc(XCType type, double rho_a, double rho_b, double sigma_aa,
                     double sigma_ab, double sigma_bb) {
    XCOutput out{};
    double ra = static_cast<double>(rho_a);
    double rb = static_cast<double>(rho_b);
    double rho = ra + rb;

    if (rho < kRhoMin)
        return out;

    if (type == XCType::LSDA) {
        double ex, vx_a, vx_b;
        slater_exchange(ra, rb, ex, vx_a, vx_b);

        double ec, vc_a, vc_b;
        vwn5_correlation(ra, rb, ec, vc_a, vc_b);

        out.exc = static_cast<double>(ex + ec);
        out.vrho_alpha = static_cast<double>(vx_a + vc_a);
        out.vrho_beta = static_cast<double>(vx_b + vc_b);
    } else if (type == XCType::PBE) {
        // PBE = LSDA + gradient corrections. Do NOT clamp per-spin densities
        // to kRhoMin here -- inflating rho_s = 0 to 1e-12 makes
        // ds2_dsigma ~ 1/rho_s^{8/3} explode by O(1e30) and produces
        // nonsense for the empty spin channel. The exchange branch
        // (pbe_exchange_spin) has its own rho_s < kRhoMin/2 early-return
        // that yields exactly zero contribution for empty spins (matching
        // libxc). The correlation branch (pw92_correlation) clamps
        // internally to kSpinDensMin = 1e-15 for the VWN-style Pade
        // evaluator, which keeps things finite without inflating the
        // sigma derivative.
        double ra_pbe = ra;
        double rb_pbe = rb;
        double rho_pbe = ra_pbe + rb_pbe;

        // LSDA part (Slater + VWN5) at clamped densities
        double ex_lda, vx_lda_a, vx_lda_b;
        slater_exchange(ra_pbe, rb_pbe, ex_lda, vx_lda_a, vx_lda_b);

        // PBE uses PW92 (not VWN5) as the LSDA correlation base; this is
        // what libxc's XC_GGA_C_PBE does internally per the PBE paper.
        double ec_lda, vc_lda_a, vc_lda_b;
        pw92_correlation(ra_pbe, rb_pbe, ec_lda, vc_lda_a, vc_lda_b);

        // PBE exchange (per-spin)
        double sa = static_cast<double>(sigma_aa);
        double sab = static_cast<double>(sigma_ab);
        double sb = static_cast<double>(sigma_bb);

        // sigma is a squared gradient magnitude -- never negative; let it
        // through as-is. libxc applies a threshold to *skip* tiny-sigma
        // grid points (it does not bump them up to a floor); since native
        // is already gated by kRhoMin on the density side, no further
        // sigma clamping is needed. Inflating sigma from 0 -> sigma_min
        // would inject a phantom gradient at every zero-gradient grid
        // point, diverging from libxc.
        sa = std::max(sa, 0.0);
        sb = std::max(sb, 0.0);
        double s_ave = 0.5 * (sa + sb);
        sab = std::max(sab, -s_ave);
        sab = std::min(sab, +s_ave);

        double ex_pbe_a, vrho_x_a, vsig_x_aa;
        double ex_pbe_b, vrho_x_b, vsig_x_bb;
        pbe_exchange_spin(ra_pbe, sa, ex_pbe_a, vrho_x_a, vsig_x_aa);
        pbe_exchange_spin(rb_pbe, sb, ex_pbe_b, vrho_x_b, vsig_x_bb);

        // Total PBE exchange energy density (per electron)
        double ex_pbe = (ex_pbe_a + ex_pbe_b) / rho_pbe;

        // PBE correlation
        double ec_pbe, vc_pbe_a, vc_pbe_b;
        double vsig_c_aa, vsig_c_ab, vsig_c_bb;
        pbe_correlation(ra_pbe, rb_pbe, ec_lda, vc_lda_a, vc_lda_b, sa, sab, sb,
                        ec_pbe, vc_pbe_a, vc_pbe_b, vsig_c_aa, vsig_c_ab,
                        vsig_c_bb);

        out.exc = static_cast<double>(ex_pbe + ec_pbe);
        out.vrho_alpha = static_cast<double>(vrho_x_a + vc_pbe_a);
        out.vrho_beta = static_cast<double>(vrho_x_b + vc_pbe_b);
        out.vsigma_aa = static_cast<double>(vsig_x_aa + vsig_c_aa);
        out.vsigma_ab = static_cast<double>(vsig_c_ab);
        out.vsigma_bb = static_cast<double>(vsig_x_bb + vsig_c_bb);
    } else if (type == XCType::B3LYP) {
        // B3LYP semi-local part only; HF-exchange admixture (alpha=0.20)
        // is added by the Fock builder.
        b3lyp_semilocal(ra, rb, sigma_aa, sigma_ab, sigma_bb, out);
    } else if (type == XCType::PBE0) {
        // PBE0 semi-local part only; HF-exchange admixture (alpha=0.25)
        // is added by the Fock builder.
        pbe0_semilocal(ra, rb, sigma_aa, sigma_ab, sigma_bb, out);
    }

    return out;
}

void evaluate_xc_batch(XCType type, int n, const double* rho_a_arr,
                       const double* rho_b_arr, const double* sigma_aa_arr,
                       const double* sigma_ab_arr, const double* sigma_bb_arr,
                       double* exc_arr, double* vrho_a_arr, double* vrho_b_arr,
                       double* vsigma_aa_arr, double* vsigma_ab_arr,
                       double* vsigma_bb_arr) {
    for (int g = 0; g < n; ++g) {
        double ra = rho_a_arr[g];
        double rb = rho_b_arr[g];
        double rho = ra + rb;

        if (rho < kRhoMin) {
            exc_arr[g] = 0;
            vrho_a_arr[g] = 0;
            vrho_b_arr[g] = 0;
            if (vsigma_aa_arr) {
                vsigma_aa_arr[g] = 0;
                vsigma_ab_arr[g] = 0;
                vsigma_bb_arr[g] = 0;
            }
            continue;
        }

        if (type == XCType::LSDA) {
            double ex, vx_a, vx_b;
            slater_exchange(ra, rb, ex, vx_a, vx_b);
            double ec, vc_a, vc_b;
            vwn5_correlation(ra, rb, ec, vc_a, vc_b);
            exc_arr[g] = ex + ec;
            vrho_a_arr[g] = vx_a + vc_a;
            vrho_b_arr[g] = vx_b + vc_b;
        } else if (type == XCType::PBE) {
            // See scalar branch for rationale: no per-spin clamping here.
            double ra_pbe = ra;
            double rb_pbe = rb;
            double rho_pbe = ra_pbe + rb_pbe;

            double ex_lda, vx_lda_a, vx_lda_b;
            slater_exchange(ra_pbe, rb_pbe, ex_lda, vx_lda_a, vx_lda_b);
            double ec_lda, vc_lda_a, vc_lda_b;
            // PBE uses PW92 (not VWN5) as the LSDA correlation base.
            pw92_correlation(ra_pbe, rb_pbe, ec_lda, vc_lda_a, vc_lda_b);

            double sa = sigma_aa_arr[g], sab = sigma_ab_arr[g],
                   sb = sigma_bb_arr[g];
            // No sigma floor -- see scalar branch for rationale.
            sa = std::max(sa, 0.0);
            sb = std::max(sb, 0.0);
            double s_ave = 0.5 * (sa + sb);
            sab = std::max(sab, -s_ave);
            sab = std::min(sab, +s_ave);

            double ex_pbe_a, vrho_x_a, vsig_x_aa;
            double ex_pbe_b, vrho_x_b, vsig_x_bb;
            pbe_exchange_spin(ra_pbe, sa, ex_pbe_a, vrho_x_a, vsig_x_aa);
            pbe_exchange_spin(rb_pbe, sb, ex_pbe_b, vrho_x_b, vsig_x_bb);
            double ex_pbe = (ex_pbe_a + ex_pbe_b) / rho_pbe;

            double ec_pbe, vc_pbe_a, vc_pbe_b;
            double vsig_c_aa, vsig_c_ab, vsig_c_bb;
            pbe_correlation(ra_pbe, rb_pbe, ec_lda, vc_lda_a, vc_lda_b, sa, sab,
                            sb, ec_pbe, vc_pbe_a, vc_pbe_b, vsig_c_aa,
                            vsig_c_ab, vsig_c_bb);

            exc_arr[g] = ex_pbe + ec_pbe;
            vrho_a_arr[g] = vrho_x_a + vc_pbe_a;
            vrho_b_arr[g] = vrho_x_b + vc_pbe_b;
            vsigma_aa_arr[g] = vsig_x_aa + vsig_c_aa;
            vsigma_ab_arr[g] = vsig_c_ab;
            vsigma_bb_arr[g] = vsig_x_bb + vsig_c_bb;
        } else if (type == XCType::B3LYP || type == XCType::PBE0) {
            // Global hybrids: GGA semi-local part. The HF-exchange
            // admixture is applied by the Fock builder, not here.
            double sa = sigma_aa_arr ? sigma_aa_arr[g] : 0.0;
            double sab = sigma_ab_arr ? sigma_ab_arr[g] : 0.0;
            double sb = sigma_bb_arr ? sigma_bb_arr[g] : 0.0;
            XCOutput out{};
            if (type == XCType::B3LYP)
                b3lyp_semilocal(ra, rb, sa, sab, sb, out);
            else
                pbe0_semilocal(ra, rb, sa, sab, sb, out);
            exc_arr[g] = out.exc;
            vrho_a_arr[g] = out.vrho_alpha;
            vrho_b_arr[g] = out.vrho_beta;
            vsigma_aa_arr[g] = out.vsigma_aa;
            vsigma_ab_arr[g] = out.vsigma_ab;
            vsigma_bb_arr[g] = out.vsigma_bb;
        }
    }
}

} // namespace scf
} // namespace cpu
} // namespace ATV
