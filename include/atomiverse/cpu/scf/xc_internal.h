/// @file cpu/scf/xc_internal.h
/// @brief Private declarations shared across the native XC translation units.
///
/// This header is internal to the native exchange-correlation engine
/// (`xc_functional.cpp` + `xc_gga.cpp`). It is NOT facade-public: only the
/// XC engine sources and the libxc-parity tests include it. The public XC
/// surface (`evaluate_xc` / `evaluate_xc_batch`) lives in `xc_functional.h`.
///
/// It declares:
///   - the three semi-local building-block functionals added for the
///     B3LYP hybrid (`vwn_rpa_correlation`, `b88_exchange_spin`,
///     `lyp_correlation`); and
///   - the shared VWN Pade machinery (`VWN5Params`, `vwn5_F`,
///     `vwn5_dFdx`) reused by both `vwn5_correlation` and
///     `vwn_rpa_correlation`.
///
/// All formulas use atomic units (Hartree energy, Bohr length). Each
/// functional is spin-polarized and closed-shell-correct as the
/// `rho_alpha = rho_beta` limit of its spin-polarized expression.
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Parameter set for one channel of a VWN Pade-approximant
///        correlation functional (paramagnetic / ferromagnetic /
///        spin-stiffness).
///
/// Both VWN5 (`XC_LDA_C_VWN`, id 7) and VWN-RPA (`XC_LDA_C_VWN_RPA`,
/// id 8) share the same Pade functional form and differ only in these
/// four constants per channel.
struct VWN5Params {
    double A;  ///< Amplitude A, Hartree convention (A_Rydberg / 2).
    double x0; ///< Pade pole location x0 [sqrt(Bohr)].
    double b;  ///< Pade denominator linear coefficient.
    double c;  ///< Pade denominator constant term.
};

/// @brief VWN Pade-approximant correlation function F(x).
///
/// F(x) = A { ln(x^2 / X) + 2b/Q * arctan(Q/(2x+b))
///          - b*x0/X(x0) * [ ln((x-x0)^2 / X)
///                           + 2(b+2*x0)/Q * arctan(Q/(2x+b)) ] }
/// with X(x) = x^2 + b*x + c and Q = sqrt(4c - b^2).
///
/// @param x Wigner-Seitz square root, x = sqrt(r_s) [sqrt(Bohr)].
/// @param p VWN parameter set for the desired channel.
/// @return Channel correlation energy density per electron [Hartree].
double vwn5_F(double x, const VWN5Params& p);

/// @brief Derivative dF/dx of the VWN Pade-approximant function.
/// @param x Wigner-Seitz square root, x = sqrt(r_s) [sqrt(Bohr)].
/// @param p VWN parameter set for the desired channel.
/// @return dF/dx [Hartree / sqrt(Bohr)].
double vwn5_dFdx(double x, const VWN5Params& p);

/// @brief RPA-parametrized Vosko-Wilk-Nusair LDA correlation
///        (libxc XC_LDA_C_VWN_RPA, id 8), spin-polarized.
///
/// Same Pade functional form as `vwn5_correlation` with the RPA
/// parameter columns of the VWN paper (Can. J. Phys. 58, 1200, 1980).
/// Closed-shell-correct as the `rho_a == rho_b` (zeta = 0) limit, where
/// it reduces to the paramagnetic Pade term.
///
/// @param rho_a Alpha spin density [electrons/Bohr^3].
/// @param rho_b Beta spin density [electrons/Bohr^3].
/// @param[out] ec   Correlation energy density per electron [Hartree].
/// @param[out] vc_a d(rho*ec)/drho_alpha, alpha potential [Hartree].
/// @param[out] vc_b d(rho*ec)/drho_beta, beta potential [Hartree].
void vwn_rpa_correlation(double rho_a, double rho_b, double& ec, double& vc_a,
                         double& vc_b);

/// @brief Becke-1988 GGA exchange for a single spin channel
///        (libxc XC_GGA_X_B88, id 106).
///
/// E_x = sum_sigma integral rho_sigma * eps_x^LDA(rho_sigma) * F(x_sigma),
/// with reduced gradient x_sigma = |grad rho_sigma| / rho_sigma^(4/3) and
/// Becke enhancement F(x) = 1 + (beta/C_x) x^2 / (1 + 6 beta x asinh(x)),
/// beta = 0.0042. An empty spin channel
/// (`rho_s` below the per-spin density floor) returns exactly zero on
/// all three channels, matching libxc.
///
/// @param rho_s    Spin density for this channel [electrons/Bohr^3].
/// @param sigma_ss |grad rho_sigma|^2 for this channel [e^2/Bohr^8].
/// @param[out] ex_s      Exchange energy density per volume for this
///                       spin channel, rho_s*eps_x [Hartree/Bohr^3].
/// @param[out] vrho_s    d(rho_s*eps_x)/drho_s, exchange potential
///                       [Hartree].
/// @param[out] vsigma_ss d(rho_s*eps_x)/dsigma_ss [Hartree*Bohr^2].
void b88_exchange_spin(double rho_s, double sigma_ss, double& ex_s,
                       double& vrho_s, double& vsigma_ss);

/// @brief Lee-Yang-Parr GGA correlation (libxc XC_GGA_C_LYP, id 131),
///        spin-polarized, in the Miehlich-Savin-Stoll-Preuss closed form.
///
/// Empirical constants a = 0.04918, b = 0.132, c = 0.2533, d = 0.349
/// (Lee-Yang-Parr, Phys. Rev. B 37, 785 (1988); Miehlich et al.,
/// Chem. Phys. Lett. 157, 200 (1989)). Closed-shell-correct as the
/// `rho_a == rho_b`, `sigma_aa == sigma_ab == sigma_bb` limit.
///
/// @param rho_a    Alpha spin density [electrons/Bohr^3].
/// @param rho_b    Beta spin density [electrons/Bohr^3].
/// @param sigma_aa |grad rho_alpha|^2 [e^2/Bohr^8].
/// @param sigma_ab grad rho_alpha . grad rho_beta [e^2/Bohr^8].
/// @param sigma_bb |grad rho_beta|^2 [e^2/Bohr^8].
/// @param[out] ec        Correlation energy density per electron
///                       [Hartree].
/// @param[out] vc_a      d(rho*ec)/drho_alpha [Hartree].
/// @param[out] vc_b      d(rho*ec)/drho_beta [Hartree].
/// @param[out] vsigma_aa d(rho*ec)/dsigma_aa [Hartree*Bohr^2].
/// @param[out] vsigma_ab d(rho*ec)/dsigma_ab [Hartree*Bohr^2].
/// @param[out] vsigma_bb d(rho*ec)/dsigma_bb [Hartree*Bohr^2].
void lyp_correlation(double rho_a, double rho_b, double sigma_aa,
                     double sigma_ab, double sigma_bb, double& ec, double& vc_a,
                     double& vc_b, double& vsigma_aa, double& vsigma_ab,
                     double& vsigma_bb);

} // namespace scf
} // namespace cpu
} // namespace ATV
