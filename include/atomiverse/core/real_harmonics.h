/// @file core/real_harmonics.h
/// @brief Arbitrary-l real solid / spherical harmonics as Cartesian polynomials.
///
/// Shared angular-momentum utility (primary consumer: GPW; the Gaussian-SCF
/// cart2sph tables are convention-pinned separately). Two representations,
/// both as Cartesian-power ->
/// coefficient lists so the downstream Gaussian machinery (Hermite-recurrence
/// overlap/kinetic, grid collocation, GTH moment integrals — all already general
/// in the Cartesian powers) can consume them for any l:
///
///   - Use A (basis): the UNNORMALIZED real solid harmonics X_l^m. Only the
///     spanned (2l+1)-dimensional pure-spherical space matters; the per-function
///     scale cancels because the contracted-GTO eigenproblem H C = S C eps is
///     invariant under per-basis-function rescaling. Reproduces, up to scale,
///     the canonical sets s:1; p:{x,y,z}; d:{xy,yz,xz,x^2-y^2,2z^2-x^2-y^2}; etc.
///
///   - Use B (GTH nonlocal projectors): the L2(S^2)-NORMALIZED real spherical
///     harmonics, returned as the coefficients of r^l * Y_lm (a degree-l Cartesian
///     polynomial), so int |Y_lm|^2 dOmega = 1. The projector sums
///     sum_m |p_lm><p_lm| over the whole shell, so by the spherical-harmonic
///     addition theorem the result is independent of m-ordering / Condon-Shortley
///     phase — any consistent orthonormal real set is correct.
///
/// Generated from the closed form of Ribaldone & Desmarais, J. Chem. Phys. 163,
/// 074102 (2025), arXiv:2412.16733, Eq. (7) (a corrected restatement of Schlegel
/// & Frisch, Int. J. Quantum Chem. 54, 83 (1995)); normalization via the exact
/// sphere monomial integral. Verified against the explicit l <= 3 forms.
#pragma once

#include <vector>

namespace ATV {

/// @brief One Cartesian term c * x^lx y^ly z^lz of a real harmonic.
struct HarmTerm {
    int lx;       ///< power of x.
    int ly;       ///< power of y.
    int lz;       ///< power of z.
    double coeff; ///< coefficient.
};

/// @brief One real-harmonic m-component: a list of Cartesian terms (lx+ly+lz=l).
using RealHarmonic = std::vector<HarmTerm>;

/// @brief Exact surface integral int_{S^2} xhat^i yhat^j zhat^k dOmega.
///
/// Zero if any exponent is odd; otherwise
/// 4pi (i-1)!!(j-1)!!(k-1)!! / (i+j+k+1)!! with (-1)!! = 0!! = 1.
///
/// @param i Power of x-hat (must be >= 0).
/// @param j Power of y-hat (must be >= 0).
/// @param k Power of z-hat (must be >= 0).
/// @return The surface integral [dimensionless, units of solid angle].
double sphere_monomial_integral(int i, int j, int k);

/// @brief Unnormalized real solid harmonics X_l^m for a shell (Use A).
///
/// @param l Angular momentum (l >= 0).
/// @return 2l+1 components, each a Cartesian-term list of total degree l. The
///         per-component overall scale is arbitrary (cancels in the basis norm).
std::vector<RealHarmonic> real_solid_harmonics(int l);

/// @brief L2(S^2)-normalized real spherical harmonics, as r^l Y_lm (Use B).
///
/// Each returned component is the degree-l Cartesian polynomial r^l Y_lm whose
/// angular part integrates to one (int |Y_lm|^2 dOmega = 1). The m-ordering is
/// unspecified (irrelevant for the shell-summed GTH projector).
///
/// @param l Angular momentum (l >= 0).
/// @return 2l+1 normalized components.
std::vector<RealHarmonic> real_spherical_harmonics_normalized(int l);

} // namespace ATV
