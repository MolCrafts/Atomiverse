/// @file cpu/gpw/internal/gth_integrals.h
/// @brief Analytic GTH pseudopotential matrix elements over primitive Cartesian
///        Gaussians (GPW spec D3).
///
/// The GPW Kohn-Sham matrix keeps the analytic ("hard") pseudopotential terms in
/// the Gaussian basis. This header provides those matrix elements between
/// *primitive* Cartesian Gaussians; contraction over a basis shell is applied by
/// the caller (a contracted GTO is a fixed linear combination of primitives, so
/// the contracted element is the coefficient-weighted sum of these primitives).
///
/// A primitive Cartesian Gaussian is
///   g_A(r) = (x-Ax)^lx (y-Ay)^ly (z-Az)^lz exp(-alpha |r-A|^2).
///
/// Two terms are implemented:
///   (1) Local short-range part  V_sr(r) = exp(-s^2/2) * sum_i C_i s^{2(i-1)},
///       with s = |r-D|/rloc and D the pseudopotential (atom) center.
///   (2) Nonlocal separable projectors: sum_l sum_{ij} h^l_ij <g_a|p_i^l><p_j^l|g_b>,
///       with the Goedecker-Teter-Hutter normalized projectors
///         p_i^l(r) = sqrt(2) r^{l+2(i-1)} exp(-r^2/(2 r_l^2))
///                    / ( r_l^{l+(4i-1)/2} sqrt(Gamma(l+(4i-1)/2)) )
///       times a real spherical harmonic Y_lm.
///
/// All integrals are evaluated by the separable Gaussian-product / polynomial-
/// moment method (Gaussian product theorem + 1D moment integrals), exact for any
/// angular momentum. The implementation is validated by closed forms and the
/// Gaussian-product symmetry in the unit test.
///
/// References:
///   - Goedecker, Teter, Hutter, Phys. Rev. B 54, 1703 (1996) — GTH form & norms.
///   - Obara, Saika, J. Chem. Phys. 84, 3963 (1986) — Gaussian product recursions.
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/gth_potential.h" // GthNonlocalChannel

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief A primitive Cartesian Gaussian.
struct Prim {
    double alpha = 0.0;           ///< exponent.
    double center[3] = {0, 0, 0}; ///< center A [Bohr].
    int l[3] = {0, 0, 0};         ///< Cartesian powers (lx, ly, lz).
};

/// @brief Matrix element <g_a | V_sr | g_b> of the GTH local short-range potential.
///
/// V_sr(r) = exp(-|r-D|^2/(2 rloc^2)) * sum_{i=1..C.size()} C[i-1] (|r-D|^2/rloc^2)^{i-1}.
///
/// @param a        Primitive Gaussian (bra).
/// @param b        Primitive Gaussian (ket).
/// @param pp_center Pseudopotential center D [Bohr] (the atom position).
/// @param rloc     Local Gaussian radius [Bohr].
/// @param c        Local coefficients C_1..C_nexp.
/// @return The integral over R^3 [Hartree-consistent].
double gth_local_sr(const Prim& a, const Prim& b, const double pp_center[3],
                    double rloc, const std::vector<double>& c);

/// @brief Overlap <g | P_{i l m}> of a primitive Gaussian with one GTH projector.
///
/// P_{ilm}(r) = p_i^l(|r-D|) Y_lm(widehat{r-D}) with the GTH-normalized radial part.
/// General l: the angular factor is the L2(S^2)-normalized real spherical harmonic
/// from real_spherical_harmonics_normalized(l), summed over its Cartesian terms.
///
/// @param g        Primitive Gaussian.
/// @param pp_center Projector/atom center D [Bohr].
/// @param l        Projector angular momentum (l >= 0).
/// @param iproj    1-based projector index within the channel (1, 2, ...).
/// @param r_l      Projector radius [Bohr].
/// @param m        0-based real-harmonic component, m in [0, 2l].
/// @return <g | P_{ilm}>.
double gth_projector_overlap(const Prim& g, const double pp_center[3], int l,
                             int iproj, double r_l, int m);

/// @brief Nonlocal pseudopotential matrix element for one l-channel:
///        sum_m sum_{ij} h_ij <g_a|p_i^l Y_lm><p_j^l Y_lm|g_b>.
///
/// @param a        Primitive Gaussian (bra).
/// @param b        Primitive Gaussian (ket).
/// @param pp_center Projector/atom center D [Bohr].
/// @param ch       One nonlocal channel (l, r_l, nproj, h matrix).
/// @return The channel's contribution to <g_a | V_nl | g_b>.
double gth_nonlocal(const Prim& a, const Prim& b, const double pp_center[3],
                    const GthNonlocalChannel& ch);

} // namespace gpw
} // namespace cpu
} // namespace ATV
