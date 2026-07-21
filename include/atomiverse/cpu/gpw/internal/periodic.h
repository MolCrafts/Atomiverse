/// @file cpu/gpw/internal/periodic.h
/// @brief Orthorhombic Γ-point lattice-image enumeration shared by the analytic
///        one-electron / GTH integrals and the grid collocation.
///
/// At the Γ point each contracted basis function is a periodic image sum
/// χ_μ(r) = Σ_T φ_μ(r − T) over lattice vectors T = (t_x a, t_y b, t_z c). The
/// analytic overlap/kinetic/pseudopotential matrices and the real-space density
/// collocation must use the *same* set of images for the invariant
/// ∫ρ dr = Tr(P S) to hold. This header centralizes the per-axis image count so
/// both sides derive it from one shared cutoff radius (set by EPS_DEFAULT).
#pragma once

#include <cmath>

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Per-axis half-count of lattice images covering a radius `rcut`.
///
/// For an orthorhombic cell an image t_d a_d can contribute to an integral whose
/// integrand reaches `rcut` only if |t_d| a_d <= rcut, i.e. |t_d| <= rcut / a_d.
/// @param L Box lengths a,b,c [Bohr].
/// @param periodic When false, all counts are 0 (isolated / Martyna-Tuckerman).
/// @param rcut Reach of the integrand [Bohr] (e.g. twice the basis radius for a
///        Gaussian *pair*).
/// @param m Output: half-count per axis; images run t_d ∈ [-m[d], m[d]].
inline void image_counts(const double L[3], bool periodic, double rcut,
                         int m[3]) {
    for (int d = 0; d < 3; ++d)
        m[d] = (periodic && L[d] > 0.0)
                   ? static_cast<int>(std::ceil(rcut / L[d] - 1e-12))
                   : 0;
}

/// @brief Image counts for a PAIR lattice sum over centers anywhere in the cell.
///
/// `image_counts` covers |T| <= rcut, which is right only when both centers sit at
/// the same point. A pair sum screens on |R_A - R_B - T| <= reach, so it needs
/// |T| <= |R_A - R_B| + reach — and two centers can be a full cell edge apart. One
/// extra shell per axis covers that: m[d] = ceil(rcut/L[d]) + 1.
///
/// Missing that shell truncates the Γ-point sum, and a truncated sum is not a Gram
/// matrix: a 13726-AO slab overlap came out with s_min = -1.92 (unit-diagonal;
/// PySCF's reference is +1.1e-13). It is invisible for a single atom at the origin,
/// where R_A = R_B and the extra shell is empty — which is exactly why it survived
/// the small-system tests.
///
/// @param L        Orthorhombic cell edges [Bohr].
/// @param periodic false ⇒ all counts 0 (isolated).
/// @param rcut     Largest pair reach [Bohr] (e.g. 2·rmax).
/// @param m        Out: images to sum per axis.
inline void pair_image_counts(const double L[3], bool periodic, double rcut,
                              int m[3]) {
    image_counts(L, periodic, rcut, m);
    if (periodic)
        for (int d = 0; d < 3; ++d)
            if (L[d] > 0.0)
                m[d] += 1;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
