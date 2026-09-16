/// @file cpu/gpw/internal/one_electron.h
/// @brief Analytic overlap (S) and kinetic-energy (T) integrals over primitive
///        Cartesian Gaussians, for the GPW KS matrix.
///
/// These are the "hard" (analytic) one-electron parts of the GPW Hamiltonian
/// H_KS = T + V_PP + V_soft[ρ]: the kinetic energy and the overlap metric of the
/// generalized eigenproblem H C = S C ε. Implemented with the McMurchie-Davidson
/// Hermite-expansion scheme (arbitrary angular momentum), operating on the shared
/// `GridPrimitive` type so the GPW basis (`GridBasisFunction`) is used directly.
#pragma once

#include "atomiverse/cpu/gpw/internal/collocate.h" // GridPrimitive, GridBasisFunction

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Overlap integral ⟨a|b⟩ of two primitive Cartesian Gaussians.
double prim_overlap(const GridPrimitive& a, const GridPrimitive& b);

/// @brief Kinetic-energy integral ⟨a|-½∇²|b⟩ of two primitive Gaussians.
double prim_kinetic(const GridPrimitive& a, const GridPrimitive& b);

/// @brief Fused ⟨a|b⟩ and ⟨a|-½∇²|b⟩ sharing the Hermite tables (hot STVnl path).
void prim_ST(const GridPrimitive& a, const GridPrimitive& b, double& s_out,
             double& t_out);

/// @brief Contracted ⟨a|b(·−T)⟩ and kinetic over all prim pairs, with **local**
///        Hermite tables reused across monoms that share (α, center). Hot path
///        for multi-k STVnl setup (one call per lattice image of an AO pair).
/// @param T Lattice translation applied to every ket primitive center [Bohr].
void contracted_ST_shift(const GridBasisFunction& a,
                         const GridBasisFunction& b, const double T[3],
                         double& s_out, double& t_out);

/// @brief Overlap of two contracted basis functions (sum over primitive pairs).
double overlap(const GridBasisFunction& a, const GridBasisFunction& b);

/// @brief Kinetic energy of two contracted basis functions.
double kinetic(const GridBasisFunction& a, const GridBasisFunction& b);

/// @brief Γ-point periodic overlap S_ab = Σ_T ⟨a | b(·−T)⟩ over the lattice
///        images T = (t_x L_x, t_y L_y, t_z L_z), t_d ∈ [−m[d], m[d]].
///
/// `m = {0,0,0}` reproduces the isolated `overlap(a,b)`. The image set must match
/// the one used by the density collocation (`build_compact_basis_grid`) so that
/// ∫ρ = Tr(P S) holds. @param L Box lengths [Bohr]. @param m Per-axis image
/// half-count from `image_counts`.
/// @param reach Sum of the two functions' radii [Bohr]; an image whose shifted
///        center-center distance exceeds it is skipped (no primitive can
///        contribute), so close pairs cost only the T=0 term.
double overlap(const GridBasisFunction& a, const GridBasisFunction& b,
               const double L[3], const int m[3], double reach);

/// @brief Γ-point periodic kinetic energy T_ab = Σ_T ⟨a | −½∇² | b(·−T)⟩.
double kinetic(const GridBasisFunction& a, const GridBasisFunction& b,
               const double L[3], const int m[3], double reach);

/// @brief Fused Γ-point periodic S and T over the same live image set.
///        Prefer this when both operators are needed (avoids a second image
///        sweep). Pure s–s and mixed-angular (L≤d, shell-centered) pairs use
///        portable xsimd on the image axis when the live image count is
///        slab-class; otherwise scalar Hermite via contracted_ST_shift.
void contracted_ST_periodic(const GridBasisFunction& a,
                            const GridBasisFunction& b, const double L[3],
                            const int m[3], double reach, double& s_out,
                            double& t_out);

} // namespace gpw
} // namespace cpu
} // namespace ATV
