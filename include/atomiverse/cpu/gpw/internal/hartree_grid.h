/// @file cpu/gpw/internal/hartree_grid.h
/// @brief Hartree (Coulomb) energy and KS matrix block on the GPW uniform grid.
///
/// The Coulomb counterpart of `xc_grid.h`: collocate the density from P, solve
/// the periodic FFT Poisson equation for V_H(r), and integrate it back into the
/// Coulomb KS block J_uv = ∫ φ_u V_H φ_v. Self-consistent by construction:
/// E_H = 1/2 ∫ ρ V_H = 1/2 Tr(P J). Periodic (neutralizing-background) Poisson;
/// the isolated/Martyna-Tuckerman variant and the GTH long-range core charge are
/// layered on in the SCF assembly task.
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Hartree result on the grid.
struct HartreeOnGrid {
    double e_hartree = 0.0;       ///< Hartree energy 1/2 ∫ρV_H [Hartree].
    std::vector<double> j_matrix; ///< Coulomb KS block J_uv, row-major nb x nb.
};

/// @brief Assemble the Hartree energy and Coulomb KS block from a density matrix.
/// @param grid  Uniform grid.
/// @param basis Basis functions.
/// @param P     Density matrix, row-major nbasis x nbasis.
/// @return E_H and J_uv (= ∫ φ_u V_H φ_v).
HartreeOnGrid hartree_grid(const UniformGrid& grid,
                           const std::vector<GridBasisFunction>& basis,
                           const std::vector<double>& P);

} // namespace gpw
} // namespace cpu
} // namespace ATV
