/// @file cpu/gpw/internal/xc_grid.h
/// @brief Exchange-correlation energy and potential on the GPW uniform grid.
///
/// Composes the collocated density (`collocate.h`) with a pointwise XC functional
/// to produce the XC energy E_xc = ∫ ρ ε_xc(ρ) dr and the real-space XC potential
/// V_xc(r). The KS matrix contribution then follows from `integrate_potential`.
/// Phase 1 uses the LDA path (PADE functional, spin-unpolarized: ρ split evenly
/// between spins); GGA gradient terms are a later task.
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief XC result on the grid.
struct XcOnGrid {
    double e_xc = 0.0;         ///< total XC energy [Hartree].
    std::vector<double> vxc_r; ///< XC potential dE_xc/dρ at each grid point.
};

/// @brief Evaluate the PADE (GTH Padé) LDA XC energy and potential on the grid.
///
/// For each grid point with total density ρ = rho_r[g], evaluates the
/// spin-unpolarized PADE functional (ρ_α = ρ_β = ρ/2): accumulates
/// E_xc += w ρ ε_xc and stores V_xc(r_g) = dE_xc/dρ. Densities below the
/// functional cutoff contribute zero.
///
/// @param grid  Uniform grid (supplies the integration weight).
/// @param rho_r Total electron density on the grid, length grid.npts().
/// @return XcOnGrid with the total energy and the real-space potential.
XcOnGrid xc_grid_pade(const UniformGrid& grid,
                      const std::vector<double>& rho_r);

} // namespace gpw
} // namespace cpu
} // namespace ATV
