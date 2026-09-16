/// @file cpu/gpw/internal/xc_grid.h
/// @brief Exchange-correlation energy and potential on the GPW uniform grid.
///
/// Composes the collocated density (`collocate.h`) with a pointwise XC functional
/// to produce the XC energy E_xc = ∫ ρ ε_xc(ρ) dr and the real-space XC potential
/// V_xc(r). The KS matrix contribution then follows from `integrate_potential`.
/// LDA path: spin-unpolarized PADE (`xc_grid_pade`) and spin-resolved PADE LSD
/// for UKS (`xc_grid_pade_lsd`). GGA spin path: `xc_grid_gga_spin` in
/// `xc_grid_gga.h`.
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief XC result on the grid (spin-unpolarized collapse: one V_xc).
struct XcOnGrid {
    double e_xc = 0.0;         ///< total XC energy [Hartree].
    std::vector<double> vxc_r; ///< XC potential dE_xc/dρ at each grid point.
};

/// @brief Spin-resolved XC result (UKS / LSD).
struct XcOnGridSpin {
    double e_xc = 0.0;         ///< total XC energy [Hartree].
    std::vector<double> vxc_a; ///< dE_xc/dρ_α on the grid.
    std::vector<double> vxc_b; ///< dE_xc/dρ_β on the grid.
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

/// @brief Spin-resolved PADE LSD on the grid (UKS).
///
/// E_xc = Σ_g w (ρ_α+ρ_β) ε_xc(ρ_α,ρ_β); potentials are vrho_a / vrho_b from
/// `cpu::xc::pade_lsd`.
///
/// @param grid  Uniform grid.
/// @param rho_a Alpha density on the grid.
/// @param rho_b Beta density on the grid.
/// @return Spin-resolved XC energy and potentials.
XcOnGridSpin xc_grid_pade_lsd(const UniformGrid& grid,
                              const std::vector<double>& rho_a,
                              const std::vector<double>& rho_b);

} // namespace gpw
} // namespace cpu
} // namespace ATV
