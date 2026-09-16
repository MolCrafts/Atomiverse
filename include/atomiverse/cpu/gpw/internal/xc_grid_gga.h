/// @file cpu/gpw/internal/xc_grid_gga.h
/// @brief GGA (PBE / PBEsol / revPBE / RPBE / BLYP / OptB88VdW) exchange-correlation energy
///        and potential on the GPW grid.
///
/// The GGA counterpart of `xc_grid_pade` (LDA). Density gradients are computed
/// spectrally (`grid_fft`), the pointwise functional is reused from the
/// project's libxc-parity `cpu::scf::evaluate_xc`, and the gradient-dependent
/// part of the potential is assembled via the divergence term. Closed-shell
/// (spin-unpolarized): rho split evenly between spins. Spin-resolved
/// (`xc_grid_gga_spin`) is the UKS path (paired with `xc_grid_pade_lsd`).
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h" // UniformGrid
#include "atomiverse/cpu/gpw/internal/xc_grid.h"   // XcOnGrid, XcOnGridSpin
#include "atomiverse/cpu/scf/xc_functional.h"      // XCType

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Evaluate a pure GGA XC energy and potential on the grid.
///
/// E_xc = sum_g w rho eps_xc; the real-space potential is
///   v_xc = vrho_a - div[ (vsigma_aa + 1/2 vsigma_ab) grad rho ]
/// (the spin-unpolarized collapse of v_a = vrho_a - div[2 vsigma_aa grad rho_a
///  + vsigma_ab grad rho_b] with rho_a=rho_b=rho/2, grad rho_a=grad rho/2).
///
/// @param grid  Uniform grid.
/// @param rho_r Total electron density on the grid, length grid.npts().
/// @param xc    Pure GGA type (`PBE`, `PBEsol`, `RevPBE`, `RPBE`, `BLYP`,
///              or `OptB88VdW`).
/// @return XcOnGrid with the total energy and the real-space potential.
XcOnGrid xc_grid_gga(const UniformGrid& grid, const std::vector<double>& rho_r,
                     cpu::scf::XCType xc);

/// @brief PBE GGA on the grid (`xc_grid_gga` with `XCType::PBE`).
XcOnGrid xc_grid_pbe(const UniformGrid& grid, const std::vector<double>& rho_r);

/// @brief PBEsol GGA on the grid (`xc_grid_gga` with `XCType::PBEsol`).
XcOnGrid xc_grid_pbesol(const UniformGrid& grid,
                        const std::vector<double>& rho_r);

/// @brief revPBE GGA on the grid (`xc_grid_gga` with `XCType::RevPBE`).
XcOnGrid xc_grid_revpbe(const UniformGrid& grid,
                        const std::vector<double>& rho_r);

/// @brief RPBE GGA on the grid (`xc_grid_gga` with `XCType::RPBE`).
XcOnGrid xc_grid_rpbe(const UniformGrid& grid, const std::vector<double>& rho_r);

/// @brief BLYP GGA on the grid (`xc_grid_gga` with `XCType::BLYP`).
XcOnGrid xc_grid_blyp(const UniformGrid& grid, const std::vector<double>& rho_r);

/// @brief optB88-vdW semi-local on the grid (`XCType::OptB88VdW`).
XcOnGrid xc_grid_optb88_vdw(const UniformGrid& grid,
                            const std::vector<double>& rho_r);

/// @brief Spin-resolved pure GGA on the grid (UKS).
///
/// E_xc = sum_g w (ρ_α+ρ_β) ε_xc; potentials
///   v_α = vrho_α − div[ 2 vσ_αα ∇ρ_α + vσ_αβ ∇ρ_β ]
///   v_β = vrho_β − div[ 2 vσ_ββ ∇ρ_β + vσ_αβ ∇ρ_α ]
/// with σ_αα=|∇ρ_α|², σ_ββ=|∇ρ_β|², σ_αβ=∇ρ_α·∇ρ_β from spectral gradients.
///
/// @param grid  Uniform grid.
/// @param rho_a Alpha density on the grid.
/// @param rho_b Beta density on the grid.
/// @param xc    Pure GGA type (`PBE`, `PBEsol`, `RevPBE`, `RPBE`, `BLYP`,
///              or `OptB88VdW`).
/// @return Spin-resolved XC energy and potentials.
XcOnGridSpin xc_grid_gga_spin(const UniformGrid& grid,
                              const std::vector<double>& rho_a,
                              const std::vector<double>& rho_b,
                              cpu::scf::XCType xc);

/// @brief Spin-resolved PBE (`xc_grid_gga_spin` with `XCType::PBE`).
XcOnGridSpin xc_grid_pbe_spin(const UniformGrid& grid,
                              const std::vector<double>& rho_a,
                              const std::vector<double>& rho_b);

/// @brief Spin-resolved PBEsol (`xc_grid_gga_spin` with `XCType::PBEsol`).
XcOnGridSpin xc_grid_pbesol_spin(const UniformGrid& grid,
                                 const std::vector<double>& rho_a,
                                 const std::vector<double>& rho_b);

/// @brief Spin-resolved revPBE (`xc_grid_gga_spin` with `XCType::RevPBE`).
XcOnGridSpin xc_grid_revpbe_spin(const UniformGrid& grid,
                                 const std::vector<double>& rho_a,
                                 const std::vector<double>& rho_b);

/// @brief Spin-resolved RPBE (`xc_grid_gga_spin` with `XCType::RPBE`).
XcOnGridSpin xc_grid_rpbe_spin(const UniformGrid& grid,
                               const std::vector<double>& rho_a,
                               const std::vector<double>& rho_b);

/// @brief Spin-resolved BLYP (`xc_grid_gga_spin` with `XCType::BLYP`).
XcOnGridSpin xc_grid_blyp_spin(const UniformGrid& grid,
                               const std::vector<double>& rho_a,
                               const std::vector<double>& rho_b);

/// @brief Spin-resolved optB88-vdW semi-local (`XCType::OptB88VdW`).
XcOnGridSpin xc_grid_optb88_vdw_spin(const UniformGrid& grid,
                                     const std::vector<double>& rho_a,
                                     const std::vector<double>& rho_b);

} // namespace gpw
} // namespace cpu
} // namespace ATV
