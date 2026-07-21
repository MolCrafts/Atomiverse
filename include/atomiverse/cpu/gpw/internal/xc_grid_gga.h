/// @file cpu/gpw/internal/xc_grid_gga.h
/// @brief GGA (PBE) exchange-correlation energy and potential on the GPW grid.
///
/// The GGA counterpart of `xc_grid_pade` (LDA). Density gradients are computed
/// spectrally (`grid_fft`), the pointwise PBE functional is reused from the
/// project's libxc-parity `cpu::scf::evaluate_xc`, and the gradient-dependent
/// part of the potential is assembled via the divergence term. Closed-shell
/// (spin-unpolarized): rho split evenly between spins.
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h" // UniformGrid
#include "atomiverse/cpu/gpw/internal/xc_grid.h"   // XcOnGrid

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Evaluate the PBE GGA XC energy and potential on the grid.
///
/// E_xc = sum_g w rho eps_xc; the real-space potential is
///   v_xc = vrho_a - div[ (vsigma_aa + 1/2 vsigma_ab) grad rho ]
/// (the spin-unpolarized collapse of v_a = vrho_a - div[2 vsigma_aa grad rho_a
///  + vsigma_ab grad rho_b] with rho_a=rho_b=rho/2, grad rho_a=grad rho/2).
///
/// @param grid  Uniform grid.
/// @param rho_r Total electron density on the grid, length grid.npts().
/// @return XcOnGrid with the total energy and the real-space potential.
XcOnGrid xc_grid_pbe(const UniformGrid& grid, const std::vector<double>& rho_r);

} // namespace gpw
} // namespace cpu
} // namespace ATV
