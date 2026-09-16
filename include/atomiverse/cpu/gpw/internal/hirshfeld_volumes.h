/// @file cpu/gpw/internal/hirshfeld_volumes.h
/// @brief Hirshfeld volume ratios V/V_free for density-scaled TS/MBD.
///
/// Computes per-atom effective volumes from a molecular density on a uniform
/// orthorhombic grid and free-atom exponential densities
/// (`basis_data/hirshfeld_tables.h`). Ratios feed `ts_pbe` / `mbd_rsscs_pbe`
/// via the existing `vol_ratio` argument.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/basis_data/hirshfeld_tables.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Hirshfeld volume ratios \(v_A = V_A / V_A^{\mathrm{free}}\).
///
/// \[
///   w_A(\mathbf r)=\rho_A^{\mathrm{free}}/{\textstyle\sum_B\rho_B^{\mathrm{free}}},
///   \quad
///   V_A=\int w_A\,\rho\,|\mathbf r-\mathbf R_A|^3\,d^3r
/// \]
///
/// Free-atom densities and \(V^{\mathrm{free}}\) use the in-tree exponential
/// model. Minimum-image distances when `periodic` is true.
///
/// @param grid      Uniform orthorhombic grid (same as GPW density grid).
/// @param rho       Molecular valence density [e/Bohr³], length grid.npts().
/// @param Z         Atomic numbers, length N.
/// @param pos_bohr  Positions [Bohr], length N.
/// @param cell_bohr Orthorhombic a,b,c [Bohr] (for min-image when periodic).
/// @param periodic  Use orthorhombic minimum image for |r−R|.
/// @return Per-atom volume ratios (positive finite).
std::vector<double>
hirshfeld_volume_ratios(const UniformGrid& grid, const std::vector<double>& rho,
                        const std::vector<int>& Z,
                        const std::vector<std::array<double, 3>>& pos_bohr,
                        const double cell_bohr[3], bool periodic);

} // namespace gpw
} // namespace cpu
} // namespace ATV
