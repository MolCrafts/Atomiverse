/// @file cpu/gpw/internal/vdw_df.h
/// @brief Nonlocal vdW-DF / vdW-DF2 correlation (DRSLL / LMKLL) for GPW grids.
///
/// Román–Pérez–Soler FFT evaluation of the Dion et al. nonlocal correlation
///
///   E_c^{nl} = ½ ∬ ρ(r) φ(r,r') ρ(r') dr dr'
///
/// with q₀(ρ,|∇ρ|²) and the community RPS kernel tables. Functional variants
/// differ only in the gradient-correction prefactor Z_ab (libvdwxc / QE):
///
/// - DF1 / DRSLL: Z_ab = −0.8491  (CP2K `NON_LOCAL TYPE DRSLL`)
/// - DF2 / LMKLL: Z_ab = −1.887   (CP2K `NON_LOCAL TYPE LMKLL`)
///
/// Same kernel tables for both (libvdwxc `vdwxc_default_kernel`). Matches
/// libvdwxc `VDWXC_DF1` / `VDWXC_DF2` energy on the same density/gradient.
///
/// Product path is **self-consistent**: each SCF iteration adds
/// V_nl = δE_c^{nl}/δρ (Soler eq. 10 + spectral divergence of the gradient
/// flux) to the KS effective potential; forces reuse that V_nl in veff.
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Result of a vdW-DF / vdW-DF2 nonlocal evaluation.
struct VdwDfResult {
    double energy = 0.0; ///< E_c^{nl} [Hartree].
    /// Local KS potential V_nl [Hartree] on the grid (empty if energy-only).
    std::vector<double> v_r;
};

/// @brief DF1 / DRSLL Z_ab gradient-correction prefactor (Dion et al.).
constexpr double kVdwDf1Zab = -0.8491;
/// @brief DF2 / LMKLL Z_ab (Lee–Murray–Kong–Lundqvist–Langreth, PRB 82 081101).
constexpr double kVdwDf2Zab = -1.887;
/// @brief Soft q-cutoff [1/Bohr] matching libvdwxc / QE.
constexpr double kVdwDfQCut = 5.0;

/// @brief Compute q₀(ρ, σ=|∇ρ|²) for DF1 (DRSLL) at every grid point.
///
/// @param rho   Total density [e/Bohr³], length N.
/// @param sigma |∇ρ|² [e²/Bohr⁸], length N.
/// @param q0    Output q₀ [1/Bohr], length N (resized).
void vdw_df_q0(const std::vector<double>& rho, const std::vector<double>& sigma,
               std::vector<double>& q0);

/// @brief Compute q₀(ρ, σ) with explicit Z_ab (DF1 or DF2).
///
/// @param rho   Total density [e/Bohr³], length N.
/// @param sigma |∇ρ|² [e²/Bohr⁸], length N.
/// @param q0    Output q₀ [1/Bohr], length N (resized).
/// @param zab   Gradient-correction prefactor (kVdwDf1Zab or kVdwDf2Zab).
void vdw_df_q0(const std::vector<double>& rho, const std::vector<double>& sigma,
               std::vector<double>& q0, double zab);

/// @brief Nonlocal vdW-DF (DRSLL / DF1) energy (+ optional potential).
///
/// @param grid       Uniform orthorhombic cell (Bohr).
/// @param rho        Total density on the grid, length grid.npts().
/// @param sigma      |∇ρ|² on the grid, same length.
/// @param potential  If true, fill `VdwDfResult::v_r` (SC KS potential).
/// @return E_c^{nl} [Hartree] and optional V_nl.
VdwDfResult vdw_df_drsll(const UniformGrid& grid,
                         const std::vector<double>& rho,
                         const std::vector<double>& sigma,
                         bool potential = false);

/// @brief Nonlocal vdW-DF2 (LMKLL / DF2) energy (+ optional potential).
///
/// Same RPS kernel as DRSLL; only Z_ab differs (kVdwDf2Zab = −1.887).
///
/// @param grid       Uniform orthorhombic cell (Bohr).
/// @param rho        Total density on the grid, length grid.npts().
/// @param sigma      |∇ρ|² on the grid, same length.
/// @param potential  If true, fill `VdwDfResult::v_r`.
/// @return E_c^{nl} [Hartree] and optional V_nl.
VdwDfResult vdw_df_lmkll(const UniformGrid& grid,
                         const std::vector<double>& rho,
                         const std::vector<double>& sigma,
                         bool potential = false);

/// @brief Result of a nonlocal vdW evaluation on a uniform grid.
struct NonlocalVdwOnGrid {
    double energy = 0.0;     ///< E_c^{nl} [Hartree].
    std::vector<double> v_r; ///< V_nl [Hartree] (empty if !potential).
};

/// @brief Evaluate nonlocal vdW (DRSLL / LMKLL / rVV10) energy and optional V_nl.
///
/// Exactly one of `vdw_df`, `vdw_df2`, `rvv10` must be true (caller-validated).
/// Spectral |\nablaρ|² is computed inside from `rho`.
///
/// @param grid       Uniform grid (Bohr).
/// @param rho        Total density.
/// @param vdw_df     DRSLL.
/// @param vdw_df2    LMKLL.
/// @param rvv10      rVV10.
/// @param potential  If true, fill `v_r`.
/// @return Energy [Ha] and optional real-space potential.
NonlocalVdwOnGrid nonlocal_vdw_on_grid(const UniformGrid& grid,
                                       const std::vector<double>& rho,
                                       bool vdw_df, bool vdw_df2, bool rvv10,
                                       bool potential);

} // namespace gpw
} // namespace cpu
} // namespace ATV
