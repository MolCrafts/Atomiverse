/// @file cpu/gpw/internal/rvv10.h
/// @brief Nonlocal rVV10 correlation for GPW grids.
///
/// Revised VV10 (Sabatini, Gorni, de Gironcoli, Phys. Rev. B **87**, 041108(R)
/// (2013)) evaluated with the Román–Pérez–Soler FFT method, matching CP2K
/// `VDW_POTENTIAL NON_LOCAL TYPE RVV10` (kernel table `rVV10_kernel_table.dat`,
/// default b = 6.3, C = 0.0093).
///
/// Energy (atomic units):
///
///   E_c^{nl} = β ∫ ρ dr + ½ ∬ θ(r) φ(r,r') θ(r') …
///
/// with q₀ = w₀/κ saturated to the q-mesh, θ_α = c ρ^{3/4} p_α(q₀), and the
/// community rVV10 kernel φ_αβ(k). Product path is **self-consistent**: each
/// SCF iteration adds V_nl = δE_c^{nl}/δρ to the KS effective potential.
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Default rVV10 range-separation parameter b (Sabatini et al.).
constexpr double kRvv10B = 6.3;
/// @brief Default rVV10 gradient coefficient C.
constexpr double kRvv10C = 0.0093;
/// @brief Soft q-cutoff = last q-mesh point [1/Bohr].
constexpr double kRvv10QCut = 0.5;
/// @brief Soft q-floor = first q-mesh point [1/Bohr].
constexpr double kRvv10QMin = 1.0e-4;

/// @brief Result of an rVV10 nonlocal evaluation.
struct Rvv10Result {
    double energy = 0.0;   ///< E_c^{nl} [Hartree] (kernel + β∫ρ).
    double e_kernel = 0.0; ///< Kernel (nonlocal) piece only [Hartree].
    double e_beta = 0.0;   ///< β∫ρ local piece [Hartree].
    /// Local KS potential V_nl [Hartree] on the grid (empty if energy-only).
    std::vector<double> v_r;
};

/// @brief Compute saturated q₀(ρ, |∇ρ|²) for rVV10 at every grid point.
///
/// @param rho   Total density [e/Bohr³], length N.
/// @param sigma |∇ρ|² [e²/Bohr⁸], length N.
/// @param q0    Output q₀ [1/Bohr], length N (resized).
/// @param b     rVV10 b parameter (default 6.3).
/// @param C     rVV10 C parameter (default 0.0093).
void rvv10_q0(const std::vector<double>& rho, const std::vector<double>& sigma,
              std::vector<double>& q0, double b = kRvv10B, double C = kRvv10C);

/// @brief Nonlocal rVV10 energy (+ optional potential) on a periodic grid.
///
/// @param grid       Uniform orthorhombic cell (Bohr).
/// @param rho        Total density on the grid, length grid.npts().
/// @param sigma      |∇ρ|² on the grid (spectral or finite-difference).
/// @param b          rVV10 b (default 6.3).
/// @param C          rVV10 C (default 0.0093).
/// @param potential  If true, fill `Rvv10Result::v_r` (SC KS potential).
/// @return E_c^{nl} [Hartree] and optional V_nl.
Rvv10Result rvv10_energy(const UniformGrid& grid,
                         const std::vector<double>& rho,
                         const std::vector<double>& sigma, double b = kRvv10B,
                         double C = kRvv10C, bool potential = false);

} // namespace gpw
} // namespace cpu
} // namespace ATV
