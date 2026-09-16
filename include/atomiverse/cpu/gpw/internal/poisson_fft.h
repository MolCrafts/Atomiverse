/// @file cpu/gpw/internal/poisson_fft.h
/// @brief FFT-based periodic Poisson solver on the GPW uniform grid.
///
/// Solves ∇²V = -4π ρ for the periodic Hartree potential (Hartree atomic units)
/// by transforming the collocated density to reciprocal space, applying
/// V(G) = 4π ρ(G)/|G|² (with the G=0 component set to zero — the usual neutral-
/// background convention for a periodic cell), and transforming back. This is
/// the Hartree step of the GPW chain (spec D2), sitting on top of the
/// `collocate.h` density grid.
///
/// Periodic only. The isolated/Martyna-Tuckerman solver needed for the
/// non-periodic H₂O tutorial is a later task (spec D4).
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Result of the periodic Poisson solve.
struct PoissonResult {
    std::vector<double> v_r; ///< Hartree potential V(r) on the grid [Hartree].
    double e_hartree = 0.0;  ///< Hartree energy 0.5 ∫ ρ V dr [Hartree].
};

/// @brief Solve the periodic Poisson equation for the Hartree potential.
/// @param grid  Uniform orthorhombic grid.
/// @param rho_r Electron density on the grid, length grid.npts().
/// @return Hartree potential on the grid and the Hartree energy 0.5 Σ_g w ρ_g V_g.
PoissonResult solve_poisson(const UniformGrid& grid,
                            const std::vector<double>& rho_r);

/// @brief Periodic Poisson with crystal-momentum transfer \(q\) (fractional):
///        \(V(G)=4\pi\rho(G)/|q+G|^2\). For \(q=0\) this is identical to
///        `solve_poisson` (G=0 dropped). For \(q\neq 0\) the G=0 mode is
///        finite (\(4\pi/|q|^2\)) — required for finite-q pair densities in
///        multi-k RPA/MP2.
/// @param grid   Uniform orthorhombic grid.
/// @param rho_r  Density on the grid, length grid.npts().
/// @param q_frac Fractional transfer \(\kappa_q\) in (−1/2, 1/2]³.
/// @return Hartree potential on the grid [Hartree].
PoissonResult solve_poisson_q(const UniformGrid& grid,
                              const std::vector<double>& rho_r,
                              const double q_frac[3]);

} // namespace gpw
} // namespace cpu
} // namespace ATV
