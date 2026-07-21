/// @file cpu/gpw/internal/mt_poisson.h
/// @brief Non-periodic (isolated / cluster) Poisson solver on the GPW grid.
///
/// Needed for isolated molecules — CP2K's `&POISSON PERIODIC NONE / PSOLVER MT`,
/// e.g. the getting-started `h2o.inp` tutorial (spec D4). Unlike the bare
/// periodic solver (`poisson_fft.h`), this returns the OPEN-boundary Hartree
/// potential, free of periodic-image interactions, so the energy of a charged or
/// dipolar isolated distribution is box-size independent once the cell padding
/// exceeds the charge extent.
///
/// Formulation: spherically-truncated Coulomb kernel (Rozzi et al.,
/// PRB 73, 205119 (2006), 0D case) — a single reciprocal-space kernel change
/// from the periodic solver,
///     v(G) = (4π/|G|²)[1 - cos(|G| R_c)]   (G≠0),   v(0) = 2π R_c²,
/// with cutoff radius R_c = min(L)/2 (the inscribed sphere). This reproduces the
/// isolated 1/r potential exactly when the density is contained within R_c and
/// the box holds the cutoff sphere. CP2K's PSOLVER MT (Martyna-Tuckerman 1999) is
/// the erf/erfc-screened variant; it plugs in behind this same interface for
/// exact CP2K-MT parity later.
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"   // UniformGrid
#include "atomiverse/cpu/gpw/internal/poisson_fft.h" // PoissonResult

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Solve the isolated (non-periodic) Poisson equation for the Hartree
///        potential via the truncated-Coulomb kernel.
/// @param grid  Uniform orthorhombic grid; the charge must lie within R_c=min(L)/2.
/// @param rho_r Density on the grid, length grid.npts().
/// @return Open-boundary Hartree potential V(r) and energy 0.5 Σ_g w ρ_g V_g.
PoissonResult solve_poisson_mt(const UniformGrid& grid,
                               const std::vector<double>& rho_r);

} // namespace gpw
} // namespace cpu
} // namespace ATV
