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
/// Martyna–Tuckerman (J. Chem. Phys. 110, 2810, 1999): transform the smooth
/// erf-screened Coulomb potential on an auxiliary grid and add its analytic
/// short-range complement. The density needs sufficient vacuum padding so
/// interacting charge separations stay inside half the cell on each axis.
/// The screening parameter and auxiliary-grid convention are documented in
/// the implementation; this is not a spherical Coulomb truncation.
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"   // UniformGrid
#include "atomiverse/cpu/gpw/internal/poisson_fft.h" // PoissonResult

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Solve the isolated (non-periodic) Poisson equation for the Hartree
///        potential via the Martyna–Tuckerman kernel.
/// @param grid Uniform orthorhombic grid with vacuum padding on every axis.
/// @param rho_r Density on the grid, length grid.npts().
/// @return Open-boundary Hartree potential V(r) and energy 0.5 Σ_g w ρ_g V_g.
PoissonResult solve_poisson_mt(const UniformGrid& grid,
                               const std::vector<double>& rho_r);

} // namespace gpw
} // namespace cpu
} // namespace ATV
