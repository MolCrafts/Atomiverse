/// @file cpu/gpw/internal/local_pp.h
/// @brief GTH local short-range pseudopotential on the grid — the potential and
///        its Hellmann-Feynman force, as one pair.
///
/// V_loc^SR(r) = Σ_A exp(-t_A/2) Σ_k C_k^A t_A^k,  t_A = |r-R_A|²/rloc_A².
///
/// CP2K collocates this on the grid and folds it into V_eff, so the integral
/// against the density is shared with the Hartree/XC potential — O(grid) instead
/// of the O(nb²·natoms) per-basis-pair analytic route.
///
/// The two functions here MUST stay together. The energy takes V_loc^SR from
/// `local_pp_grid`, so the gradient of that same energy needs (a) the grid V_eff
/// to carry V_loc^SR (which makes the ∂χ/∂R grid term cover the basis-derivative
/// part) and (b) `add_local_pp_hf_forces` for the term where the potential itself
/// moves with its atom. Assembling the force from the *analytic* per-pair
/// derivative instead differentiates a different (undiscretized) functional than
/// the SCF converged: the two agree only up to the grid discretization error of
/// V_loc^SR, which for a hard semicore PP is enormous. That mismatch was a real
/// bug (ZnO analytic-vs-FD 0.53 Ha/Bohr) — hence one shared kernel, not two.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Collocate the GTH local short-range pseudopotential onto the grid.
///
/// @param grid     Uniform real-space grid.
/// @param atoms    Atoms carrying `rloc` [Bohr] and the `cloc` coefficients [Ha].
/// @param periodic Apply the minimum image convention to r - R_A.
/// @return V_loc^SR sampled on every grid point [Hartree], length grid.npts().
std::vector<double> local_pp_grid(const UniformGrid& grid,
                                  const std::vector<GpwAtom>& atoms,
                                  bool periodic);

/// @brief Add the Hellmann-Feynman force of the grid-collocated local PP.
///
/// F_A += ∫ ρ(r) · exp(-t/2)·[poly'(t) - ½·poly(t)] · 2(r-R_A)/rloc² dr, i.e.
/// -∂/∂R_A of ∫ρ V_loc^SR at fixed ρ. Pairs with `local_pp_grid`; call it exactly
/// when V_loc^SR was put on the grid (and then keep the analytic per-pair
/// `gth_local_sr` out of the Pulay terms, or the basis-derivative part is counted
/// twice).
///
/// @param grid     Same grid the potential was collocated on.
/// @param atoms    Atoms carrying `rloc` [Bohr] and `cloc` [Ha].
/// @param periodic Apply the minimum image convention to r - R_A.
/// @param rho      Electron density on the grid [e/Bohr³], length grid.npts().
/// @param forces   Accumulator, one 3-vector per atom [Hartree/Bohr]; added to.
void add_local_pp_hf_forces(const UniformGrid& grid,
                            const std::vector<GpwAtom>& atoms, bool periodic,
                            const std::vector<double>& rho,
                            std::vector<std::array<double, 3>>& forces);

} // namespace gpw
} // namespace cpu
} // namespace ATV
