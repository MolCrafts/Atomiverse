/// @file cpu/gpw/internal/gpw_stress.h
/// @brief GPW cell stress: fixed-P strain FD and reconverged strain check.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

/// @brief Fixed-\(P\) strain-FD stress at a **supplied** state
///        [Hartree/Bohr³] — no reference SCF.
///
/// Builds \(P/W\) (Γ) or \(\{P_k, W_k\}\)
/// (multi-k, dual per spin for UKS) from `scf` and central-FDs the
/// non-self-consistent Lagrangian under orthorhombic diagonal strain.
/// Used by the zero-SCF linearized XL-BOMD warm step (shadow-density
/// stress at \(P = D = q[n]\), \(W\) from the \(H[n]\) eigenpairs).
///
/// @param atoms Geometry [Bohr] + basis/PP (same as `GpwDft::run`).
/// @param method SCF configuration (`cell` must match atoms' box).
/// @param scf Stationary payload (`converged` must be true): Γ needs
///        `P`/`C`/`eps` (+ β for UKS); multi-k needs `C_occ_k`/`eps_k`
///        (+ β for UKS).
/// @param h Dimensionless strain step (default 2e-3).
/// @param axes Per-axis mask (x,y,z); a masked axis reports exactly 0.
/// @return Voigt stress σ [Hartree/Bohr³], order xx,yy,zz,xy,xz,yz.
std::array<double, 6>
gpw_stress_at_state_impl(const std::vector<GpwAtom>& atoms,
                         const GpwDft& method, const GpwResult& scf,
                         double h = 2e-3,
                         const std::array<bool, 3>& axes = {true, true, true});

/// @brief Multi-SCF strain-FD Voigt stress [Hartree/Bohr³] (oracle / twin).
///
/// Orthorhombic only: diagonal components from central FD of fully re-converged
/// SCF energy under axis-aligned strain at fixed fractional coordinates;
/// off-diagonal are 0. One unstrained SCF supplies the common initial density
/// for every independent +/- strain sample (both spins for UKS). It changes
/// only initialization; every sample still meets the method's SCF thresholds.
/// @param atoms Reference atoms [Bohr] and basis/PP data.
/// @param method SCF parameters; cell must match the reference atoms.
/// @param h Dimensionless central strain step.
/// @param axes Diagonal components to evaluate; masked components are zero.
/// @param guess_C Optional initial reference orbitals; strain samples use the
///        converged reference density, preserving both UKS spin channels.
/// @return Voigt stress [Ha/Bohr³], xx, yy, zz, xy, xz, yz.
std::array<double, 6>
gpw_stress_fd_impl(const std::vector<GpwAtom>& atoms, const GpwDft& method,
                   double h = 2e-3,
                   const std::array<bool, 3>& axes = {true, true, true},
                   const linalg::RealMatrix* guess_C = nullptr);

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
