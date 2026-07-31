/// @file cpu/gpw/internal/gpw_stress.h
/// @brief GPW cell stress: analytic fixed-P (product) + multi-SCF strain-FD oracle.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

/// @brief Analytic (Hellmann–Feynman fixed-\(P\)) Voigt stress [Hartree/Bohr³].
///
/// One SCF → central FD of the non-self-consistent total energy \(E[P;\varepsilon]\)
/// under orthorhombic diagonal strain at fixed fractional coordinates (GPW
/// analogue of plane-wave Nielsen–Martin in this tree). Supports Gamma RKS,
/// Gamma UKS (spin densities \(P_\alpha,P_\beta\) + \(W_\alpha,W_\beta\)),
/// multi-k closed-shell RKS (k-summed fixed-\(P_k\) Lagrangian), multi-k
/// collinear UKS (fixed \(\{P_k^\sigma,W_k^\sigma\}\), spin XC), Dudarev
/// DFT+U on Γ and multi-k (BZ-averaged Mulliken \(E_U\) at strained
/// \(S\)/\(S_k\)), and geometry-only dispersion (D3 / D3(BJ) / DFT-D4 / TS /
/// XDM / MBD as configured). Off-diagonal = 0.
///
/// @param atoms Geometry [Bohr] + basis/PP (same as `GpwDft::run`).
/// @param method SCF configuration (`cell` must match atoms' box).
/// @param h Dimensionless strain step for the fixed-\(P\) energy FD (default 2e-3).
/// @param axes Per-axis mask (x,y,z); a masked axis skips samples and reports 0.
/// @param guess_C Optional warm-start MO coefficients for the reference SCF.
/// @return Voigt stress σ [Hartree/Bohr³], order xx,yy,zz,xy,xz,yz.
std::array<double, 6>
gpw_stress_analytic_impl(const std::vector<GpwAtom>& atoms,
                         const GpwDft& method, double h = 2e-3,
                         const std::array<bool, 3>& axes = {true, true, true},
                         const linalg::RealMatrix* guess_C = nullptr);

/// @brief Analytic fixed-\(P\) Voigt stress at a **supplied** state
///        [Hartree/Bohr³] — no reference SCF.
///
/// The payload-extraction + fixed-\(P\) FD tail of
/// `gpw_stress_analytic_impl`: builds \(P/W\) (Γ) or \(\{P_k, W_k\}\)
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
/// off-diagonal are 0.
std::array<double, 6>
gpw_stress_fd_impl(const std::vector<GpwAtom>& atoms, const GpwDft& method,
                   double h = 2e-3,
                   const std::array<bool, 3>& axes = {true, true, true},
                   const linalg::RealMatrix* guess_C = nullptr);

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
