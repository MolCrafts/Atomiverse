/// @file cpu/planewave/internal/pw_stress.h
/// @brief Plane-wave cell stress (one-SCF NSCF strain FD + full-SCF FD oracle).
#pragma once

#include <array>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

struct PWStrainReference;
struct PWDensityView;

/// @brief Frozen-density NSCF strain-FD stress σ [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
///
/// One reference SCF then 12 NSCF evaluations under symmetric strain at
/// h=2e-3, holding the reference integer PW/density modes and FFT shapes fixed.
/// Density coefficients scale by Ω₀/Ω. This is the numerical Nielsen–Martin
/// HF derivative, not a closed-form tensor. Mechanical sign:
/// \(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\).
///
/// @param frame Geometry [Å]; not mutated.
/// @param method Plane-wave method members (ecut, k-mesh, UPF path, …).
std::array<double, 6> pw_stress_from_density(const cpu::Frame& frame,
                                             const PW& method,
                                             PWStrainReference reference,
                                             PWDensityView density,
                                             const PWResult& result);

/// @brief Strain finite-difference stress oracle [eV/Å³] (12 SCF solves).
///
/// Central differences of SCF energy under symmetric strain at fixed
/// fractional coordinates. Used only for dual-gate tests against the
/// one-SCF frozen-density path.
/// Off-diagonal Voigt entries divide the symmetric-shear derivative by two
/// to return the physical σ_xy, σ_xz and σ_yz tensor components.
///
/// @param frame Geometry [Å]; not mutated.
/// @param method Plane-wave method members.
/// @param h Dimensionless strain step (default 2e-3).
std::array<double, 6> pw_stress_fd_impl(const cpu::Frame& frame,
                                        const PW& method, double h = 2e-3);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
