/// @file cpu/planewave/internal/pw_stress.h
/// @brief Plane-wave cell stress (Nielsen–Martin analytic + strain-FD oracle).
#pragma once

#include <array>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Nielsen–Martin analytic stress σ [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
///
/// One SCF then Hellmann–Feynman cell derivatives: kinetic (bands), Hartree,
/// local PP (radial form-factor \(G\)-derivative), Ewald ion–ion, LDA XC
/// volume term, and Kleinman–Bylander nonlocal (UPF projectors). Sign and
/// scale match strain-FD: mechanical \(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\).
///
/// @param frame Geometry [Å]; not mutated.
/// @param method Plane-wave method members (ecut, k-mesh, UPF path, …).
/// @param energy_ry_out Optional: total energy [Ry] from the production SCF.
std::array<double, 6> pw_stress_analytic_impl(const cpu::Frame& frame,
                                              const PW& method,
                                              double* energy_ry_out = nullptr);

/// @brief Strain finite-difference stress oracle [eV/Å³] (12 SCF solves).
///
/// Central differences of SCF energy under symmetric strain at fixed
/// fractional coordinates. Used only for dual-gate tests against the
/// analytic path.
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
