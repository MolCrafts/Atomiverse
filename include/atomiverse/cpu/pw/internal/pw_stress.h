/// @file cpu/pw/internal/pw_stress.h
/// @brief Analytic plane-wave cell stress (Voigt 6-vector).
#pragma once

#include <array>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Analytic stress σ [eV/Å³], Voigt order xx,yy,zz,xy,xz,yz.
///
/// Includes kinetic (bands), Hartree, local PP Hellmann–Feynman cell
/// derivative, and Ewald ion–ion. XC contribution uses the LDA Parseval
/// form from ρ and V_xc. KB nonlocal stress is included when UPF projectors
/// are present (final diagonalization per k, same as forces).
std::array<double, 6> pw_stress_analytic_impl(const cpu::Frame& frame,
                                              const PWConfig& cfg);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
