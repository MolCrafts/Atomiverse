/// @file cpu/pw/internal/pw_forces.h
/// @brief Analytic ionic forces for plane-wave DFT (local HF + Ewald + KB later).
///
/// After one SCF, F_I = F_I^loc + F_I^nl + F_I^Ewald. Plane waves are
/// cell-fixed (no AO Pulay). See `pw-breadth-01-analytic-forces` spec.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Analytic forces F_a = −dE/dR_a [eV/Å] after a full SCF.
///
/// Local Hellmann–Feynman + Ewald + nonlocal Kleinman–Bylander (UPF).
/// Toy path has F_nl = 0. Non-converged SCF still returns forces at the
/// last density (with a warning).
///
/// @param frame Geometry [Å]; not mutated.
/// @param cfg   Plane-wave configuration.
/// @param energy_ry_out Optional: total (or Mermin free) energy [Ry] of the
///        SCF run here, so a caller needing both quantities pays for one SCF.
/// @return Forces [eV/Å], one xyz triple per atom.
std::vector<std::array<double, 3>>
pw_forces_analytic_impl(const cpu::Frame& frame, const PW& method,
                        double* energy_ry_out = nullptr);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
