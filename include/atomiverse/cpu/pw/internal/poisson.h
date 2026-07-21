/// @file cpu/pw/internal/poisson.h
/// @brief internal::PoissonSolver — Hartree potential in reciprocal space.
///
/// Facade-private step helper owned by `cpu::pw::PWSolver`. NOT a
/// `cpu::Component` subclass and never registered into a Plan
/// (gaussian-SCF parity: see `cpu/scf/internal/`).
#pragma once

#include "atomiverse/cpu/pw/internal/pw_workspace.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Solves V_H(G) = 4π ρ(G) / G² with the G=0 term zeroed.
class PoissonSolver {
  public:
    /// @brief Compute the Hartree potential from `ws.rho_G` into `ws.V_H_G`.
    /// @param ws Plane-wave workspace [densities in Bohr⁻³, V in Hartree].
    void step(PWWorkspace& ws);
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
