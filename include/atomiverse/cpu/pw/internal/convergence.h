/// @file cpu/pw/internal/convergence.h
/// @brief internal::ConvergenceCheck — density-change SCF convergence test.
///
/// Facade-private step helper owned by `cpu::pw::PWSolver`. NOT a
/// `cpu::Component` subclass; it never touches the driver loop's
/// `IterationStatus` (borrowed by the Context as `ctx.loop`) — the plane-wave
/// SCF manages its own internal convergence, exactly like `cpu::scf::SCFSolver`.
#pragma once

#include "atomiverse/cpu/pw/internal/pw_workspace.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Converged when ‖ρ(G) − ρ_old(G)‖₂ < tol after a warmup period.
class ConvergenceCheck {
  public:
    /// @param tol    Convergence tolerance on the density-change L2 norm.
    /// @param warmup Iterations to skip before testing (mixer settling).
    explicit ConvergenceCheck(AccReal tol, int warmup = 2)
        : tol_(tol), warmup_(warmup) {}

    /// @brief Reset the iteration counter (call once per SCF run).
    void reset() { iter_ = 0; }

    /// @brief Test convergence against the current / previous density.
    /// @param ws Plane-wave workspace.
    /// @return true when past warmup and the density change is below tol.
    bool step(const PWWorkspace& ws);

  private:
    AccReal tol_ = AccReal(1e-8);
    int warmup_ = 2;
    int iter_ = 0;
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
