/// @file cpu/planewave/internal/convergence.h
/// @brief internal::ConvergenceCheck — density-change SCF convergence test.
///
/// Facade-private residual helper used by `cpu::pw::PW` stages. NOT a
/// `cpu::Component` subclass and never writes `IterationStatus` itself —
/// the Driver Criterion (`PwConverged`) judges the bool this helper sets.
/// `cpu::pw::PW` *is* a `Driver<pw::Phase>` node (restricted fixed-occ path);
/// this type only measures ‖Δρ‖₂.
#pragma once

#include "atomiverse/cpu/planewave/internal/pw_workspace.h"

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

    /// @brief Test convergence against an already-computed density-change
    ///        norm — the CUDA backend reduces ‖ρ − ρ_old‖₂ on the device, so
    ///        it has the norm but not the host arrays. Same counter, same
    ///        warmup policy, one definition.
    /// @param resid_l2 ‖ρ(G) − ρ_old(G)‖₂ (dimensionless, same measure as the
    ///        workspace overload computes).
    /// @return true when past warmup and @p resid_l2 is below tol.
    bool step(AccReal resid_l2);

  private:
    AccReal tol_ = AccReal(1e-8);
    int warmup_ = 2;
    int iter_ = 0;
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
