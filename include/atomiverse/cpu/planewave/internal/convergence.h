/// @file cpu/planewave/internal/convergence.h
/// @brief internal::ConvergenceCheck — unmixed SCF residual convergence test.
///
/// Facade-private residual helper used by `cpu::pw::PW` stages. NOT a
/// `cpu::Component` subclass and never writes `IterationStatus` itself —
/// the Driver Criterion (`PwConverged`) judges the bool this helper sets.
/// `cpu::pw::PW` *is* a `Driver<pw::Phase>` node (restricted fixed-occ path);
/// this type only measures ‖Δρ‖₂.
#pragma once

#include "atomiverse/core/precision.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Converged when ‖ρ_out(G) − ρ_in(G)‖₂ < tol after warmup.
class ConvergenceCheck {
  public:
    /// @param tol    Convergence tolerance on the unmixed residual L2 norm.
    /// @param warmup Iterations to skip before testing (mixer settling).
    explicit ConvergenceCheck(AccReal tol, int warmup = 2)
        : tol_(tol), warmup_(warmup) {}

    /// @brief Reset the iteration counter (call once per SCF run).
    void reset() { iter_ = 0; }

    /// @brief Test the residual measured before damping or Pulay mixing.
    /// @param resid_l2 ‖ρ_out(G) − ρ_in(G)‖₂.
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
