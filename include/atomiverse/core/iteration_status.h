/// @file core/iteration_status.h
/// @brief IterationStatus — one loop's outcome. **Owned by the loop, not the Context.**
///
///   - **converged** — the loop terminated successfully. Written ONLY by the driver,
///     from the reduced verdict of its `Criterion`s. A component cannot write it:
///     there is no path from a component to this field, by construction.
///   - **stop_requested** — explicit early termination. A component MAY request this
///     (a NaN detector, a wall-time guard, a user interrupt); a criterion may too.
///   - **reason** — which criterion ended the loop, for logs and diagnostics.
///
/// ## The ownership rule, and the bug it kills
///
/// Each loop node owns its own `IterationStatus` as a member. The Context merely
/// *borrows* a pointer to the active loop's (`Context::loop`, a `LoopServices` slice).
///
/// It used to live on the Context by value, and that shipped a bug twice:
///
///   - probe 2: `LBFGS` wrote `converged = true` unconditionally, because the optimizer
///     owned the stopping criterion.
///   - probe 3: so the first NEB image to converge would have stopped the entire band —
///     and `ReplicaSet` had to grow a second, private convergence table to work around
///     a status it could not trust.
///
/// A pointer that points at the current loop cannot reach the loop above it. The
/// inner-loop-stops-the-outer-loop bug is now unwritable rather than merely forbidden.
///
/// SCF/PW internals never touch this — they own their own convergence, inside their own
/// solver.
#pragma once

#include <string>

namespace ATV {

/// @brief One loop's iteration status. Owned by the loop node; borrowed by the Context.
struct IterationStatus {
    bool converged = false;      ///< Optimizer normal success.
    bool stop_requested = false; ///< Explicit early termination.
    std::string reason;          ///< Human-readable detail.

    /// @brief Reset to "fresh iteration" state.
    void reset() noexcept {
        converged = false;
        stop_requested = false;
        reason.clear();
    }
};

} // namespace ATV
