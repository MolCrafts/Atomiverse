/// @file cpu/component/criterion.h
/// @brief cpu::Criterion — termination as a Component kind.
///
/// A `Criterion` is a Component that also answers ONE question after every
/// iteration of the loop it is registered on: *should the loop stop?* It returns
/// a `Verdict` — `Continue`, `Converged` or `Stop` — and nothing else. It only
/// ever RETURNS a verdict; it has no field with which to write the loop's
/// status, so a component still cannot declare its own loop converged. The
/// driver PULLS every registered criterion's `verdict()`, composes them by
/// severity (Stop > Converged > Continue), all-reduces once across MPI ranks,
/// and — the driver alone — writes `converged` / `stop_requested`.
///
/// ## Why a Component, and why not a template
///
/// Termination used to be a free-standing loop-handler policy object — a first-
/// class citizen that was NOT a component, polled through a private side channel.
/// That was the last place where something with a per-iteration job description
/// lived outside the Component lifecycle and ownership rules. A `Criterion` is a
/// Component: same build/launch/destroy lifecycle, same `add_criterion(rvalue)` /
/// `add_criterion_ref(lvalue)` ownership convention as every other registration
/// path.
///
/// Being a Component is also what lets a criterion bind a phase and MEASURE:
/// e.g. SCF's `ScfConverged` binds `scf::Phase::check` to read the energy Δ /
/// DIIS residual it then judges. A verdict-only criterion (the common case — the
/// built-ins in `cpu/driver/criteria.h`) declares no `default_bindings()` and
/// simply enters the driver's polling list.
///
/// It is deliberately **not** a template: `verdict()` sees a `LoopState` and
/// **never a phase**. A criterion that knew about `md::Phase` could only ever
/// serve MD. Knowing nothing about stages, one criterion serves every workflow —
/// `StopOnNaN` is as meaningful to a band of NEB images as to an MD trajectory.
/// Type erasure in this architecture happens at the **node** boundary (a driver
/// is a node; a component is a node) — never at the stage boundary; stages stay
/// typed enums, all the way down.
///
/// ## MPI
///
/// **A criterion's verdict is reduced across ranks by the driver, not by the
/// criterion.** Ranks that disagree about when to stop deadlock at the next
/// collective, and that is a bug no criterion author should have to know about.
/// Write `verdict()` as if the process were alone; the driver all-reduces the
/// severity.
#pragma once

#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Everything a criterion is allowed to look at.
struct LoopState {
    int iteration;           ///< 0-based index of the iteration just completed.
    const cpu::Frame& frame; ///< The system state as it now stands.
    const cpu::Context&
        runtime; ///< Execution context (threads, MPI, service slices).
    const ConvergenceMetrics&
        metrics; ///< What this loop's components measured. @see facade/loop.h
};

/// @brief A termination criterion: a Component the driver polls once per iteration.
class Criterion : public Component {
  public:
    /// @brief Judge the iteration just completed.
    ///
    /// Non-const (a criterion may update its own measurement state), takes the
    /// loop's read-only `LoopState`. Only RETURNS a verdict — it cannot write
    /// the loop's status; the driver composes the returned verdicts and writes
    /// convergence itself.
    /// @param state Iteration index, frame, context, metrics.
    /// @return `Continue`, `Converged` or `Stop`. @see ATV::Verdict
    virtual ATV::Verdict verdict(const cpu::LoopState& state) = 0;
};

} // namespace cpu
} // namespace ATV
