/// @file core/verdict.h
/// @brief Verdict + ConvergenceMetrics — the vocabulary a loop's termination is
///        decided in. Backend-agnostic, CUDA-free.
///
/// Termination used to be a **class**. `OptimizeDriver` stopped on convergence,
/// `DynamicsDriver` ran a budget, `SamplingDriver` ran trials, `PathDriver` waited for
/// every replica — six classes differing by one `if`, and a seventh required the moment
/// anyone wanted "stop when the temperature runs away" or "stop after an hour".
///
/// It is a **policy**. This header is the vocabulary it is written in.
#pragma once

namespace ATV {

/// @brief What a criterion says after an iteration. Ordered by severity.
///
/// Composition rule, and it is not negotiable: **Stop > Converged > Continue**. With
/// several criteria attached, the most severe verdict wins. `Stop` outranks `Converged`
/// because a run that has gone numerically insane must not be able to report success by
/// also happening to satisfy a force tolerance.
enum class Verdict {
    Continue = 0, ///< Keep looping.
    Converged =
        1, ///< Terminate — successfully. The answer is the current state.
    Stop =
        2 ///< Terminate — without success (budget, NaN, wall clock, interrupt).
};

/// @brief The most severe of two verdicts.
constexpr Verdict worse_of(Verdict a, Verdict b) {
    return (static_cast<int>(a) >= static_cast<int>(b)) ? a : b;
}

/// @brief What becomes of the iteration just executed. **Not a termination.**
///
/// Monte Carlo's accept/reject is the reason this enum exists, and the reason it is
/// *not* a `Verdict`: rejecting a trial move does not end the run, it discards one
/// iteration's work and keeps going. Conflating the two is what made `SamplingDriver` a
/// class — it had to exist purely to own an `if` that ran between the stages and the
/// loop's termination test.
///
/// So there are two orthogonal policy slots on a loop, and every loop has both:
///
///   - `Criterion`     — *should the loop stop?*        (after the outcome is applied)
///   - `OutcomePolicy` — *does this iteration count?*   (before the loop asks that)
///
/// Metropolis is one implementation of the second. Nothing about it is Monte-Carlo-
/// specific from the loop's point of view — a rejected iteration is a rejected
/// iteration.
enum class Outcome {
    Commit, ///< Keep the iteration's work. The default when no policy is attached.
    Rollback ///< Discard it and restore the pre-iteration state.
};

/// @brief What the loop's components measured this iteration, for criteria to judge.
///
/// Named for what it carries: the convergence-judgment quantities (forces,
/// displacements, ΔE, band fraction) that a termination `Criterion` reads.
///
/// **The split this struct exists to enforce**: an optimizer *measures*; a criterion
/// *decides*. Until now `LBFGS` did both — it carried five convergence tolerances in its
/// constructor and wrote `outer_status.converged = true` itself. So changing the
/// stopping criteria meant editing the optimizer, two algorithms had to duplicate the
/// same five-criteria test, and — the part that actually shipped as a bug, twice — a
/// nested optimizer's convergence stopped the *enclosing* loop.
///
/// Fields not written this iteration keep their previous value; `has_displacement` says
/// whether the displacement pair is meaningful yet (it is not, on iteration 0, when
/// there is no previous geometry to difference against).
struct ConvergenceMetrics {
    double energy = 0.0;    ///< Total energy [eV].
    double max_force = 0.0; ///< Max atomic force magnitude [eV/Å].
    double rms_force = 0.0; ///< RMS force [eV/Å].
    double max_displacement =
        0.0; ///< Max displacement since last iteration [Å].
    double rms_displacement = 0.0; ///< RMS displacement [Å].
    double energy_change = 0.0;    ///< |ΔE| since last iteration [eV].
    bool has_displacement =
        false; ///< False on iteration 0 / after a migration reset.

    /// @brief Fraction of coupled replicas that have met their criterion, in [0,1].
    ///
    /// A band is converged only when this is 1 — the AND that `PathDriver` used to spell
    /// out by hand. Left at 0 by every non-replica loop, which is why
    /// `AllReplicasConverged` is the only criterion that reads it.
    double replicas_converged = 0.0;
};

/// @brief CUDA-only compatibility alias for the pre-rename name.
///
/// The CUDA backend still spells `LoopMetrics` (its handler machinery is
/// unchanged until CUDA grows a Criterion kind — deferred GPU-node work); CPU
/// code must use `ConvergenceMetrics`. Remove with the CUDA criterion migration.
using LoopMetrics = ConvergenceMetrics;

} // namespace ATV
