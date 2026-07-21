/// @file cpu/driver/criteria.h
/// @brief The built-in termination Criteria — the loop node's stock verdicts.
///
/// Each of these used to BE a driver category, then a non-component policy
/// object. They are now ~10 lines apiece of verdict-only `Criterion`, and they
/// compose: a
/// Minimize run that must also give up on numerical insanity is
/// `add_criterion(ForceConverged{}) + add_criterion(StopOnNaN{})`, which needed
/// a new class twice over before.
///
/// They live in `cpu/driver/` — not `cpu/component/` — because they are the
/// loop node's built-in termination defaults (`Minimize` attaches
/// `ForceConverged` as a build-time default in `install_defaults()`,
/// `PathDriver` attaches `AllReplicasConverged` in its constructor), and
/// because their whole point is cross-workflow
/// generality: `StopOnNaN` serves MD, Minimize, Path and MC alike. The
/// `Criterion` base (the component KIND) is `cpu/component/criterion.h`;
/// domain-specific criteria stay in their domain (SCF's `ScfConverged` is in
/// `cpu/scf/stages/`).
///
/// ## Writing your own
///
/// @code
///   class StopWhenTooHot : public cpu::Criterion {
///     public:
///       explicit StopWhenTooHot(double t_max) : t_max_(t_max) {}
///       std::string name() const override { return "stop_when_too_hot"; }
///       void launch(const cpu::LaunchContext&) override {} // verdict-only
///       Verdict verdict(const cpu::LoopState& s) override {
///           return (temperature(s.frame) > t_max_) ? Verdict::Stop
///                                                  : Verdict::Continue;
///       }
///     private:
///       double t_max_;
///   };
///
///   sim.add_criterion(StopWhenTooHot{2000.0});
/// @endcode
///
/// This is LAMMPS's `fix halt`, written from OUTSIDE the library.
#pragma once

#include <cmath>
#include <string>

#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/facade/loop.h"

namespace ATV {
namespace cpu {

/// @brief The five-criteria geometry-optimization convergence test.
///
/// ALL five must hold. It reads `ConvergenceMetrics`, which the Optimizer
/// component publishes — so the tolerances live here, with the *policy*, and
/// not (as they did) as five of the eight positional constructor arguments of
/// `LBFGS`, where changing a tolerance meant touching the algorithm and where
/// `CG` had to duplicate the identical test.
class ForceConverged : public Criterion {
  public:
    /// @brief Construct the convergence test from its five tolerances.
    /// @param ftol Max force [eV/Å].
    /// @param rms_ftol RMS force [eV/Å].
    /// @param max_dtol Max displacement [Å].
    /// @param rms_dtol RMS displacement [Å].
    /// @param etol Energy change [eV].
    explicit ForceConverged(double ftol = 0.05, double rms_ftol = 0.03,
                            double max_dtol = 0.002, double rms_dtol = 0.001,
                            double etol = 1e-5)
        : ftol_(ftol), rms_ftol_(rms_ftol), max_dtol_(max_dtol),
          rms_dtol_(rms_dtol), etol_(etol) {}

    /// @brief Identifier reported as the stop reason.
    /// @return `"force_converged"`.
    std::string name() const override { return "force_converged"; }

    /// @brief Verdict-only: nothing to measure, nothing to launch.
    void launch(const LaunchContext&) override {}

    /// @brief Declare convergence when all five criteria hold.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Converged` if every criterion holds, else `Continue`.
    Verdict verdict(const LoopState& s) override {
        const ConvergenceMetrics& m = s.metrics;
        // Judge the METRICS — never the iteration counter. Whether a displacement is
        // meaningful is a fact only the component knows (it has the previous geometry),
        // and it says so in `has_displacement`. A criterion that second-guesses this with
        // `iteration == 0` gets it wrong, and I know because I wrote that first: in the
        // minimizer phase order the optimizer has ALREADY taken a step by the time
        // `convergence_check` runs, so driver-iteration 0 is not the optimizer's first
        // step. Skipping the displacement test there declared an untouched lattice
        // converged, having moved nothing.
        const bool disp_ok = !m.has_displacement
                             || (m.max_displacement < max_dtol_
                                 && m.rms_displacement < rms_dtol_);
        const bool ok = m.max_force < ftol_ && m.rms_force < rms_ftol_
                        && disp_ok && m.energy_change < etol_;
        return ok ? Verdict::Converged : Verdict::Continue;
    }

  private:
    double ftol_, rms_ftol_, max_dtol_, rms_dtol_, etol_;
};

/// @brief A coupled chain is converged when EVERY replica is — never when the first is.
///
/// The AND that `PathDriver` used to spell out inline. Reads
/// `ConvergenceMetrics::replicas_converged`, which the band optimizer publishes
/// as a fraction.
class AllReplicasConverged : public Criterion {
  public:
    /// @brief Identifier reported as the stop reason.
    /// @return `"all_replicas_converged"`.
    std::string name() const override { return "all_replicas_converged"; }

    /// @brief Verdict-only: nothing to measure, nothing to launch.
    void launch(const LaunchContext&) override {}

    /// @brief Declare convergence when the whole band has converged.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Converged` when `metrics.replicas_converged` reaches 1.0, else
    ///         `Continue`.
    Verdict verdict(const LoopState& s) override {
        return (s.metrics.replicas_converged >= 1.0) ? Verdict::Converged
                                                     : Verdict::Continue;
    }
};

/// @brief Abandon a run whose energy has gone non-finite.
///
/// `Stop`, never `Converged` — which is exactly why the severity order matters. A blown-up
/// system has zero force on every atom (they are all NaN, and NaN fails every `<`
/// comparison... but a *diverged* one can genuinely park at a force minimum of garbage).
/// Reporting success for it would be worse than crashing.
class StopOnNaN : public Criterion {
  public:
    /// @brief Identifier reported as the stop reason.
    /// @return `"stop_on_nan"`.
    std::string name() const override { return "stop_on_nan"; }

    /// @brief Verdict-only: nothing to measure, nothing to launch.
    void launch(const LaunchContext&) override {}

    /// @brief Abandon the run when the measured energy is not finite.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Stop` if `metrics.energy` [eV] is NaN or infinite, else `Continue`.
    Verdict verdict(const LoopState& s) override {
        const double e = s.metrics.energy;
        return std::isfinite(e) ? Verdict::Continue : Verdict::Stop;
    }
};

} // namespace cpu
} // namespace ATV
