/// @file cuda/driver/criterion.cuh
/// @brief Built-in termination criteria for CUDA staged workflows.
///
/// Criteria are Components (cuda/component/criterion.cuh). The driver builds
/// and destroys each instance through its Plan, executes any measurement
/// bindings, then polls verdict(). All declared convergence requirements must
/// succeed; stop-only guards are neutral unless they request Stop.
///
/// These policies currently judge host ConvergenceMetrics. ConvCheck publishes
/// those metrics after a device reduction and readback. Whole-loop device
/// termination requires a separate device-executable policy contract.
#pragma once

#include "atomiverse/core/precision.h"
#include <cmath>
#include <memory>
#include <string>

#include "atomiverse/core/verdict.h"
#include "atomiverse/cuda/component/criterion.cuh"
#include "atomiverse/cuda/frame/frame.cuh"

namespace ATV {
namespace cuda {

// ─── Built-in handlers ──────────────────────────────────────────────────────
//
// Each of these used to BE a driver category. They are now ~10 lines apiece, and they
// compose: a Minimize run that must also give up on a blown-up geometry is
// `ForceConverged{} + StopOnNaN{}`, which needed a new class before.

/// @brief The five-criteria geometry-optimization convergence test.
///
/// ALL five must hold. It reads `ConvergenceMetrics`, which the Optimizer component publishes
/// (`cuda::minimizer::ConvCheck` does the reduction) — so the tolerances live here, with
/// the *policy*, and not (as they did) as five of the eight positional constructor
/// arguments of `cuda::minimizer::LBFGS`, where changing a tolerance meant touching the
/// algorithm and where `CG` had to carry the identical five-way test.
///
/// Identical rule, identical defaults, identical units as `cpu::ForceConverged` — the
/// two backends must converge on the same criteria or a CPU/GPU parity test means
/// nothing.
class ForceConverged : public Criterion {
  public:
    bool contributes_to_convergence() const override { return true; }
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

    /// @brief Declare convergence when all five criteria hold.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Converged` if every criterion holds, else `Continue`.
    Verdict verdict(const LoopState& s) override {
        const ConvergenceMetrics& m = s.metrics;
        // Judge the METRICS — never the iteration counter. Whether a displacement is
        // meaningful is a fact only the component knows (it has the previous geometry),
        // and it says so in `has_displacement`: `SteepestDescent` never measures one, so
        // it publishes false and the displacement criteria are simply not part of its
        // test. A handler that second-guesses this with `iteration == 0` gets it wrong —
        // in the minimizer phase order the optimizer has ALREADY taken a step by the time
        // `convergence_check` runs, so driver-iteration 0 is not the optimizer's first
        // step.
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

/// @brief Abandon a run whose energy has gone non-finite.
///
/// `Stop`, never `Converged` — which is exactly why the severity order matters. A
/// blown-up system has zero force on every atom (they are all NaN, and NaN fails every
/// `<` comparison), so reporting success for it would be worse than crashing.
///
/// Reads only `ConvergenceMetrics::energy`, i.e. what a component already brought back to the
/// host. It does not — and on this backend must not — read the device frame.
class StopOnNaN : public Criterion {
  public:
    /// @brief Identifier reported as the stop reason.
    /// @return `"stop_on_nan"`.
    std::string name() const override { return "stop_on_nan"; }

    /// @brief Abandon the run when the measured energy is not finite.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Stop` if `metrics.energy` [eV] is NaN or infinite, else `Continue`.
    Verdict verdict(const LoopState& s) override {
        const double e = s.metrics.energy;
        return std::isfinite(e) ? Verdict::Continue : Verdict::Stop;
    }
};

} // namespace cuda
} // namespace ATV
