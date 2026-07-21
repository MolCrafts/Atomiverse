/// @file cuda/driver/loop_handler.cuh
/// @brief LoopHandler — termination as a reusable, user-definable policy (CUDA).
///
/// The CUDA mirror of `cpu/driver/loop_handler.h`. Same vocabulary
/// (`ATV::Verdict` / `ATV::LoopMetrics`, from the CUDA-free `facade/loop.h`), same
/// composition rule (**Stop > Converged > Continue**), same split of labour: a
/// component *measures* and publishes into `ctx.runtime.loop.metrics`; a handler
/// *decides*.
///
/// ## Why this is not a template
///
/// `LoopHandler` sees a `LoopState` and **never a phase**. That is the whole point: a
/// handler that knew about `md::Phase` could only ever serve MD. Knowing nothing about
/// stages, one handler serves every workflow — `StopOnNaN` is as meaningful to a
/// minimization as to an MD trajectory, and neither of them had any way to say so
/// before.
///
/// Type erasure in this architecture happens at the **node** boundary (a driver is a
/// node; a component is a node) and at the **policy** boundary (here). It never happens
/// at the stage boundary — stages stay typed enums, all the way down.
///
/// ## Writing one
///
/// @code
///   class StopWhenTooHot : public cuda::LoopHandler {
///     public:
///       explicit StopWhenTooHot(double t_max) : t_max_(t_max) {}
///       const char* name() const override { return "stop_when_too_hot"; }
///       Verdict after_iteration(const cuda::LoopState& s) override {
///           return (s.metrics.energy > t_max_) ? Verdict::Stop : Verdict::Continue;
///       }
///     private:
///       double t_max_;
///   };
///
///   sim.add_handler(StopWhenTooHot{2000.0});
/// @endcode
///
/// This is LAMMPS's `fix halt`, and it was **impossible** to write before: "stop or not"
/// was a private `if` inside `cuda::DynamicsDriver` / `cuda::OptimizeDriver`.
///
/// ## Note on device residency
///
/// A handler runs on the **host**, once per iteration, after the stages. It therefore
/// judges only what a component has already brought back to the host — that is what
/// `LoopMetrics` is: the host-side mirror of a device reduction (see
/// `cuda::minimizer::ConvCheck`, which fuses the four force/displacement metrics into
/// one kernel and one pinned D2H). A handler must never dereference device pointers off
/// `LoopState::frame`; the frame is there for structural facts (atom count, box), not
/// for per-atom reads.
///
/// ## MPI
///
/// `cpu::CPUDriver` all-reduces the handler verdict across ranks (`MPI_MAX` over the
/// severity) so that ranks cannot disagree about when to stop and deadlock at the next
/// collective. The CUDA driver is **single-rank** — it has no communicator and no
/// `comm/` layer wired into the driver — so there is no reduction here. Handler authors
/// write `after_iteration` as if the process were alone on both backends; the day CUDA
/// grows a multi-GPU driver loop, the reduction goes into `CUDADriver::poll_handlers_`,
/// exactly as on the CPU, and no handler changes.
#pragma once

#include <cmath>
#include <memory>
#include <string>

#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/facade/loop.h"

namespace ATV {
namespace cuda {

/// @brief Everything a handler is allowed to look at.
struct LoopState {
    int iteration; ///< 0-based index of the iteration just completed.
    const cuda::Frame& frame; ///< Device-resident state as it now stands.
    const cuda::Context&
        runtime; ///< Execution context (streams, service slices).
    const LoopMetrics& metrics; ///< What this loop's components measured.
                                ///< @see facade/loop.h
};

/// @brief A termination policy, asked once after every iteration of the loop it is on.
class LoopHandler {
  public:
    virtual ~LoopHandler() = default;

    /// @brief Identifier, used in the stop reason the driver reports.
    virtual const char* name() const = 0;

    /// @brief Judge the iteration just completed.
    /// @param state Iteration index, frame, context, metrics.
    /// @return `Continue`, `Converged` or `Stop`. @see ATV::Verdict
    virtual Verdict after_iteration(const LoopState& state) = 0;
};

// ─── Built-in handlers ──────────────────────────────────────────────────────
//
// Each of these used to BE a driver category. They are now ~10 lines apiece, and they
// compose: a Minimize run that must also give up on a blown-up geometry is
// `ForceConverged{} + StopOnNaN{}`, which needed a new class before.

/// @brief The five-criteria geometry-optimization convergence test.
///
/// ALL five must hold. It reads `LoopMetrics`, which the Optimizer component publishes
/// (`cuda::minimizer::ConvCheck` does the reduction) — so the tolerances live here, with
/// the *policy*, and not (as they did) as five of the eight positional constructor
/// arguments of `cuda::minimizer::LBFGS`, where changing a tolerance meant touching the
/// algorithm and where `CG` had to carry the identical five-way test.
///
/// Identical rule, identical defaults, identical units as `cpu::ForceConverged` — the
/// two backends must converge on the same criteria or a CPU/GPU parity test means
/// nothing.
class ForceConverged : public LoopHandler {
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
    const char* name() const override { return "force_converged"; }

    /// @brief Declare convergence when all five criteria hold.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Converged` if every criterion holds, else `Continue`.
    Verdict after_iteration(const LoopState& s) override {
        const LoopMetrics& m = s.metrics;
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
/// Reads only `LoopMetrics::energy`, i.e. what a component already brought back to the
/// host. It does not — and on this backend must not — read the device frame.
class StopOnNaN : public LoopHandler {
  public:
    /// @brief Identifier reported as the stop reason.
    /// @return `"stop_on_nan"`.
    const char* name() const override { return "stop_on_nan"; }

    /// @brief Abandon the run when the measured energy is not finite.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Stop` if `metrics.energy` [eV] is NaN or infinite, else `Continue`.
    Verdict after_iteration(const LoopState& s) override {
        const double e = s.metrics.energy;
        return std::isfinite(e) ? Verdict::Continue : Verdict::Stop;
    }
};

} // namespace cuda
} // namespace ATV
