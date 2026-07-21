/// @file cuda/minimizer/conv_check.cuh
/// @brief ConvCheck — the shared convergence-METRIC reduction for the CUDA Optimizer
///        components (`SteepestDescent` / `CG` / `LBFGS`).
///
/// **It measures. It does not decide.** It used to do both: it carried the five
/// tolerances and wrote `outer_status.converged = true` itself. So the criteria could
/// not be changed without touching the algorithm, and — the part that shipped as a bug
/// on the CPU twice — an inner loop's convergence stopped the loop above it.
///
/// Now it reduces the four metrics on the device, brings them (plus the potential
/// energy) to the host in one pinned D2H, and PUBLISHES them into
/// `ctx.runtime.loop.metrics`. A `cuda::LoopHandler` — `cuda::ForceConverged` by
/// default, attached by the `Minimize` preset — reads them and returns the `Verdict`.
/// The tolerances live with the policy, in the handler.
///
/// Per the CLAUDE.md "no duplication across methods" rule the shared reduction lives
/// here once, owned by each Optimizer component (it is NOT a Plan component itself).
///
/// Execution-model classification: **host-dependent compute**. The metric reduction is
/// a GPU-pure fused kernel, but the metrics must reach the host every iteration — the
/// driver polls its handlers on the host after each `step_once()`. `measure()` therefore
/// issues one pinned D2H + `cudaStreamSynchronize` per minimizer iteration. This is the
/// documented exception for the minimizer (an iteration is dominated by the force
/// evaluation; the deferred `AsyncBridge` ring is not applicable because its K-step lag
/// would overrun the convergence point).
///
/// @see cuda/driver/loop_handler.cuh for the handler that judges these metrics
/// @see cpu/minimizer/lbfgs.cpp for the CPU publish-don't-decide sibling
#pragma once

#include <vector>

#include "atomiverse/cpu/minimizer/lbfgs.h" // cpu::minimizer::OptRecord
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace minimizer {

/// @brief Per-step optimization record — shared with the CPU backend so
///        diagnostics tooling reads one type.
using OptRecord = ATV::cpu::minimizer::OptRecord;

/// @brief Fused force/displacement metric reduction, published to the loop.
///
/// Lifecycle mirrors the owning component: `build()` allocates the
/// device accumulators + pinned host mirror, `measure()` runs once per
/// `convergence_check` phase, `destroy()` releases.
///
/// Copyable only before `build()` (raw device pointers; components are
/// copied into the Plan at registration time, before any allocation).
class ConvCheck {
  public:
    /// @brief Allocate device metric accumulators + pinned host mirror.
    /// @param n_atoms Atom count (fixes the RMS normalizations).
    void build(int n_atoms);

    /// @brief Release all resources allocated in build().
    void destroy();

    /// @brief Reduce the metrics, sync, publish them, record the trace entry.
    ///
    /// Launches the fused metric kernel, copies the four metrics + the frame potential
    /// energy D2H into pinned memory, synchronizes the compute stream, appends an
    /// `OptRecord` to @p trace, and writes `ctx.runtime.loop.metrics` — max/RMS force,
    /// max/RMS displacement, energy, |ΔE|, and `has_displacement`.
    ///
    /// Displacement and energy-change criteria are not meaningful at `step == 0`
    /// (nothing has moved yet, and there is no previous energy): displacement is then
    /// flagged `has_displacement = false`, and `energy_change` is published as a
    /// large sentinel so no handler can call an unstarted run converged. Exactly as on
    /// the CPU.
    ///
    /// Because `measure()` ends with a stream synchronize, any D2H copies the caller
    /// enqueued on the same stream beforehand are complete when it returns (LBFGS
    /// batches its line-search scalars this way).
    ///
    /// @param ctx     Launch context (frame, stream, loop services).
    /// @param x_prev  Pre-step positions, planar [x:0..n, y:n..2n, z:2n..3n] device
    ///                buffer (saved by the optimizer's search_direction step). Pass
    ///                `nullptr` when the optimizer keeps no previous geometry
    ///                (`SteepestDescent`) — the displacement metrics are then simply
    ///                not measured, and the handler is told so. May be stale at
    ///                step 0 — it is not read then.
    /// @param step    Optimizer step counter (post search_direction increment,
    ///                mirroring the CPU components).
    /// @param trace   Per-step diagnostics sink (host).
    void measure(const LaunchContext& ctx, const work_t* x_prev, int step,
                 std::vector<OptRecord>& trace);

    /// @brief Potential energy read back by the most recent measure() [eV].
    double last_energy() const { return last_energy_; }

    /// @brief Max force magnitude from the most recent measure() [eV/Angstrom].
    double max_force() const { return max_force_; }

  private:
    int n_ = 0;                   ///< Atom count.
    work_t* d_metrics_ = nullptr; ///< Device {max_f, sum_f2, max_disp, sum_d2}.
    work_t* h_metrics_ = nullptr; ///< Pinned mirror of d_metrics_.
    double* h_energy_ = nullptr;  ///< Pinned mirror of frame.scalars->energy.
    double energy_prev_ = 0;      ///< Energy at the previous measure [eV].
    double last_energy_ = 0;      ///< Energy at the latest measure [eV].
    double max_force_ = 0; ///< Max |F| at the latest measure [eV/Angstrom].
};

} // namespace minimizer
} // namespace cuda
} // namespace ATV
