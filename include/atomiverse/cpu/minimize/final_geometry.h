/// @file cpu/minimize/final_geometry.h
/// @brief FinalGeometry -- single-frame XYZ snapshot of the optimizer output.
///
/// `launch()` only marks that a snapshot is pending; the gather + write
/// happens exactly once in `destroy()` -- the recorders' end-of-run hook,
/// invoked by the driver's `teardown_pass()` after the optimization loop
/// finishes. After `driver.run()` returns, the file holds the last geometry
/// visited by the optimizer -- whether convergence was reached or the
/// iteration budget was exhausted.
///
/// @note Single-frame by design. For a per-step trajectory, use
///       `cpu::minimizer::TrajectoryWriter`.
///
/// MPI-aware: under multi-rank every rank joins a collective owner gather
/// (`md::mpi::gather_to_root`, tag-ordered) and only rank 0 writes the
/// complete geometry.
///
/// Usage:
/// @code
///   opt.on(minimizer::Phase::end_of_step)
///      .add(cpu::minimizer::FinalGeometry{"aspirin_final.xyz"});
/// @endcode
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/recorder.h"

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Writer that persists a single-frame XYZ of the final geometry.
class FinalGeometry : public cpu::Recorder {
  public:
    /// @brief Construct.
    /// @param filename Output XYZ path (overwritten once at end of run).
    explicit FinalGeometry(std::string filename);

    std::string name() const override { return "final_geometry"; }
    /* role via Recorder base */

    std::vector<DefaultBinding> default_bindings() const override {
        // Run late in end_of_step so that converged-step forces/energy have
        // already been written to the frame by earlier components.
        return {ATV::make_binding(ATV::minimizer::Phase::end_of_step, 10)};
    }

    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief End-of-run hook: gather owners to rank 0 and write the file
    ///        once, iff at least one step launched since the last write.
    void destroy(const cpu::DestroyContext& ctx) override;

    /// @brief Number of final-geometry snapshots written (at most one per
    ///        completed `run()`).
    int n_writes() const { return n_writes_; }

  private:
    std::string filename_;
    int n_writes_ = 0;
    bool write_pending_ = false;
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
