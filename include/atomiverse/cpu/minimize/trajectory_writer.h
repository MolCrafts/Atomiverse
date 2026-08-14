/// @file cpu/minimizer/trajectory_writer.h
/// @brief TrajectoryWriter -- append geometry frames to an XYZ trajectory.
///
/// Per-step recorder for geometry optimization or AIMD runs. Binds by default
/// to `minimizer::Phase::end_of_step`; compose on any phase via `on().add()`.
///
/// Usage:
/// @code
///   cpu::minimizer::Minimize opt;
///   opt.set_optimizer(cpu::minimizer::LBFGS{/*...*/});
///   opt.on(minimizer::Phase::force)
///      .add(cpu::scf::RHF{cpu::scf::BasisSet::STO3G});
///   opt.on(minimizer::Phase::end_of_step)
///      .add(cpu::minimizer::TrajectoryWriter{"aspirin_traj.xyz", /*stride=*/1});
///   opt.build();
///   opt.run(frame, 200);
/// @endcode
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/recorder.h"
#include "atomiverse/cpu/record/xyz.h"

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Append the current geometry to an XYZ file each step.
///
/// The first launched step truncates any pre-existing file; every later
/// step appends one frame through molrs. Respects `stride` to thin long
/// trajectories. The gather scratch and the XYZ writer are members, reused
/// across launches (no allocation on the steady-state path).
///
/// MPI-aware: under multi-rank every rank joins a collective owner gather
/// (`md::mpi::gather_for_root_io`, which enters `gather_to_root` tag-ordered)
/// and only rank 0 writes the complete geometry.
class TrajectoryWriter : public cpu::Recorder {
  public:
    /// @brief Construct.
    /// @param filename Output XYZ path (appended; existing file is overwritten
    ///                 on the first launched step).
    /// @param stride   Write every `stride` steps [steps]. Any value below 1
    ///                 is silently clamped to 1 (write every step) — this
    ///                 constructor never throws. Its MD-side sibling
    ///                 `cpu::XyzRecorder` deliberately does the opposite and
    ///                 throws on `freq < 1`, so do not read a shared
    ///                 contract into the two.
    explicit TrajectoryWriter(std::string filename, int stride = 1);

    std::string name() const override { return "trajectory_writer"; }
    /* role via Recorder base */

    /// @brief Default binding: at the end of each optimizer iteration.
    /// @return One binding on `minimizer::Phase::end_of_step` (order 0).
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::end_of_step, 0)};
    }

    /// @brief Write one trajectory frame when the stride gate opens.
    /// @param ctx Launch context — supplies the step index [steps], the
    ///            frame to persist [Angstrom] and the MPI runtime.
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Number of frames written so far [count]. Rank 0 only; other
    ///        ranks skip the write and stay at 0.
    int n_frames_written() const { return n_written_; }

  private:
    std::string filename_;
    int stride_;
    XYZ writer_;             ///< molrs-backed writer (reused every launch)
    cpu::Frame gathered_{0}; ///< multi-rank gather scratch (rank 0 only)
    int n_written_ = 0;
    bool truncated_ = false;
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
