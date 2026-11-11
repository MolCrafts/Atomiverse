/// @file cpu/minimize/trajectory_writer.h
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
/// The first write truncates any pre-existing file; every later write
/// appends one frame through molpy. The gather scratch and the XYZ writer
/// are members, reused across launches (no allocation on the steady-state
/// path).
///
/// **Step convention (LAMMPS `minimize` + `dump`).** A frame's `step` meta
/// is the number of completed optimizer iterations and the frame is the
/// geometry after exactly that many (LAMMPS `update->ntimestep`). The run's
/// setup writes the initial geometry as step 0 (LAMMPS `Min::setup` →
/// `Output::setup`); iteration `ctx.step` writes step `ctx.step + 1` when
/// that is a multiple of `stride`; a minimization stopped by a criterion
/// before its budget writes its final geometry even off cadence (LAMMPS
/// `Min::run`); no step is written twice (LAMMPS `last_dump`). So a run that
/// converges after 7 iterations at stride 3 writes steps 0, 3, 6, 7.
///
/// MPI-aware: under multi-rank every rank joins a collective owner gather
/// (`md::mpi::gather_for_root_io`, which enters `gather_to_root` tag-ordered)
/// and only rank 0 writes the complete geometry.
class TrajectoryWriter : public cpu::Recorder {
  public:
    /// @brief Construct.
    /// @param filename Output XYZ path (appended; existing file is overwritten
    ///                 on the first launched step).
    /// @param stride   Write every `stride` iterations. Any value below 1
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

    /// @brief Start of the run: write the initial geometry as step 0.
    void setup_output(const cpu::LaunchContext& ctx) override;
    /// @brief End of iteration `ctx.step`: write it as step `ctx.step + 1`
    ///        when on stride.
    void launch(const cpu::LaunchContext& ctx) override;
    /// @brief Early stop: write the final geometry as step `ctx.step`
    ///        unless already written.
    void final_output(const cpu::LaunchContext& ctx) override;

    /// @brief Number of frames written so far [count]. Rank 0 only; other
    ///        ranks skip the write and stay at 0.
    int n_frames_written() const { return n_written_; }

  private:
    std::string filename_;
    int stride_;
    XYZ writer_;             ///< molpy-backed writer (reused every launch)
    cpu::Frame gathered_{0}; ///< multi-rank gather scratch (rank 0 only)
    int n_written_ = 0;
    bool truncated_ = false;
    BigInt last_step_ = -1; ///< LAMMPS `last_dump`

    /// @brief Gather and write the geometry labelled `step` [iterations].
    void write_(const cpu::LaunchContext& ctx, BigInt step);
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
