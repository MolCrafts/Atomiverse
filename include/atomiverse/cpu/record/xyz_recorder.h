/// @file cpu/record/xyz_recorder.h
/// @brief XyzRecorder -- append MD geometry frames to an XYZ trajectory.
///
/// *XYZ* is the plain-text geometry format: one frame is an atom-count line, a
/// free-form comment line, then one `element x y z` line per atom (coordinates
/// in Angstrom). A *trajectory* is many such frames concatenated into a single
/// file — what visualisers and downstream analysis tools read.
///
/// This is the MD-phase trajectory sink: a `Recorder` — the Component kind
/// whose only job is writing outside the process (LAMMPS `dump`) — that binds
/// by default to `md::Phase::end_of_step` (the phase the driver runs once the
/// step's integration is complete) and writes one frame every `freq` steps.
/// Compose it on any other phase through `on(phase).add()`.
///
/// Usage:
/// @code
///   cpu::md::MD sim;
///   sim.set_integrator(cpu::md::VelocityVerlet{1.0});
///   sim.set_record(cpu::XyzRecorder{"traj.xyz", /*freq=*/10});
///   sim.build();
///   sim.run(frame, 1000);   // ctx.step 0…999 -> frames at 0, 10, 20, …
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

/// @brief Append the current geometry to an XYZ trajectory every `freq` steps.
///
/// The first launched write truncates any pre-existing file; every later
/// write of the same recorder appends one frame through molrs. A freshly
/// constructed recorder therefore always starts its own trajectory rather
/// than extending a previous run's.
///
/// The stride gate reads `LaunchContext::step`, which the driver counts from
/// zero and cumulatively across chunked `run(cpu::Frame&, …)` calls — that
/// overload continues one trajectory, while `run(molrs::FrameRef&, …)`
/// materializes a fresh working frame per call and does not
/// (@see cpu::md::MD::run). So an `n`-step run at `freq` writes the steps
/// `0, freq, 2*freq, …` that fall below `n` — a 10-step run at freq 5 writes
/// 2 frames (steps 0 and 5), never 3.
///
/// Each atom's element symbol comes from its `kAtom_type` value, which the
/// underlying writer hands to molrs as an **atomic number Z** (type 18 is
/// written as `Ar`). A frame typed with LAMMPS-style 1..N type indices
/// therefore names the elements of those Z values, not the intended species.
///
/// MPI-aware: the gate is evaluated on the identical `ctx.step` on every
/// rank, so all ranks enter the collective owner gather
/// (`md::mpi::gather_for_root_io`, tag-ordered) in lockstep and only rank 0
/// writes the complete geometry. The gather scratch and the XYZ writer are
/// members, reused across launches (no allocation on the steady-state path).
class XyzRecorder : public cpu::Recorder {
  public:
    /// @brief Construct.
    /// @param filename Output XYZ path [non-empty]. Truncated on the first
    ///                 launched write, appended to afterwards.
    /// @param freq     Write cadence [steps; >= 1] — one frame is written
    ///                 whenever `step % freq == 0`.
    /// @throws std::invalid_argument if `filename` is empty or `freq < 1`.
    XyzRecorder(std::string filename, int freq);

    std::string name() const override { return "xyz"; }
    /* role via Recorder base */

    /// @brief Default binding: after the MD step is complete, behind the
    ///        observers that publish per-step quantities (order +10).
    /// @return One binding on `md::Phase::end_of_step`.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +10)};
    }

    /// @brief Write one trajectory frame when the stride gate opens.
    /// @param ctx Launch context — supplies the step index [steps], the
    ///            frame to persist [Angstrom] and the MPI runtime.
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Number of frames written so far [count]. Rank 0 only; other
    ///        ranks skip the write and stay at 0.
    ///
    /// Reachable after the run only if the driver *borrowed* the recorder
    /// (`set_record_ref(rec)`); `set_record(cpu::XyzRecorder{…})` transfers
    /// ownership into the driver and leaves no handle to ask.
    int n_frames_written() const { return n_written_; }

  private:
    std::string filename_;
    int freq_;
    XYZ writer_;             ///< molrs-backed writer (reused every launch)
    cpu::Frame gathered_{0}; ///< multi-rank gather scratch (rank 0 only)
    int n_written_ = 0;
    bool truncated_ = false;
};

} // namespace cpu
} // namespace ATV
