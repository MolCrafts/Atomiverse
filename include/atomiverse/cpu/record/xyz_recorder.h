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
///   sim.run(frame, 1000);   // frames at steps 0, 10, 20, …, 1000
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
/// write of the same recorder appends one frame through molpy. A freshly
/// constructed recorder therefore always starts its own trajectory rather
/// than extending a previous run's.
///
/// **Step convention (LAMMPS `dump`).** A frame's step (the `step` meta in
/// its comment line) is the number of completed MD steps and the frame holds
/// the state after exactly that many steps (LAMMPS `update->ntimestep`,
/// src/verlet.cpp `Verlet::run`). At the start of every run `setup_output`
/// writes the current state when its step is a multiple of `freq` (LAMMPS
/// `Output::setup`); the end_of_step launch of the step with zero-based
/// index `ctx.step` writes step `ctx.step + 1` when that is a multiple of
/// `freq` (LAMMPS `Output::write`); a step is never written twice (LAMMPS
/// `last_dump`). The completed-step count is cumulative across chunked
/// `run(cpu::Frame&, …)` calls — that overload continues one trajectory,
/// while `run(molrs::FrameRef&, …)` materializes a fresh working frame per
/// call and does not (@see cpu::md::MD::run). So a fresh `n`-step run at
/// `freq` writes the steps `0, freq, 2*freq, …` up to and including `n` —
/// a 10-step run at freq 5 writes 3 frames (steps 0, 5 and 10).
/// `cpu::minimizer::Minimize` follows the same convention per iteration
/// and also writes the final state, off cadence, when a criterion stops it
/// early (`final_output`, LAMMPS `Min::run`). In a single-pass
/// `cpu::Evaluate` (LAMMPS `run 0`) the pass writes its state as step
/// `ctx.step` whatever the cadence.
///
/// Each atom's element symbol comes from its species `kAtom_atomic_number`
/// (Z = 18 is written as `Ar`, Z = 0 as the wildcard). The force-field type
/// is never written as an element: a frame without a species column (a
/// coarse-grained model, a LAMMPS data file) is refused with the how-to.
///
/// MPI-aware: the gate is evaluated on the identical step on every
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
    ///                 whenever the completed-step count is a multiple of
    ///                 `freq` (including the initial state, step 0).
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

    /// @brief Start of a run (LAMMPS `Output::setup`): write the current
    ///        state as step `ctx.step` when it is a multiple of `freq` and
    ///        was not already written.
    /// @param ctx Launch context — `ctx.step` is the completed-step count.
    void setup_output(const cpu::LaunchContext& ctx) override;

    /// @brief End of a step (LAMMPS `Output::write`): write the state as
    ///        step `ctx.step + 1` when that is a multiple of `freq`.
    /// @param ctx Launch context — supplies the zero-based index of the step
    ///            just integrated, the frame to persist [Angstrom] and the
    ///            MPI runtime.
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Minimization stopped before its budget (LAMMPS `Min::run`):
    ///        write the final state as step `ctx.step` (iterations
    ///        completed), off cadence, unless that step is already written.
    void final_output(const cpu::LaunchContext& ctx) override;

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
    XYZ writer_;             ///< molpy-backed writer (reused every launch)
    cpu::Frame gathered_{0}; ///< multi-rank gather scratch (rank 0 only)
    int n_written_ = 0;
    bool truncated_ = false;
    BigInt last_step_ = -1; ///< LAMMPS `last_dump`: last step written

    /// @brief Gather and write the current geometry labelled `step`
    ///        [completed steps]; collective, rank 0 writes.
    void write_(const cpu::LaunchContext& ctx, BigInt step);
};

} // namespace cpu
} // namespace ATV
