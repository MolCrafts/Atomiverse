/// @file cpu/record/mrec_recorder.h
/// @brief MrecRecorder -- stream MD frames into a `*.mrec` trajectory store.
///
/// The `*.mrec` sibling of `cpu::XyzRecorder`: a `Recorder` — the Component
/// kind whose only job is writing outside the process (LAMMPS `dump`) — that
/// binds by default to `md::Phase::end_of_step` and hands one frame every
/// `freq` steps to a `cpu::MrecWriter`. Where XYZ is text and geometry-only,
/// the mrec frame is the whole exchange frame (positions, velocities, forces,
/// charges, images, tags, species, masses, box, serializable meta) with a
/// `step` axis and, when the run has an integrator, a `time` axis [fs].
///
/// Usage:
/// @code
///   cpu::md::MD sim;
///   sim.set_integrator(cpu::md::VelocityVerlet{1.0});
///   sim.set_record(cpu::MrecRecorder{"traj.mrec", /*freq=*/10});
///   sim.build();
///   sim.run(frame, 1000);   // ctx.step 0…999 -> frames at 0, 10, 20, …
/// @endcode
///
/// molrs buffers the appended frames and lands them a chunk at a time; the
/// buffered tail is committed by `flush()` (checkpoint), by the automatic
/// landing every `flush_every` frames, and finally by `destroy()` — which the
/// driver fires when the run is torn down (the preset's destructor, or the
/// next `run()` on a different frame). So after `sim.run(...)` returns the
/// last chunk may still be in memory: destroy the driver, or on a borrowed
/// recorder (`set_record_ref`) call `close()`, before reading the store.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/recorder.h"
#include "atomiverse/cpu/component/timestep_protocol.h"
#include "atomiverse/cpu/record/mrec.h"

namespace ATV {
namespace cpu {

/// @brief Append the current frame to a `*.mrec` trajectory every `freq` steps.
///
/// The first launched write mints the store — replacing any store already at
/// the path — so a freshly constructed recorder always starts its own
/// trajectory rather than extending a previous run's. The stride gate reads
/// `LaunchContext::step`, zero-based and cumulative across chunked
/// `run(cpu::Frame&, …)` calls (@see cpu::XyzRecorder for the cadence
/// arithmetic; a 10-step run at freq 5 writes steps 0 and 5).
///
/// **Time axis.** The recorder is a `TimestepConsumer`: `cpu::md::MD` pushes
/// the integrator's dt [fs] before `build()`, and every frame is then stamped
/// `time = step * dt` — the same zero-based iteration index `Thermo` and
/// `XyzRecorder` report. A loop with no `TimestepProvider` (Minimize) records
/// steps only. Whether frames carry a time is fixed by the first append.
///
/// **Schema is pinned by the first frame.** A serializable meta key that
/// first appears on a later step is a molrs error naming the key: mount every
/// observer before the first recorded frame; a run that starts recording a
/// new quantity midway needs a new store.
///
/// MPI-aware: the gate is evaluated on the identical `ctx.step` on every
/// rank, so all ranks enter the collective owner gather
/// (`md::mpi::gather_for_root_io`, tag-ordered) in lockstep and only rank 0
/// appends the complete frame. The gather scratch and the writer are
/// members, reused across launches.
class MrecRecorder : public cpu::Recorder, public cpu::TimestepConsumer {
  public:
    /// @brief Construct.
    /// @param filename    Output store [non-empty, `*.mrec`]. Minted on the
    ///                    first launched write, replacing an existing store.
    /// @param freq        Write cadence [steps; >= 1] — one frame is appended
    ///                    whenever `step % freq == 0`.
    /// @param flush_every Automatic commit cadence [frames]; `0` lets molrs
    ///                    land whole chunks at its own cadence. `1` commits
    ///                    every frame (crash-safe, write-amplifying).
    /// @param durable     Sync the touched files on `flush()` / `close()` /
    ///                    `destroy()` so a returned checkpoint survives a
    ///                    power loss. Automatic landings never sync.
    /// @throws std::invalid_argument if `filename` is empty or does not end in
    ///         `.mrec`, or `freq < 1`.
    MrecRecorder(std::string filename, int freq, std::uint64_t flush_every = 0,
                 bool durable = true);

    std::string name() const override { return "mrec"; }
    /* role via Recorder base */

    /// @brief Default binding: after the MD step is complete, behind the
    ///        observers that publish per-step quantities (order +10).
    /// @return One binding on `md::Phase::end_of_step`.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +10)};
    }

    /// @brief Demand side of the timestep protocol: receives the integrator
    ///        dt so frames can carry a time axis.
    TimestepConsumer* as_timestep_consumer() override { return this; }
    /// @brief A run without an integrator still records (steps only).
    bool timestep_required() const override { return false; }
    /// @brief Receive the nuclear MD timestep [fs].
    /// @throws std::invalid_argument when `dt_fs` is not finite and > 0.
    void set_timestep(double dt_fs) override;

    /// @brief Append one trajectory frame when the stride gate opens.
    /// @param ctx Launch context — supplies the step index [steps], the
    ///            frame to persist and the MPI runtime.
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief End of the run: commit the buffered tail and release the store.
    ///
    /// The driver calls this from teardown paths that include destructors, so
    /// a storage failure here is reported through the engine logger rather
    /// than thrown; call `close()` first when the exception is wanted.
    void destroy(const cpu::DestroyContext& ctx) override;

    /// @brief Checkpoint: commit every buffered frame now.
    /// @throws std::runtime_error naming the store on a storage failure.
    void flush() { writer_.flush(); }

    /// @brief Commit the buffered tail and release the store; idempotent.
    /// @throws std::runtime_error naming the store on a storage failure.
    void close() { writer_.close(); }

    /// @brief Number of frames appended so far [count]. Rank 0 only; other
    ///        ranks skip the write and stay at 0.
    ///
    /// Reachable after the run only if the driver *borrowed* the recorder
    /// (`set_record_ref(rec)`); `set_record(cpu::MrecRecorder{…})` transfers
    /// ownership into the driver and leaves no handle to ask.
    int n_frames_written() const { return n_written_; }

    /// @brief Number of frames committed to the store [count]; rank 0 only.
    std::uint64_t n_frames_committed() const { return writer_.committed(); }

  private:
    std::string filename_;
    int freq_;
    double dt_fs_ = 0.0;     ///< integrator dt [fs]; 0 ⇒ steps only
    MrecWriter writer_;      ///< molrs-backed writer (reused every launch)
    cpu::Frame gathered_{0}; ///< multi-rank gather scratch (rank 0 only)
    int n_written_ = 0;
};

} // namespace cpu
} // namespace ATV
