/// @file cuda/record/mrec_recorder.cuh
/// @brief MrecRecorder -- stream CUDA MD frames into a `*.mrec` trajectory,
///        asynchronously and losslessly, through the driver's shared capture
///        buffer.
///
/// The CUDA sibling of `cpu::MrecRecorder` (cpu/record/mrec_recorder.h): the
/// same constructor, cadence, columns, time axis, flush/close behaviour and
/// the same molpy writer (`cpu::MrecWriter`). What differs is only how a
/// frame reaches the host: the recorder is a `cuda::AsyncConsumer` of the
/// driver's one `cuda::CaptureBuffer` (iron law 10):
///
/// @code
///   stepping thread  admit the step's capture on the host ledger — with
///                    space: host counters only, no device read, no wait;
///                    budget full: wait for a consumer to release a frame
///                    (lossless backpressure, the sanctioned exception to
///                    iron law 10), or throw a consumer's failure
///   compute stream   [graph-capturable] the step's state is frozen once into
///                    a pinned slot shared by every consumer due on the step
///   worker thread    converts the slot to a cpu::Frame (rows in tag order),
///                    appends it through cpu::MrecWriter, releases the slot
/// @endcode
///
/// **Lossless backpressure.** The buffer is the driver's: its byte budget
/// (`Driver::set_capture_buffer_bytes`, Python `MD(capture_buffer_bytes=…)`,
/// default CMake `ATV_CAPTURE_BUFFER_BYTES`) bounds every frame in flight
/// across all of the driver's asynchronous consumers, and a step captured for
/// several of them is one frame of it. No frame is ever dropped. The first
/// wait of a run is logged, the run's waits are summarised at its end, and
/// the buffer's cumulative blocked time, blocking episodes and peak occupancy
/// are recorded in the store's `status` section (`buffer`) and returned by
/// the driver's `capture_stats()`.
///
/// **Failures.** An append that fails (disk full, I/O error) — or any other
/// consumer of the driver failing — stops the run: the next step (or the
/// end-of-run retrieve) throws `std::runtime_error` naming the failed
/// consumer, the step and the cause. The store keeps the frames written
/// before the failure and records `state = "failed"` with the error.
///
/// **Completion.** The end-of-run retrieve drains the buffer: `run()`
/// returns once every captured frame is appended. `flush()` / `close()`
/// commit them.
///
/// Usage:
/// @code
///   cuda::md::MD sim;
///   sim.set_integrator(cuda::md::VelocityVerlet{1.0});
///   sim.set_record(cuda::MrecRecorder{"traj.mrec", /*freq=*/10});
///   sim.build();
///   sim.run(frame, 1000);   // frames at steps 0, 10, 20, …, 1000
/// @endcode
///
/// As on CPU, after `run()` returns every frame is appended but the tail may
/// not be committed: destroy the driver, or on a borrowed recorder
/// (`set_record_ref`) call `flush()` or `close()`, before reading the store.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/asynchronous/async_consumer.cuh"
#include "atomiverse/cuda/component/timestep_protocol.cuh"

namespace ATV {
namespace cpu {
class Frame; // fwd -- the worker-side exchange frame
} // namespace cpu

namespace cuda {

/// @brief Append the frozen device state to a `*.mrec` trajectory every
///        `freq` steps; frames cross to the host asynchronously, never
///        dropped.
///
/// **Columns** (identical to `cpu::MrecRecorder`): positions [Å] (primary +
/// correction reconstructed in FP64), velocities [Å/fs], forces [eV/Å]
/// (fixed point decoded), charges [e], images (int32 per component, so
/// unwrapped coordinates are `x + ix·a + iy·b + iz·c`), tags, mass [amu],
/// the cell, the serializable system scalars (energies, kinetic, conserved,
/// virial tensors) and `step`, plus each identity column the device frame
/// carries — force-field type as `type_id`, species Z as `atomic_number`
/// (a coarse-grained run has the first only, a QM geometry the second); an
/// absent column is absent from the store, never zero-filled, and a type is
/// never written as a species. Both are static for the run and copied once
/// when the buffer is allocated. Rows are written in ascending tag order on
/// both backends. `time = step · dt` [fs] when an integrator published a dt;
/// a run without one records steps only.
///
/// **Step convention (LAMMPS `dump`)**, exactly as on CPU: a frame's `step`
/// is the number of completed MD steps and the frame holds the state after
/// exactly that many steps (LAMMPS `update->ntimestep`); `time = step · dt`.
/// At the start of every run `setup_output` captures the current state when
/// its step is a multiple of `freq` (LAMMPS `Output::setup`); the in-loop
/// capture at the end of the step with zero-based index `k` is labelled
/// `k + 1` and fires when that is a multiple of `freq` (LAMMPS
/// `Output::write`). The completed-step count is cumulative over continued
/// `run()` calls, and a step is never written twice (LAMMPS `last_dump`),
/// so `run(frame, 4); run(frame, 4)` at freq 2 records steps 0, 2, 4, 6, 8
/// with step 4 once. The cadence is decided by the host step counter
/// (`frequency()` / `frequency_phase()`, no device read). Under
/// `cuda::minimizer::Minimize` a step is an optimizer iteration (LAMMPS
/// `Min::setup` / `Min::run`): the initial state is step 0 and a criterion
/// stop before the budget also captures the final state off cadence
/// (`final_output`). The recorder asks the force field for the energy and
/// the virial on its steps (`demand()`; the CPU engine evaluates both every
/// step), so the recorded `energy*` and `stress` meta belong to the recorded
/// step.
///
/// **One recorder records one run.** The first committed frame replaces any
/// store already at the path; after `close()` / teardown the recorder
/// refuses a new `build()` and any further capture.
class MrecRecorder : public AsyncRecorder, public TimestepConsumer {
  public:
    /// @brief Construct; nothing touches the device or the filesystem yet.
    /// @param filename     Output store [non-empty, `*.mrec`].
    /// @param freq         Write cadence [steps; >= 1].
    /// @param flush_every  Automatic commit cadence [frames]; `0` lets molpy
    ///                     land whole chunks at its own cadence.
    /// @param durable      Sync the touched files on `flush()` / `close()` /
    ///                     teardown.
    /// @throws std::invalid_argument on an empty path, a path without the
    ///         `.mrec` suffix or `freq < 1`.
    MrecRecorder(std::string filename, int freq, std::uint64_t flush_every = 0,
                 bool durable = true);
    ~MrecRecorder() override;
    /// @brief Movable before `build()` — the driver takes recorders by value.
    MrecRecorder(MrecRecorder&&) noexcept;
    MrecRecorder& operator=(MrecRecorder&&) = delete;
    MrecRecorder(const MrecRecorder&) = delete;
    MrecRecorder& operator=(const MrecRecorder&) = delete;

    std::string name() const override { return "mrec"; }
    /// @brief `cuda::MrecRecorder: writing '<path>'` (errors, warnings).
    std::string capture_name() const override;

    /// @brief Host-side cadence gate on the host step counter: launched at
    ///        the end of step index `k` when `(k + 1) % freq == 0`.
    int frequency() const override { return freq_; }
    /// @brief Energy and virial of each recorded step are evaluated.
    EvaluationDemand demand() const override {
        return {true, true, freq_, freq_ - 1};
    }
    /// @brief Every per-atom column but the static identity.
    CaptureColumns capture_columns() const override;

    /// @brief After the step is complete, behind the observers (order +10),
    ///        the same slot as `cpu::MrecRecorder`.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +10)};
    }

    TimestepConsumer* as_timestep_consumer() override { return this; }
    /// @brief A run without an integrator still records (steps only).
    bool timestep_required() const override { return false; }
    /// @brief Receive the nuclear MD timestep [fs].
    /// @throws std::invalid_argument when `dt_fs` is not finite and > 0.
    void set_timestep(double dt_fs) override;

    /// @brief Checkpoint: every frame captured so far is appended and
    ///        committed.
    ///
    /// Waits for the compute stream and the capture buffer, so call it
    /// between `run()` calls, never from inside the loop.
    /// @throws std::runtime_error naming the store on a storage failure, or
    ///         carrying a consumer failure the run has not raised yet.
    void flush();

    /// @brief Drain, commit the tail, write the `status` section (buffer
    ///        accounting, and the failure if any) and release the store;
    ///        idempotent. A capture after `close()` throws std::logic_error.
    /// @throws std::runtime_error naming the store on a storage failure.
    void close();

    /// @brief Frames appended to the store so far [count] (live).
    std::uint64_t n_frames_written() const;
    /// @brief Frames committed to the store so far [count].
    std::uint64_t n_frames_committed() const;

  protected:
    /// @brief Worker-thread sink for one converted frame (rows in tag
    ///        order). Default: `cpu::MrecWriter::append`. Runs off the
    ///        stepping thread; frames arrive in step order.
    virtual void append_frame(const cpu::Frame& frame, std::int64_t step,
                              double time, bool has_time);

    /// @throws std::logic_error after `close()` (one recorder, one run).
    void on_build(const BuildContext& ctx) override;
    void consume(const CaptureView& view) override;
    /// @brief Teardown: commit, write the status, release. A storage
    ///        failure is logged, not thrown; call `close()` first to get it.
    void on_destroy(const DestroyContext& ctx) override;
    /// @throws std::logic_error after `close()`.
    void require_open() const override;

  private:
    struct Impl;

    /// @brief `close()`; with @p raise false (teardown) failures are logged.
    void close_(bool raise);

    std::string filename_;
    int freq_;
    double dt_fs_ = 0.0; ///< integrator dt [fs]; 0 ⇒ steps only
    std::unique_ptr<Impl> impl_;
};

} // namespace cuda
} // namespace ATV
