/// @file cpu/record/mrec.h
/// @brief MrecWriter -- streaming `*.mrec` trajectory writer backed by molrs.
///
/// *mrec* is molrs' scientific record store (a Zarr v3 tree under a `*.mrec/`
/// path). This writer is the engine's trajectory output path: one writer per
/// run, one `cpu::Frame` per `append`. molrs buffers the appended frames and
/// lands them whole chunks at a time; `flush()` is the checkpoint (every
/// buffered frame is committed, durably when the writer was made `durable`)
/// and `close()` ends the run. The first appended frame pins the schema — a
/// block or column it did not carry can never be added later — and a frame
/// that omits a block carries the previous content forward, so a producer
/// writes static blocks once and only the changing ones afterwards.
///
/// CUDA-safe: the molrs handles live behind a private implementation, so this
/// header pulls neither `rust/cxx.h` nor the generated bridge.
///
/// @see cpu::MrecRecorder (cpu/record/mrec_recorder.h) for the MD Recorder.
/// @see cpu::write_frame_zarr (cpu/io/frame_writer.h) for the single-frame
///      checkpoint door and cpu::ZarrReader for reading frame 0 back.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace ATV {
namespace cpu {

class Frame; // fwd -- CUDA-safe

/// @brief Append `cpu::Frame`s to one `*.mrec` trajectory through molrs.
///
/// Each append exports the frame through the canonical exchange
/// (`cpu::to_molrs`: positions, velocities, forces, charges, images, tags,
/// box and every serializable meta entry) plus the two per-run inputs the
/// exchange skips on purpose, `atomic_number` and `mass` — a trajectory is
/// read by tools that never saw the input file, so species travel with it.
/// molrs dedups a block whose bytes did not change, so re-sending the static
/// columns costs a compare, not a write.
///
/// The store is minted lazily on the first `append`; a store already standing
/// at the path is replaced then (a fresh writer starts its own trajectory,
/// the same rule as `cpu::XYZ::write`). Steps must strictly increase, and
/// whether frames carry a time is fixed by the first append. A closed writer
/// refuses further appends — one writer records one run.
class MrecWriter {
  public:
    /// @brief Construct; nothing touches the filesystem until `append`.
    /// @param path        Output store path. Must end in `.mrec` — molrs
    ///                    identifies a scientific record by that suffix and
    ///                    refuses the retired `.zarr` brand.
    /// @param flush_every Automatic landing cadence [frames]; every this many
    ///                    appended frames molrs commits the buffer on its own.
    ///                    `0` leaves the cadence to molrs (one inner chunk).
    /// @param durable     `true` syncs the touched files on `flush()` /
    ///                    `close()` so a returned call means the frames
    ///                    survive a power loss; automatic landings never sync.
    /// @throws std::invalid_argument if `path` is empty or lacks the `.mrec`
    ///         suffix.
    MrecWriter(std::string path, std::uint64_t flush_every, bool durable);

    /// @brief Best-effort `close()`: an I/O failure here is logged through the
    ///        engine logger, never thrown. Call `close()` yourself to get the
    ///        exception.
    ~MrecWriter();
    MrecWriter(const MrecWriter&) = delete;
    MrecWriter& operator=(const MrecWriter&) = delete;
    /// @brief Movable — a driver takes its recorders by value
    ///        (`set_record(cpu::MrecRecorder{…})`). The moved-from writer is
    ///        empty: its destructor is a no-op and `append` refuses it.
    MrecWriter(MrecWriter&&) noexcept; // defined where Impl is complete
    /// @brief Not assignable: overwriting a live writer would drop its
    ///        buffered frames without a `close()`.
    MrecWriter& operator=(MrecWriter&&) = delete;

    /// @brief Buffer one frame.
    /// @param frame    Host-resident frame; owned rows `[0, nlocal)` are
    ///                 written [Angstrom, Angstrom/fs, eV/Angstrom, e].
    /// @param step     Frame's step index; must exceed the previous one.
    /// @param time     Frame's time [fs]; read only when `has_time`.
    /// @param has_time Whether this run carries a time axis (fixed by the
    ///                 first append).
    /// @throws std::logic_error   after `close()`.
    /// @throws std::runtime_error naming the path when molrs rejects the frame
    ///         (non-increasing step, a block / column / meta key outside the
    ///         pinned schema, a dtype change) or the store cannot be written.
    void append(const Frame& frame, std::int64_t step, double time,
                bool has_time);

    /// @brief Commit every buffered frame (the checkpoint). No-op before the
    ///        first append and after `close()`.
    /// @throws std::runtime_error naming the path on a storage failure.
    void flush();

    /// @brief Commit whatever is buffered and release the writer. Idempotent.
    /// @throws std::runtime_error naming the path on a storage failure; the
    ///         writer is released either way and what was already committed
    ///         stays readable.
    void close();

    /// @brief Frames committed to the store so far [count]. Live count while
    ///        open; after a successful `close()` equals `appended()`.
    std::uint64_t committed() const;

    /// @brief Frames handed to `append` so far [count].
    std::uint64_t appended() const noexcept { return appended_; }

    /// @brief Whether `close()` has run.
    bool closed() const noexcept { return closed_; }

    /// @brief Output store path.
    const std::string& path() const noexcept { return path_; }

  private:
    struct Impl;

    std::string path_;
    std::uint64_t flush_every_;
    bool durable_;
    std::uint64_t appended_ = 0;
    std::uint64_t committed_at_close_ = 0;
    bool closed_ = false;
    std::unique_ptr<Impl> impl_;
};

} // namespace cpu
} // namespace ATV
