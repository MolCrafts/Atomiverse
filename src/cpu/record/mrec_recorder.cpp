/// @file cpu/record/mrec_recorder.cpp
/// @brief MrecRecorder -- molrs-backed `*.mrec` append on the MD end_of_step
///        phase. MPI-aware: collective owner gather, rank-0 I/O.

#include "atomiverse/cpu/record/mrec_recorder.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/communication/gather_for_io.h"

namespace ATV {
namespace cpu {

MrecRecorder::MrecRecorder(std::string filename, int freq,
                           std::uint64_t flush_every, bool durable)
    : filename_(std::move(filename)), freq_(freq),
      writer_(filename_, flush_every, durable) {
    // Loud on both degenerate inputs rather than silently substituting a
    // default: an empty path writes nowhere (the writer already rejects it
    // for lacking the .mrec suffix, this names the recorder), and freq < 1
    // has no cadence to mean (freq 0 would divide by zero in the gate below).
    if (filename_.empty())
        throw std::invalid_argument(
            "MrecRecorder: filename is empty — pass the output *.mrec path");
    if (freq_ < 1)
        throw std::invalid_argument("MrecRecorder: freq must be >= 1 step, got "
                                    + std::to_string(freq_));
}

void MrecRecorder::set_timestep(double dt_fs) {
    if (!(dt_fs > 0.0) || !std::isfinite(dt_fs))
        throw std::invalid_argument("MrecRecorder: timestep must be finite "
                                    "and > 0 fs, got "
                                    + std::to_string(dt_fs));
    dt_fs_ = dt_fs;
}

void MrecRecorder::launch(const cpu::LaunchContext& ctx) {
    // Stride gate first: ctx.step is identical on every rank, so the
    // collective below is still entered in lockstep.
    if ((ctx.step % freq_) != 0)
        return;

    // Multi-rank: collective owner gather (every rank participates before
    // the rank gate) so the written frame holds the complete tag-ordered
    // state; returns nullptr on non-root ranks.
    const cpu::Frame* src =
        md::mpi::gather_for_root_io(ctx.frame, ctx.runtime, gathered_);
    if (!src)
        return;

    // `step` is the driver's zero-based iteration index (what Thermo and
    // XyzRecorder report); the time axis is that index on the integrator's
    // clock, present only when an integrator pushed a dt.
    const bool has_time = dt_fs_ > 0.0;
    const double time = has_time ? static_cast<double>(ctx.step) * dt_fs_ : 0.0;
    writer_.append(*src, static_cast<std::int64_t>(ctx.step), time, has_time);
    ++n_written_;
}

void MrecRecorder::destroy(const cpu::DestroyContext& ctx) {
    (void)ctx;
    // Reached from `flush_live_()` inside the preset's destructor as well as
    // from an explicit teardown: throwing here would terminate the process,
    // so the failure is reported, not raised. `close()` is the throwing door.
    try {
        writer_.close();
    } catch (const std::exception& e) {
        ATV_ERROR("cpu::MrecRecorder: closing '{}' at run teardown failed "
                  "(frames after the last checkpoint may be lost): {}",
                  filename_, e.what());
    }
}

} // namespace cpu
} // namespace ATV
