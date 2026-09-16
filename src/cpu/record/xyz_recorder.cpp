/// @file cpu/record/xyz_recorder.cpp
/// @brief XyzRecorder -- molrs-backed XYZ append on the MD end_of_step phase.
///        MPI-aware: collective owner gather, rank-0 I/O.

#include "atomiverse/cpu/record/xyz_recorder.h"

#include <stdexcept>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/gather_for_io.h"

namespace ATV {
namespace cpu {

XyzRecorder::XyzRecorder(std::string filename, int freq)
    : filename_(std::move(filename)), freq_(freq), writer_(filename_) {
    // Loud on both degenerate inputs rather than silently substituting a
    // default: an empty path writes nowhere, and freq < 1 has no cadence to
    // mean (freq 0 would divide by zero in the gate below).
    if (filename_.empty())
        throw std::invalid_argument(
            "XyzRecorder: filename is empty — pass the output XYZ path");
    if (freq_ < 1)
        throw std::invalid_argument("XyzRecorder: freq must be >= 1 step, got "
                                    + std::to_string(freq_));
}

void XyzRecorder::launch(const cpu::LaunchContext& ctx) {
    // Stride gate first: ctx.step is identical on every rank, so the
    // collective below is still entered in lockstep.
    if ((ctx.step % freq_) != 0)
        return;

    // Publish iteration identity through the Frame's typed metadata channel,
    // so the step travels with the geometry into the XYZ comment line.
    ctx.frame.meta.get(kStep) = static_cast<std::int64_t>(ctx.step);

    // Multi-rank: collective owner gather (every rank participates before
    // the rank gate) so the written frame holds the complete tag-ordered
    // geometry; returns nullptr on non-root ranks.
    const cpu::Frame* src =
        md::mpi::gather_for_root_io(ctx.frame, ctx.runtime, gathered_);
    if (!src)
        return;

    // First write truncates any pre-existing file; later writes append.
    if (!truncated_) {
        writer_.write(*src);
        truncated_ = true;
    } else {
        writer_.append(*src);
    }
    ++n_written_;
}

} // namespace cpu
} // namespace ATV
