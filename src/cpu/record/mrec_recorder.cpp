/// @file cpu/record/mrec_recorder.cpp
/// @brief MrecRecorder -- molpy-backed `*.mrec` append on the MD end_of_step
///        phase. MPI-aware: collective owner gather, rank-0 I/O.

#include "atomiverse/cpu/record/mrec_recorder.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/core/frame_schema.h"
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

// LAMMPS src/output.cpp Output::setup / Output::write and src/verlet.cpp
// Verlet::run, e8c4c278a8. Copyright (2003) Sandia Corporation.
// GPL-2.0-only. `update->ntimestep` is incremented at the top of every step,
// so an end-of-step dump at timestep N holds the state after N steps; setup
// writes the current timestep when it is a multiple of `every`, and
// `last_dump` keeps a timestep from being written twice across runs.

void MrecRecorder::setup_output(const cpu::LaunchContext& ctx) {
    // ctx.step = steps completed so far (identical on every rank).
    if ((ctx.step % freq_) != 0 || ctx.step == last_step_)
        return;
    write_(ctx, ctx.step);
}

void MrecRecorder::launch(const cpu::LaunchContext& ctx) {
    // End of the step with zero-based index ctx.step: ctx.step + 1 steps are
    // complete. Stride gate first: identical on every rank, so the
    // collective below is still entered in lockstep.
    const BigInt step = loop_output_step(ctx);
    if ((step % freq_) != 0 || step == last_step_)
        return;
    write_(ctx, step);
}

void MrecRecorder::final_output(const cpu::LaunchContext& ctx) {
    // LAMMPS Min::run: a minimization that stops early writes its final
    // state at the final timestep, unless already written (`last_dump`).
    if (ctx.step == last_step_)
        return;
    write_(ctx, ctx.step);
}

void MrecRecorder::write_(const cpu::LaunchContext& ctx, BigInt step) {
    // Every rank records the step before the rank gate (they all take part
    // in the gather), so the never-twice rule stays rank-identical.
    last_step_ = step;

    // Multi-rank: collective owner gather (every rank participates before
    // the rank gate) so the written frame holds the complete tag-ordered
    // state; returns nullptr on non-root ranks.
    const cpu::Frame* src =
        md::mpi::gather_for_root_io(ctx.frame, ctx.runtime, gathered_);
    if (!src)
        return;

    // The time axis is the completed-step count on the integrator's clock,
    // present only when an integrator pushed a dt.
    const bool has_time = dt_fs_ > 0.0;
    const double time = has_time ? static_cast<double>(step) * dt_fs_ : 0.0;
    append_frame(tag_ordered_(*src), static_cast<std::int64_t>(step), time,
                 has_time);
    ++n_written_;
}

const cpu::Frame& MrecRecorder::tag_ordered_(const cpu::Frame& src) {
    // The multi-rank gather is tag-ordered already.
    if (&src == &gathered_)
        return gathered_;
    const int n = src.nlocal();
    const TagInt* tags = src.span(kAtom_tag).data();
    if (std::is_sorted(tags, tags + n))
        return src;
    // Single rank: the live rows are spatially sorted. Write a tag-ordered
    // copy (the run state itself is FrameImmutable to a recorder).
    order_.resize(static_cast<std::size_t>(n));
    std::iota(order_.begin(), order_.end(), 0);
    std::stable_sort(order_.begin(), order_.end(),
                     [&](int a, int b) { return tags[a] < tags[b]; });
    ordered_ = src.clone_structure(n);
    for (int row = 0; row < n; ++row)
        ordered_.copy_row_from(src, row, order_[static_cast<std::size_t>(row)]);
    ordered_.box = src.box;
    ordered_.meta.copy_serializable_from(src.meta);
    return ordered_;
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
