/// @file cpu/record/xyz_recorder.cpp
/// @brief XyzRecorder -- molpy-backed XYZ append on the MD end_of_step phase.
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

// LAMMPS src/output.cpp Output::setup / Output::write and src/verlet.cpp
// Verlet::run, e8c4c278a8. Copyright (2003) Sandia Corporation.
// GPL-2.0-only. `update->ntimestep` is incremented at the top of every step,
// so an end-of-step dump at timestep N holds the state after N steps; setup
// writes the current timestep when it is a multiple of `every`, and
// `last_dump` keeps a timestep from being written twice across runs.

void XyzRecorder::setup_output(const cpu::LaunchContext& ctx) {
    // ctx.step = steps completed so far (identical on every rank).
    if ((ctx.step % freq_) != 0 || ctx.step == last_step_)
        return;
    write_(ctx, ctx.step);
}

void XyzRecorder::launch(const cpu::LaunchContext& ctx) {
    // End of the step with zero-based index ctx.step: ctx.step + 1 steps are
    // complete. Stride gate first: identical on every rank, so the
    // collective below is still entered in lockstep.
    const BigInt step = loop_output_step(ctx);
    if ((step % freq_) != 0 || step == last_step_)
        return;
    write_(ctx, step);
}

void XyzRecorder::final_output(const cpu::LaunchContext& ctx) {
    // LAMMPS Min::run: a minimization that stops early writes its final
    // state at the final timestep, unless already written (`last_dump`).
    if (ctx.step == last_step_)
        return;
    write_(ctx, ctx.step);
}

void XyzRecorder::write_(const cpu::LaunchContext& ctx, BigInt step) {
    // Every rank records the step before the rank gate (they all take part
    // in the gather), so the never-twice rule stays rank-identical.
    last_step_ = step;

    // Publish the step through the Frame's typed metadata channel, so it
    // travels with the geometry into the XYZ comment line.
    ctx.frame.meta.get(kStep) = static_cast<std::int64_t>(step);

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
