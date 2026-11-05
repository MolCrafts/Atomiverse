/// @file cpu/minimize/trajectory_writer.cpp
/// @brief TrajectoryWriter -- molpy-backed XYZ append.
///        MPI-aware: collective owner gather, rank-0 I/O.

#include "atomiverse/cpu/minimize/trajectory_writer.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/gather_for_io.h"

namespace ATV {
namespace cpu {
namespace minimizer {

TrajectoryWriter::TrajectoryWriter(std::string filename, int stride)
    : filename_(std::move(filename)), stride_(stride < 1 ? 1 : stride),
      writer_(filename_) {}

// LAMMPS src/min.cpp Min::setup / Min::run, src/min_cg.cpp MinCG::iterate
// and src/output.cpp Output::setup / Output::write, e8c4c278a8. Copyright
// (2003) Sandia Corporation. GPL-2.0-only. Setup dumps the initial state as
// step 0; iteration s ends at `ntimestep = ++update->ntimestep` (completed
// iteration s + 1) and dumps when that is a multiple of `every`; a stop
// before MAXITER dumps the final state; `last_dump` writes no step twice.

void TrajectoryWriter::setup_output(const cpu::LaunchContext& ctx) {
    if ((ctx.step % stride_) != 0 || ctx.step == last_step_)
        return;
    write_(ctx, ctx.step);
}

void TrajectoryWriter::launch(const cpu::LaunchContext& ctx) {
    // Stride gate first: identical on every rank, so the collective below is
    // still entered in lockstep.
    const BigInt step = loop_output_step(ctx);
    if ((step % stride_) != 0 || step == last_step_)
        return;
    write_(ctx, step);
}

void TrajectoryWriter::final_output(const cpu::LaunchContext& ctx) {
    if (ctx.step == last_step_)
        return;
    write_(ctx, ctx.step);
}

void TrajectoryWriter::write_(const cpu::LaunchContext& ctx, BigInt step) {
    last_step_ = step; // every rank, before the rank gate

    // Publish iteration identity through the Frame's typed metadata channel.
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

} // namespace minimizer
} // namespace cpu
} // namespace ATV
