/// @file cpu/minimize/final_geometry.cpp
/// @brief FinalGeometry -- write the final geometry once, at end of run.
///        MPI-aware: collective owner gather, rank-0 I/O.

#include "atomiverse/cpu/minimize/final_geometry.h"

#include "atomiverse/cpu/communication/gather_for_io.h"
#include "atomiverse/cpu/record/xyz.h"

namespace ATV {
namespace cpu {
namespace minimizer {

FinalGeometry::FinalGeometry(std::string filename)
    : filename_(std::move(filename)) {}

void FinalGeometry::launch(const cpu::LaunchContext& /*ctx*/) {
    // Record-only: the contract is "final geometry after run()", so the
    // gather + write happens exactly once in destroy() (end-of-run hook)
    // instead of re-gathering and rewriting the file every end_of_step.
    write_pending_ = true;
}

void FinalGeometry::destroy(const cpu::DestroyContext& ctx) {
    if (!write_pending_)
        return;
    write_pending_ = false;

    // Multi-rank: collective owner gather (every rank enters, in lockstep
    // because write_pending_ is set identically on all ranks); returns
    // nullptr on non-root ranks.
    cpu::Frame gathered{0};
    const cpu::Frame* src =
        md::mpi::gather_for_root_io(ctx.frame, ctx.runtime, gathered);
    if (!src)
        return;

    // XYZ::write truncates any existing file: the result is a single-frame
    // snapshot of the final geometry.
    XYZ writer(filename_);
    writer.write(*src);
    ++n_writes_;
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
