/// @file cpu/asynchronous/async_component_base.cpp
/// @brief `cpu::AsyncComponentBase` implementation -- thin dispatcher into the
/// snapshot ring slot most recently stamped by `Publish`.

#include "atomiverse/cpu/asynchronous/async_component_base.h"

#include "atomiverse/cpu/asynchronous/analysis_executor.h"

namespace ATV {
namespace cpu {

void AsyncComponentBase::build(const BuildContext& ctx) {
    exec_ = ctx.runtime.analysis.executor;
    if (!exec_)
        return;
    exec_->register_subscriber(this);
    // Pre-size ring slot buffers for this subscriber's mask so
    // `Publish::launch()` never reallocates at fire time.
    // Multiple calls are idempotent (monotonic).
    exec_->prealloc_slots(ctx.frame.nlocal(), capture_mask());
}

void AsyncComponentBase::launch(const LaunchContext& ctx) {
    if (!exec_)
        return;

    const int freq = frequency();
    if (freq > 0) {
        if (ctx.step == 0 || ctx.step % freq != 0)
            return;
    } else {
        // freq == 0: every step after the publisher has fired (step >= 1).
        if (ctx.step == 0)
            return;
    }

    // The snapshot filled at this end-of-step was stamped by
    // `Publish::launch()` at order +10; we run at +20 and pick up
    // `last_published_slot()` in the same step. If the step tag on the slot
    // does not match `ctx.step`, the publisher did not fire (e.g. no firing
    // subscribers under any mask) and we silently skip.
    const int slot_idx = exec_->last_published_slot();
    if (slot_idx < 0)
        return;

    const FrameSnapshot& cand = exec_->slot(slot_idx);
    if (cand.step != ctx.step)
        return;

    exec_->submit(this, slot_idx);
}

void AsyncComponentBase::destroy(const DestroyContext& /*ctx*/) {
    // finalize() is called by AnalysisExecutor::shutdown() after workers join.
}

} // namespace cpu
} // namespace ATV
