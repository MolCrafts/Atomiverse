/// @file cpu/asynchronous/async_consumer.cpp
/// @brief cpu::AsyncConsumer: cadence + LAMMPS step labels on the stepping
///        thread, capture-once through the driver's CaptureBuffer, delivery
///        on the consumer's worker thread.

#include "atomiverse/cpu/asynchronous/async_consumer.h"

#include <exception>
#include <stdexcept>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/asynchronous/capture_buffer.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/recorder.h"

namespace ATV {
namespace cpu {

void AsyncConsumer::build(const BuildContext& ctx) {
    buffer_ = ctx.runtime.capture.buffer;
    if (!buffer_)
        throw std::logic_error(
            capture_name()
            + ": an asynchronous consumer needs its driver's capture buffer; "
              "build it through a driver (cpu::md::MD, cpu::minimizer::"
              "Minimize, …), not standalone");
    buffer_->subscribe(*this);
    last_step_ = -1;
}

void AsyncConsumer::capture_(const LaunchContext& ctx, BigInt step,
                             CaptureKind kind) {
    if (!buffer_)
        throw std::logic_error(capture_name() + ": capture before build()");
    buffer_->capture(*this, ctx.frame, step, kind);
    last_step_ = step;
}

void AsyncConsumer::launch(const LaunchContext& ctx) {
    // The Runtime's cadence gate passed (`frequency()` / `frequency_phase()`):
    // the end of the step with zero-based index ctx.step, labelled with the
    // completed-step count (LAMMPS `update->ntimestep`).
    const BigInt step = loop_output_step(ctx);
    if (step == last_step_)
        return;
    capture_(ctx, step, CaptureKind::step);
}

void AsyncConsumer::setup_output(const LaunchContext& ctx) {
    // LAMMPS src/output.cpp Output::setup, e8c4c278a8. Copyright (2003)
    // Sandia Corporation. GPL-2.0-only: the current state when its step is
    // on cadence, never a step twice (`last_dump`).
    const int every = frequency();
    if (every > 0 && ctx.step % every != 0)
        return;
    if (ctx.step == last_step_)
        return;
    capture_(ctx, ctx.step, CaptureKind::setup);
}

void AsyncConsumer::final_output(const LaunchContext& ctx) {
    // LAMMPS src/min.cpp Min::run, e8c4c278a8: a minimization stopped
    // before MAXITER outputs its final state, unless already written.
    if (ctx.step == last_step_)
        return;
    capture_(ctx, ctx.step, CaptureKind::final_state);
}

void AsyncConsumer::destroy(const DestroyContext& /*ctx*/) {
    if (!buffer_)
        return;
    // Every snapshot this consumer was due is consumed before it is
    // finalized; a consumer failure stays pending for the driver's drain.
    buffer_->unsubscribe(*this);
    buffer_ = nullptr;
    try {
        finalize();
    } catch (const std::exception& e) {
        ATV_ERROR("{}: finalize failed: {}", capture_name(), e.what());
        throw;
    }
}

void AsyncConsumer::consume_capture(const CaptureTicket& ticket) {
    on_snapshot(buffer_->view(ticket));
}

} // namespace cpu
} // namespace ATV
