/// @file src/cpu/snapshot_publisher.cpp
/// @brief `cpu::Publish` implementation: collects subscriber
/// `CaptureMask`s, fills a ring slot. Each `AsyncComponentBase` dispatches
/// itself at `end_of_step @ +20` against the slot most recently published.

#include "atomiverse/cpu/async/snapshot_publisher.h"

#include "atomiverse/cpu/async/analysis_executor.h"
#include "atomiverse/cpu/async/async_component_base.h"

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

namespace {

bool subscriber_fires(const AsyncComponentBase* sub, int step) {
    const int f = sub->frequency();
    if (f <= 0)
        return step > 0; // freq 0 = every step (after step 0)
    return step > 0 && (step % f) == 0;
}

CaptureMask mask_union_all(const std::vector<AsyncComponentBase*>& subs) {
    CaptureMask u;
    u.positions = u.velocities = u.forces = u.types = u.tags = false;
    for (const auto* sub : subs) {
        if (!sub)
            continue;
        const CaptureMask m = sub->capture_mask();
        u.positions = u.positions || m.positions;
        u.velocities = u.velocities || m.velocities;
        u.forces = u.forces || m.forces;
        u.types = u.types || m.types;
        u.tags = u.tags || m.tags;
    }
    return u;
}

// Copy src[0..n) into out[0..n); vectors already sized by prealloc_slots().
template <class T>
void copy_range(std::vector<double>& out, const T* src, int n, bool capture) {
    if (!capture)
        return; // leave pre-sized buffer untouched (consumer reads n_owned)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)] = src[i]; // implicit widen to double
}

void copy_range_int(std::vector<int>& out, const int* src, int n,
                    bool capture) {
    if (!capture)
        return;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)] = src[i];
}

template <class T>
void copy_range_i64(std::vector<std::int64_t>& out, const T* src, int n,
                    bool capture) {
    if (!capture)
        return;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)] = src[i]; // widen tag tier to int64
}

} // namespace

void Publish::build(const BuildContext& ctx) {
    exec_ = ctx.runtime.analysis.executor;
    // Pre-sizing per mask happens inside `AsyncComponentBase::build()` for
    // each registered subscriber -- the publisher runs first in plan order
    // (lower order = earlier build), so `subscribers()` is still empty here.
    (void)ctx;
}

void Publish::launch(const LaunchContext& ctx) {
    if (!exec_)
        return;

    const auto& subs = exec_->subscribers();
    if (subs.empty())
        return; // zero-subscriber fast path (≤ 100 ns)

    const int step = ctx.step;

    bool any_firing = false;
    for (const auto* s : subs) {
        if (s && subscriber_fires(s, step)) {
            any_firing = true;
            break;
        }
    }
    if (!any_firing)
        return;

    // Static capture-mask union over ALL subscribers, resolved once and
    // reused -- the captured field set is uniform for every captured step and
    // does not depend on which subscribers fire. Subscriber set is fixed after
    // build(); compute lazily on the first capturing launch (the publisher at
    // +10 builds before subscribers at +20, so the list is empty in build()).
    if (!mask_ready_) {
        static_mask_ = mask_union_all(subs);
        mask_ready_ = true;
    }
    const CaptureMask mask = static_mask_;

    const int slot_idx = exec_->acquire_slot();
    FrameSnapshot& snap = exec_->slot(slot_idx);

    const int n = ctx.frame.nlocal();
    snap.step = step;
    snap.n_owned = n;
    snap.box = ctx.frame.simbox;

    copy_range(snap.x, ctx.frame.span(kAtom_x).data(), n, mask.positions);
    copy_range(snap.y, ctx.frame.span(kAtom_y).data(), n, mask.positions);
    copy_range(snap.z, ctx.frame.span(kAtom_z).data(), n, mask.positions);
    copy_range(snap.vx, ctx.frame.span(kAtom_vx).data(), n, mask.velocities);
    copy_range(snap.vy, ctx.frame.span(kAtom_vy).data(), n, mask.velocities);
    copy_range(snap.vz, ctx.frame.span(kAtom_vz).data(), n, mask.velocities);
    copy_range(snap.fx, ctx.frame.span(kAtom_fx).data(), n, mask.forces);
    copy_range(snap.fy, ctx.frame.span(kAtom_fy).data(), n, mask.forces);
    copy_range(snap.fz, ctx.frame.span(kAtom_fz).data(), n, mask.forces);
    copy_range_int(snap.type, ctx.frame.span(kAtom_type).data(), n, mask.types);
    copy_range_i64(snap.tag, ctx.frame.span(kAtom_tag).data(), n, mask.tags);

    exec_->publish(slot_idx, step);
    // Dispatch happens per-subscriber inside `AsyncComponentBase::launch()`
    // at end_of_step @ +20.
}

void Publish::destroy(const DestroyContext& /*ctx*/) {}

} // namespace cpu
} // namespace ATV
