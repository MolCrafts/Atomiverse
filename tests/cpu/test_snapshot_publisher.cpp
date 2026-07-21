/// @file tests/cpu/test_snapshot_publisher.cpp
/// @brief RED tests for `cpu::Publish` zero-subscriber fast-path
///        and field-copy correctness.
///
/// Spec: .claude/specs/async-analysis-snapshot-ring-01-uniformity.md.
/// Required:
///   - zero subscribers → mean launch() cost < 100 ns (10 000 iters)
///   - positions, box, step propagate into FrameSnapshot
///   - CaptureMask respected (excluded fields remain empty/unmodified)

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <string>
#include <thread>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/async/async_component_base.h"
#include "atomiverse/cpu/async/async_observer.h"
#include "atomiverse/cpu/async/frame_snapshot.h"
#include "atomiverse/cpu/async/snapshot_publisher.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/driver/backend.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"

#include "cpu/driver_test_helpers.h"

using namespace ATV;
using atv_test::NoOpForce;

namespace {

cpu::Frame make_small_frame(int n) {
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_x)[i] = 1.0 + 0.1 * i;
        frame.span(ATV::kAtom_y)[i] = 2.0 + 0.1 * i;
        frame.span(ATV::kAtom_z)[i] = 3.0 + 0.1 * i;
        frame.span(ATV::kAtom_vx)[i] = 0.01 * i;
        frame.span(ATV::kAtom_vy)[i] = 0.02 * i;
        frame.span(ATV::kAtom_vz)[i] = 0.03 * i;
        frame.span(ATV::kAtom_mass)[i] = 1.0;
    }
    frame.simbox = ATV::cpu::Box::cubic(10.0);
    return frame;
}

/// @brief Capture target — records the last snapshot received.
///
/// Inherits `cpu::md::AsyncObserver` to get the MD-flavoured default
/// binding (`md::Phase::end_of_step @ +20`), so `sim.add_ref(cap)` works
/// without explicit `on()` wiring.
class PositionCapture : public cpu::md::AsyncObserver {
  public:
    std::string name() const override { return "position_capture"; }
    cpu::CaptureMask capture_mask() const override {
        cpu::CaptureMask m;
        m.positions = true;
        m.velocities = false;
        return m;
    }
    void on_snapshot(const cpu::FrameSnapshot& snap) override {
        std::lock_guard<std::mutex> g(mtx_);
        last_ = snap; // copy
        ++n_handled_;
    }
    int handled() const { return n_handled_.load(std::memory_order_relaxed); }
    cpu::FrameSnapshot copy_last() const {
        std::lock_guard<std::mutex> g(mtx_);
        return last_;
    }

  private:
    mutable std::mutex mtx_;
    cpu::FrameSnapshot last_;
    std::atomic<int> n_handled_{0};
};

/// @brief Integrator that stamps positions with an analytic trajectory
/// `x_i(step) = tag_i + step` at `initial_integrate`, so a torn or
/// mis-stamped snapshot is detectable per-atom on the consumer side.
class StepStamper : public cpu::Integrator {
  public:
    std::string name() const override { return "step_stamper"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0)};
    }
    void launch(const cpu::LaunchContext& ctx) override {
        auto x = ctx.frame.span(kAtom_x);
        const auto tag = ctx.frame.span(kAtom_tag);
        const int n = ctx.frame.nlocal();
        for (int i = 0; i < n; ++i)
            x[i] = static_cast<double>(tag[i]) + static_cast<double>(ctx.step);
    }
};

/// @brief Slow async observer that verifies every delivered snapshot is a
/// single-step, correctly-stamped capture: `x_i == tag_i + step` for all i.
class ConsistencyCheck : public cpu::md::AsyncObserver {
  public:
    std::string name() const override { return "consistency_check"; }
    cpu::CaptureMask capture_mask() const override {
        cpu::CaptureMask m;
        m.positions = true;
        m.tags = true;
        return m;
    }
    void on_snapshot(const cpu::FrameSnapshot& snap) override {
        // Artificially lag the worker so the hot loop races far ahead and
        // drop_oldest discards frames; any DELIVERED frame must still be
        // internally consistent.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        bool ok = snap.n_owned > 0;
        for (int i = 0; i < snap.n_owned; ++i) {
            const double expected = static_cast<double>(snap.tag[i])
                                    + static_cast<double>(snap.step);
            if (std::abs(snap.x[i] - expected) > 1e-9) {
                ok = false;
                break;
            }
        }
        if (!ok)
            consistent_.store(false, std::memory_order_relaxed);
        ++n_handled_;
    }
    bool consistent() const {
        return consistent_.load(std::memory_order_relaxed);
    }
    int handled() const { return n_handled_.load(std::memory_order_relaxed); }

  private:
    std::atomic<bool> consistent_{true};
    std::atomic<int> n_handled_{0};
};

/// @brief Velocity-capturing observer with a high frequency so it does NOT
/// fire during a short run — used to prove the capture mask is the static
/// union of ALL subscribers, not just the ones firing this step.
class VelocityLowFreq : public cpu::md::AsyncObserver {
  public:
    std::string name() const override { return "velocity_low_freq"; }
    int frequency() const override { return 1000; }
    cpu::CaptureMask capture_mask() const override {
        cpu::CaptureMask m;
        m.positions = false;
        m.velocities = true;
        return m;
    }
    void on_snapshot(const cpu::FrameSnapshot& /*snap*/) override {}
};

} // namespace

// ── zero-subscriber fast path ────────────────────────────────────────

TEST(Publish, ZeroSubscribersEarlyReturn) {
    auto frame = make_small_frame(100);
    cpu::Context ctx;
    cpu::Publish pub;
    cpu::BuildContext bctx{ctx, frame};
    pub.build(bctx);

    constexpr int iters = 10'000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < iters; ++k) {
        cpu::LaunchContext lctx{ctx, frame, 0, k};
        pub.launch(lctx);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    auto mean_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
        / iters;
    EXPECT_LT(mean_ns, 100) << "zero-subscriber mean launch() cost " << mean_ns
                            << " ns exceeds 100 ns budget";
}

// ── positions propagate end-to-end ───────────────────────────────────

TEST(Publish, SnapshotCopiesPositions) {
    auto frame = make_small_frame(64);

    // `cap` MUST outlive `sim`: sim only stores a non-owning pointer (lvalue
    // overload of Driver::add). Inspecting cap after the run requires it
    // to still be alive in the same scope.
    PositionCapture cap;

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{1e-6});
    sim.add(NoOpForce{});
    sim.add_ref(cap);
    sim.build();
    sim.run(frame,
            5); // give the publisher a step where async fires (step >= 1)

    const int n = frame.nlocal();
    ASSERT_GE(cap.handled(), 1) << "no snapshot was delivered";
    auto snap = cap.copy_last();
    ASSERT_EQ(static_cast<int>(snap.x.size()), n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(snap.x[i], frame.span(ATV::kAtom_x)[i], 1e-12);
        EXPECT_NEAR(snap.y[i], frame.span(ATV::kAtom_y)[i], 1e-12);
        EXPECT_NEAR(snap.z[i], frame.span(ATV::kAtom_z)[i], 1e-12);
    }
}

// ── box + step propagate ─────────────────────────────────────────────

TEST(Publish, SnapshotCopiesBoxAndStep) {
    auto frame = make_small_frame(16);

    PositionCapture cap;

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{1e-6});
    sim.add(NoOpForce{});
    sim.add_ref(cap);
    sim.build();
    sim.run(frame, 5);

    ASSERT_GE(cap.handled(), 1);
    auto snap = cap.copy_last();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(snap.box.cell()(i, j), frame.simbox.cell()(i, j),
                        1e-12);
    EXPECT_GE(snap.step, 1)
        << "publisher fires after step 0; first capture must be step >= 1";
}

// ── CaptureMask respected ────────────────────────────────────────────

TEST(Publish, CaptureMaskRespected) {
    auto frame = make_small_frame(32);

    PositionCapture cap;

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{1e-6});
    sim.add(NoOpForce{});
    sim.add_ref(cap);
    sim.build();
    sim.run(frame, 5);

    ASSERT_GE(cap.handled(), 1);
    auto snap = cap.copy_last();
    EXPECT_EQ(static_cast<int>(snap.x.size()), frame.nlocal());
    EXPECT_EQ(static_cast<std::size_t>(snap.vx.size()),
              static_cast<std::size_t>(0))
        << "velocities must remain empty when CaptureMask.velocities=false";
    EXPECT_EQ(static_cast<std::size_t>(snap.vy.size()),
              static_cast<std::size_t>(0));
    EXPECT_EQ(static_cast<std::size_t>(snap.vz.size()),
              static_cast<std::size_t>(0));
}

// ── static capture-mask union over ALL subscribers (RED) ─────────────
//
// Two subscribers: `cap` (positions, fires every step) and `vel`
// (velocities, frequency 1000 → does not fire in a 5-step run). The
// captured field set must be the static union of ALL subscribers, so the
// delivered snapshot carries `vel`'s velocities even on steps where only
// `cap` fires. A per-step (firing-only) union would leave velocities at
// their pre-sized zero state → RED before the static-mask change.

TEST(Publish, StaticMaskUnionAllSubscribers) {
    auto frame = make_small_frame(40);
    const int n = frame.nlocal();
    // Distinctive, all-nonzero velocities so "not copied" (zeros) is
    // distinguishable from "copied".
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_vx)[i] = 100.0 + i;
        frame.span(ATV::kAtom_vy)[i] = 200.0 + i;
        frame.span(ATV::kAtom_vz)[i] = 300.0 + i;
    }

    PositionCapture cap; // positions, every step, records snapshots
    VelocityLowFreq vel; // velocities, freq 1000 (won't fire in 5 steps)

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{1e-6});
    sim.add(NoOpForce{});
    sim.add_ref(cap);
    sim.add_ref(vel);
    sim.build();
    sim.run(frame, 5);

    ASSERT_GE(cap.handled(), 1) << "no snapshot delivered to cap";
    auto snap = cap.copy_last();
    ASSERT_EQ(static_cast<int>(snap.vx.size()), n)
        << "velocity buffer must be pre-sized for the velocity subscriber";
    // Velocities must be the actually-copied frame values, not the pre-sized
    // zeros — proving the static union included the non-firing subscriber.
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(snap.vx[i], 100.0 + i, 1e-9)
            << "static capture-mask union must copy a non-firing subscriber's "
               "fields (i="
            << i << ")";
    }
}

// ── freeze-then-dispatch consistency under a lagging worker (lock) ────
//
// Every-step capture with an artificially slow worker (drop_oldest discards
// most frames). The analytic trajectory x_i = tag_i + step makes any torn
// (cross-step) or mis-stamped delivered snapshot detectable per-atom. Locks
// the invariant the whole chain rests on; passes on CPU because Publish
// freezes synchronously on the hot-path thread before the next step mutates
// the frame.

TEST(Publish, FreezeDispatchConsistencyUnderLag) {
    auto frame = make_small_frame(48);
    const int n = frame.nlocal();
    for (int i = 0; i < n; ++i)
        frame.span(ATV::kAtom_tag)[i] = i + 1; // nonzero, distinct global tags

    ConsistencyCheck cap;

    cpu::md::MD sim;
    sim.add(StepStamper{}); // sets x_i = tag_i + step each step
    sim.add(NoOpForce{});
    sim.add_ref(cap);
    sim.build();
    sim.run(frame, 40); // hot loop races ahead of the 5 ms/frame worker

    ASSERT_GE(cap.handled(), 1) << "no snapshot was delivered";
    EXPECT_TRUE(cap.consistent())
        << "a delivered snapshot was torn or mis-stamped (x_i != tag_i + step)";
}
