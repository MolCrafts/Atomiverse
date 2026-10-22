/// @file tests/cpu/test_analysis_executor.cpp
/// @brief Tests for `cpu::AnalysisExecutor` lifecycle and the single
///        (drop_newest) backpressure path.
///
/// Spec: .claude/specs/async-analysis-snapshot-ring-01-uniformity.md.
/// Verifies:
///   - clean init/shutdown (no thread leak)
///   - round-robin slot IDs + drop_newest counter
///   - drop_newest is the sole backpressure behaviour under load
///   - ring depth K is a constant global (capacity never changes after init)
///   - shutdown drains and joins cleanly

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <thread>
#include <vector>

#include "atomiverse/cpu/asynchronous/analysis_executor.h"
#include "atomiverse/cpu/asynchronous/async_component_base.h"
#include "atomiverse/cpu/asynchronous/frame_snapshot.h"
#include "atomiverse/cpu/asynchronous/snapshot_publisher.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// @brief Test-only async component used to measure submission behaviour.
class SlowCapture : public cpu::AsyncComponentBase {
  public:
    explicit SlowCapture(
        std::chrono::milliseconds delay = std::chrono::milliseconds(50))
        : delay_(delay) {}

    std::string name() const override { return "slow_capture"; }
    cpu::CaptureMask capture_mask() const override {
        cpu::CaptureMask m;
        m.positions = true;
        return m;
    }

    void on_snapshot(const cpu::FrameSnapshot& /*snap*/) override {
        std::this_thread::sleep_for(delay_);
        ++n_handled_;
    }

    int handled() const { return n_handled_.load(std::memory_order_relaxed); }

  private:
    std::chrono::milliseconds delay_;
    std::atomic<int> n_handled_{0};
};

} // namespace

// ── clean init/shutdown ──────────────────────────────────────────────

TEST(AnalysisExecutor, InitShutdownClean) {
    cpu::AnalysisExecutor exec;
    cpu::Context ctx;

    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 2;
    cfg.ring_size = 3;

    exec.init(cfg, ctx);
    // No implicit leak: shutdown must join all workers.
    exec.shutdown();
    SUCCEED();
}

// ── round-robin slot IDs + drop counter ──────────────────────────────

TEST(AnalysisExecutor, AcquireSlotRoundRobin) {
    cpu::AnalysisExecutor exec;
    cpu::Context ctx;
    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 0; // no workers — just test the ring mechanics
    cfg.ring_size = 3;
    exec.init(cfg, ctx);

    int s0 = exec.acquire_slot();
    int s1 = exec.acquire_slot();
    int s2 = exec.acquire_slot();
    EXPECT_EQ(s0, 0);
    EXPECT_EQ(s1, 1);
    EXPECT_EQ(s2, 2);

    // 4th acquire with drop_newest wraps and bumps the drop counter.
    int s3 = exec.acquire_slot();
    EXPECT_EQ(s3, -1);
    EXPECT_EQ(exec.n_dropped(), 1);

    exec.shutdown();
}

// ── drop_newest backpressure (the sole policy) ───────────────────────

TEST(AnalysisExecutor, DropNewestBackpressure) {
    cpu::AnalysisExecutor exec;
    cpu::Context ctx;
    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 1;
    cfg.ring_size = 3;
    exec.init(cfg, ctx);

    SlowCapture sub{std::chrono::milliseconds(100)};
    for (int k = 0; k < 5; ++k) {
        int slot = exec.acquire_slot();
        if (slot < 0)
            continue;
        exec.publish(slot, k);
        exec.submit(&sub, slot);
    }

    // With ring_size=3 and 5 submissions, 2 overwrites must be counted.
    EXPECT_EQ(exec.n_dropped(), 2);

    exec.shutdown();
}

// ── ring depth K is a constant global ────────────────────────────────

TEST(AnalysisExecutor, RingDepthIsConstantGlobalK) {
    cpu::AnalysisExecutor exec;
    cpu::Context ctx;
    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 1;
    cfg.ring_size = 4; // global ring depth K
    exec.init(cfg, ctx);

    EXPECT_EQ(exec.ring_capacity(), 4);

    // Saturate well past K with a slow worker: capacity must NOT grow — the
    // single drop_newest policy overwrites in place at constant depth K.
    SlowCapture sub{std::chrono::milliseconds(40)};
    for (int k = 0; k < 12; ++k) {
        int slot = exec.acquire_slot();
        if (slot < 0)
            continue;
        exec.publish(slot, k);
        exec.submit(&sub, slot);
    }
    EXPECT_EQ(exec.ring_capacity(), 4)
        << "ring depth K must stay constant; no dynamic growth path";
    EXPECT_GT(exec.n_dropped(), 0);

    exec.shutdown();
}

// ── shutdown drains in-flight jobs ───────────────────────────────────

TEST(AnalysisExecutor, ShutdownWithInFlightJobs) {
    cpu::AnalysisExecutor exec;
    cpu::Context ctx;
    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 2;
    cfg.ring_size = 3;
    exec.init(cfg, ctx);

    SlowCapture sub{std::chrono::milliseconds(30)};
    for (int k = 0; k < 3; ++k) {
        int slot = exec.acquire_slot();
        if (slot < 0)
            continue;
        exec.publish(slot, k);
        exec.submit(&sub, slot);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    exec.shutdown();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    // Must join within a generous deadline; no hang.
    EXPECT_LT(elapsed_ms, 2000);
}

namespace {
class HeldCapture : public cpu::AsyncComponentBase {
  public:
    std::promise<void> started, release, finished;
    double before = 0, after = 0;
    std::string name() const override { return "held_capture"; }
    cpu::CaptureMask capture_mask() const override { return {}; }
    void on_snapshot(const cpu::FrameSnapshot& snapshot) override {
        before = snapshot.x.at(0);
        started.set_value();
        release.get_future().wait();
        after = snapshot.x.at(0);
        finished.set_value();
    }
};
} // namespace

TEST(AnalysisExecutor, AllSubscribersRetainSnapshotUntilTheirCallbacksFinish) {
    cpu::Context ctx;
    cpu::AnalysisExecutor exec;
    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 2;
    cfg.ring_size = 1;
    exec.init(cfg, ctx);
    HeldCapture first, second;
    const int slot = exec.acquire_slot();
    exec.slot(slot).x = {42.0};
    exec.publish(slot, 1);
    exec.submit(&first, slot);
    exec.submit(&second, slot);
    first.started.get_future().wait();
    second.started.get_future().wait();
    first.release.set_value();
    first.finished.get_future().wait();
    // The first callback completing cannot release the second callback's data.
    EXPECT_EQ(exec.acquire_slot(), -1);
    second.release.set_value();
    exec.quiesce();
    EXPECT_DOUBLE_EQ(first.after, 42.0);
    EXPECT_DOUBLE_EQ(second.after, 42.0);
    EXPECT_EQ(exec.acquire_slot(), slot);
    exec.shutdown();
}

TEST(AnalysisExecutor, PublisherResizesExclusiveSlotAfterPopulationGrowth) {
    cpu::Context ctx;
    cpu::AnalysisExecutor exec;
    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 0;
    cfg.ring_size = 1;
    exec.init(cfg, ctx);
    cpu::Frame frame(1);
    SlowCapture observer;
    observer.build({ctx, frame});
    cpu::Publish publisher;
    publisher.build({ctx, frame});
    publisher.launch({ctx, frame, 0, 1});
    frame.grow(8);
    frame.set_nlocal(8);
    for (int i = 0; i < 8; ++i)
        frame.span(kAtom_x)[i] = i + 0.5;
    publisher.launch({ctx, frame, 0, 2});
    const auto& snapshot = exec.slot(exec.last_published_slot());
    EXPECT_EQ(snapshot.n_owned, 8);
    ASSERT_EQ(snapshot.x.size(), 8u);
    EXPECT_DOUBLE_EQ(snapshot.x[7], 7.5);
    observer.destroy({ctx, frame});
    exec.shutdown();
}

TEST(AnalysisExecutor, OneSubscriberReceivesSnapshotsInOrderAcrossWorkers) {
    struct OrderedCapture : cpu::AsyncComponentBase {
        std::promise<void> entered, release;
        std::atomic<int> calls{0};
        std::vector<int> steps;
        std::string name() const override { return "ordered_capture"; }
        cpu::CaptureMask capture_mask() const override { return {}; }
        void on_snapshot(const cpu::FrameSnapshot& snap) override {
            if (calls.fetch_add(1) == 0) {
                entered.set_value();
                release.get_future().wait();
            }
            steps.push_back(snap.step);
        }
    } ordered;
    HeldCapture other;
    cpu::Context ctx;
    cpu::AnalysisExecutor exec;
    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 2;
    cfg.ring_size = 2;
    exec.init(cfg, ctx);
    const int first = exec.acquire_slot();
    exec.slot(first).x = {1};
    exec.publish(first, 1);
    exec.submit(&ordered, first);
    ordered.entered.get_future().wait();
    const int second = exec.acquire_slot();
    exec.slot(second).x = {2};
    exec.publish(second, 2);
    exec.submit(&ordered, second);
    exec.submit(&other, second);
    other.started.get_future().wait();
    // Worker two can serve another observer, but not race ordered's history.
    EXPECT_EQ(ordered.calls.load(), 1);
    other.release.set_value();
    ordered.release.set_value();
    exec.quiesce();
    EXPECT_EQ(ordered.steps, (std::vector<int>{1, 2}));
    exec.shutdown();
}

namespace {
class ThrowingCapture : public cpu::AsyncComponentBase {
  public:
    int finalized = 0;
    std::string name() const override { return "throwing_capture"; }
    cpu::CaptureMask capture_mask() const override { return {}; }
    void on_snapshot(const cpu::FrameSnapshot&) override {
        throw std::runtime_error("snapshot callback failed");
    }
    void finalize() override { ++finalized; }
};
} // namespace

TEST(AnalysisExecutor, CallbackExceptionReachesDrainAndReleasesReader) {
    cpu::Context ctx;
    cpu::AnalysisExecutor exec;
    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 1;
    cfg.ring_size = 1;
    exec.init(cfg, ctx);
    ThrowingCapture sub;
    const int slot = exec.acquire_slot();
    exec.publish(slot, 1);
    exec.submit(&sub, slot);
    EXPECT_THROW(exec.quiesce(), std::runtime_error);
    EXPECT_NO_THROW(exec.quiesce());
    EXPECT_EQ(exec.acquire_slot(), slot);
    EXPECT_NO_THROW(exec.shutdown());
}

TEST(AnalysisExecutor, SubscriberDestructionPreservesErrorForExplicitDrain) {
    cpu::Context ctx;
    cpu::AnalysisExecutor exec;
    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 1;
    exec.init(cfg, ctx);
    cpu::Frame frame(1);
    ThrowingCapture sub;
    sub.build({ctx, frame});
    const int slot = exec.acquire_slot();
    exec.publish(slot, 1);
    exec.submit(&sub, slot);
    EXPECT_NO_THROW(sub.destroy({ctx, frame}));
    EXPECT_EQ(sub.finalized, 1);
    EXPECT_TRUE(exec.subscribers().empty());
    EXPECT_THROW(exec.quiesce(), std::runtime_error);
    EXPECT_NO_THROW(exec.shutdown());
    EXPECT_EQ(sub.finalized, 1);
}

TEST(AnalysisExecutor, AbandonedCaptureReleasesItsReservation) {
    cpu::Context ctx;
    cpu::AnalysisExecutor exec;
    cpu::AnalysisExecutor::Config cfg;
    cfg.n_threads = 0;
    cfg.ring_size = 1;
    exec.init(cfg, ctx);
    const int slot = exec.acquire_slot();
    exec.abandon_slot(slot);
    EXPECT_EQ(exec.acquire_slot(), slot);
    exec.shutdown();
}
