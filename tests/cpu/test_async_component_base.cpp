/// @file tests/cpu/test_async_component_base.cpp
/// @brief RED tests for `cpu::AsyncComponentBase` delivery semantics.
///
/// Spec: .claude/specs/parallel-md-engine.md § B.4.
/// Guarantees tested:
///   - on_snapshot() runs on an analysis thread, not the main thread
///   - frequency() is respected (step > 0 && step % freq == 0 firing rule)
///   - finalize() called exactly once at end of run / shutdown
///   - (MPI builds) analysis sub-communicator usable inside on_snapshot()

#include <gtest/gtest.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/async/analysis_executor.h"
#include "atomiverse/cpu/async/async_component_base.h"
#include "atomiverse/cpu/async/async_observer.h"
#include "atomiverse/cpu/async/frame_snapshot.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"

#include "cpu/driver_test_helpers.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

using namespace ATV;
using atv_test::NoOpForce;

namespace {

cpu::Frame make_tiny(int n) {
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_x)[i] = 0.1 * i;
        frame.span(ATV::kAtom_mass)[i] = 1.0;
    }
    frame.simbox = ATV::cpu::Box::cubic(10.0);
    return frame;
}

/// @brief Records caller thread and fire counts for analysis-thread assertions.
///
/// Uses `cpu::md::AsyncObserver` so `sim.add_ref(rec)` picks up the MD
/// default binding (`md::Phase::end_of_step @ +20`).
class ThreadRecorder : public cpu::md::AsyncObserver {
  public:
    std::string name() const override { return "thread_recorder"; }
    cpu::CaptureMask capture_mask() const override {
        cpu::CaptureMask m;
        m.positions = true;
        return m;
    }
    int frequency() const override { return freq_; }
    void set_frequency(int f) { freq_ = f; }

    void on_snapshot(const cpu::FrameSnapshot& /*snap*/) override {
        std::lock_guard<std::mutex> g(mtx_);
        thread_id_ = std::this_thread::get_id();
        ++fire_count_;
    }
    void finalize() override { ++finalize_count_; }

    std::thread::id captured_thread_id() const {
        std::lock_guard<std::mutex> g(mtx_);
        return thread_id_;
    }
    int fire_count() const {
        return fire_count_.load(std::memory_order_relaxed);
    }
    int finalize_count() const {
        return finalize_count_.load(std::memory_order_relaxed);
    }

  private:
    mutable std::mutex mtx_;
    std::thread::id thread_id_{};
    std::atomic<int> fire_count_{0};
    std::atomic<int> finalize_count_{0};
    int freq_ = 0;
};

} // namespace

// ── on_snapshot runs off the main thread ─────────────────────────────

TEST(AsyncComponentBase, OnSnapshotRunsOnAnalysisThread) {
    auto frame = make_tiny(8);
    std::thread::id main_id = std::this_thread::get_id();

    // `rec` MUST outlive `sim`: Driver::add_ref is explicitly non-owning.
    ThreadRecorder rec;

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{1e-6});
    sim.add(NoOpForce{});
    sim.add_ref(rec);
    sim.build();
    sim.run(frame, 5); // step >= 1 needed for publisher to fire

    EXPECT_GE(rec.fire_count(), 1);
    EXPECT_NE(rec.captured_thread_id(), main_id)
        << "on_snapshot() ran on the main thread — analysis executor not "
           "firing async";
}

// ── frequency respected ──────────────────────────────────────────────

TEST(AsyncComponentBase, FrequencyRespected) {
    auto frame = make_tiny(4);

    ThreadRecorder rec;
    rec.set_frequency(5);

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{1e-6});
    sim.add(NoOpForce{});
    sim.add_ref(rec);
    sim.build();
    sim.run(frame, 20);

    // Spec: fires on step > 0 && step % freq == 0 — steps 5, 10, 15.
    EXPECT_EQ(rec.fire_count(), 3)
        << "frequency=5 over 20 steps should fire at steps 5/10/15";
}

// ── finalize called once ─────────────────────────────────────────────

TEST(AsyncComponentBase, FinalizeCalledOnce) {
    auto frame = make_tiny(4);

    ThreadRecorder rec;
    {
        cpu::md::MD sim;
        sim.add(cpu::md::VelocityVerlet{1e-6});
        sim.add(NoOpForce{});
        sim.add_ref(rec);
        sim.build();
        sim.run(frame, 5);
    } // sim shutdown drains executor and calls finalize() once

    EXPECT_EQ(rec.finalize_count(), 1);
}

// ── MPI sub-communicator usable inside on_snapshot (MPI builds only) ─

#ifdef ATV_HAS_MPI
namespace {
class MpiAllreduceSub : public cpu::md::AsyncObserver {
  public:
    std::string name() const override { return "mpi_allreduce_sub"; }
    cpu::CaptureMask capture_mask() const override {
        cpu::CaptureMask m;
        m.positions = true;
        return m;
    }
    void on_snapshot(const cpu::FrameSnapshot& /*snap*/) override {
        double local = 1.0;
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM,
                      this->executor()->analysis_comm());
        int size = 0;
        MPI_Comm_size(this->executor()->analysis_comm(), &size);
        last_global_ = global;
        last_size_ = size;
    }
    double last_global() const { return last_global_.load(); }
    int last_size() const { return last_size_.load(); }

  private:
    std::atomic<double> last_global_{0.0};
    std::atomic<int> last_size_{0};
};
} // namespace

TEST(AsyncComponentBase, MpiAllreduceInOnSnapshotWorks) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        GTEST_SKIP()
            << "Test process was not launched under MPI; skipping "
               "MPI sub-communicator collective check. Re-run with "
               "`mpirun -n N ./test_cpu_async_component_base` to exercise.";
    }

    auto frame = make_tiny(4);

    MpiAllreduceSub sub;

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{1e-6});
    sim.add(NoOpForce{});
    sim.add_ref(sub);
    sim.build();
    sim.run(frame, 5);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    EXPECT_EQ(sub.last_size(), world_size);
    EXPECT_NEAR(sub.last_global(), static_cast<double>(world_size), 1e-12);
}
#endif // ATV_HAS_MPI
