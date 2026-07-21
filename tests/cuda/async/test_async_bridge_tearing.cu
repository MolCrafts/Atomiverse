/// @file tests/cuda/async/test_async_bridge_tearing.cu
/// @brief Snapshot-consistency (no-tearing) regression for the AsyncBridge
///        device-ring freeze (async-analysis-snapshot-ring-02-device-ring).
///
/// Drives a cuda::md::MD loop in which an analytic integrator stamps every
/// atom's x with x_i = tag_i + step at end_of_step (just before the async
/// capture), and an AsyncBridge-derived observer captures positions+tags with a
/// deliberately slowed host consumer. With the compute-stream freeze, every
/// DELIVERED snapshot must satisfy x_i == tag_i + snap.step for ALL atoms —
/// i.e. no tearing (mixed step-N / step-N+1 atoms) and no mis-stamping. The
/// pre-fix live-pointer D2H raced the next step's stamp and could deliver torn
/// frames; this test locks the fixed behaviour.
///
/// Also asserts the freeze path adds zero `cuda::stream_synchronize` calls on
/// the hot path (the freeze uses D2D + events, never a wrapper sync).

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/async/async_bridge.cuh"
#include "atomiverse/cuda/async/record_base.cuh"
#include "atomiverse/cuda/async/sync_guard.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/md/init.cuh"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/md/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

using cuda::real_t;

/// @brief Stamp x_i = tag_i + step for every atom (analytic trajectory).
__global__ void stamp_positions_kernel(real_t* x, const std::int64_t* tag,
                                       int n, double step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = static_cast<real_t>(static_cast<double>(tag[i]) + step);
}

/// @brief Integrator-slot component that writes the analytic trajectory into
///        x at end_of_step @ +5 — after per-step reductions and before the
///        async observer freezes at +20.
class StampPositions : public cuda::Component {
  public:
    std::string name() const override { return "stamp_positions"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, 5)};
    }
    void launch(const cuda::LaunchContext& ctx) override {
        const int n = ctx.frame.n_atoms();
        if (n <= 0)
            return;
        int block = 128;
        int grid = (n + block - 1) / block;
        stamp_positions_kernel<<<
            grid, block, 0, static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
            ctx.frame.atoms.x, ctx.frame.atoms.tag, n,
            static_cast<double>(ctx.step));
    }
};

/// @brief AsyncBridge observer that captures positions + tags and verifies
///        each delivered snapshot is single-step and correctly stamped.
class ConsistencyBridge : public cuda::AsyncBridge {
  public:
    std::string name() const override { return "consistency_bridge"; }
    int frequency() const override { return 1; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, 20)};
    }
    bool consistent() const {
        return consistent_.load(std::memory_order_relaxed);
    }
    int handled() const { return handled_.load(std::memory_order_relaxed); }

  protected:
    void on_build(const cuda::BuildContext& ctx) override {
        n_ = ctx.frame.n_atoms();
        auto n = static_cast<size_t>(n_);
        for (int i = 0; i < kRingDepth; ++i) {
            cudaMalloc(&d_x_[i], n * sizeof(real_t));
            cudaMalloc(&d_tag_[i], n * sizeof(std::int64_t));
            cudaMallocHost(&h_x_[i], n * sizeof(real_t));
            cudaMallocHost(&h_tag_[i], n * sizeof(std::int64_t));
        }
    }

    void freeze_slot(const cuda::LaunchContext& ctx, int slot) override {
        auto ms = static_cast<cudaStream_t>(ctx.runtime.stream);
        auto n = static_cast<size_t>(n_);
        cudaMemcpyAsync(d_x_[slot], ctx.frame.atoms.x, n * sizeof(real_t),
                        cudaMemcpyDeviceToDevice, ms);
        cudaMemcpyAsync(d_tag_[slot], ctx.frame.atoms.tag,
                        n * sizeof(std::int64_t), cudaMemcpyDeviceToDevice, ms);
    }

    void d2h_slot(const cuda::LaunchContext& ctx, int slot) override {
        auto ds = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
        auto n = static_cast<size_t>(n_);
        cudaMemcpyAsync(h_x_[slot], d_x_[slot], n * sizeof(real_t),
                        cudaMemcpyDeviceToHost, ds);
        cudaMemcpyAsync(h_tag_[slot], d_tag_[slot], n * sizeof(std::int64_t),
                        cudaMemcpyDeviceToHost, ds);
    }

    void on_data_ready(const cuda::LaunchContext& ctx, int slot) override {
        // Lag the worker so the hot loop races ahead; any delivered frame must
        // still be internally consistent.
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        bool ok = n_ > 0;
        for (int i = 0; i < n_; ++i) {
            const double expected = static_cast<double>(h_tag_[slot][i])
                                    + static_cast<double>(ctx.step);
            if (std::abs(static_cast<double>(h_x_[slot][i]) - expected)
                > 0.25) {
                ok = false;
                break;
            }
        }
        if (!ok)
            consistent_.store(false, std::memory_order_relaxed);
        handled_.fetch_add(1, std::memory_order_relaxed);
    }

    void on_destroy(const cuda::DestroyContext&) override {
        for (int i = 0; i < kRingDepth; ++i) {
            if (d_x_[i]) {
                cudaFree(d_x_[i]);
                d_x_[i] = nullptr;
            }
            if (d_tag_[i]) {
                cudaFree(d_tag_[i]);
                d_tag_[i] = nullptr;
            }
            if (h_x_[i]) {
                cudaFreeHost(h_x_[i]);
                h_x_[i] = nullptr;
            }
            if (h_tag_[i]) {
                cudaFreeHost(h_tag_[i]);
                h_tag_[i] = nullptr;
            }
        }
    }

  private:
    int n_ = 0;
    real_t* d_x_[kRingDepth] = {};
    std::int64_t* d_tag_[kRingDepth] = {};
    real_t* h_x_[kRingDepth] = {};
    std::int64_t* h_tag_[kRingDepth] = {};
    std::atomic<bool> consistent_{true};
    std::atomic<int> handled_{0};
};

/// @brief Small LJ cube with distinct nonzero global tags.
cpu::Frame make_tagged_cube(int n_side = 4, double density = 0.8) {
    int n = n_side * n_side * n_side;
    double box_len = std::cbrt(double(n) / density);
    double spacing = box_len / double(n_side);
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.atoms.x[idx] = (double(ix) + 0.5) * spacing;
                frame.atoms.y[idx] = (double(iy) + 0.5) * spacing;
                frame.atoms.z[idx] = (double(iz) + 0.5) * spacing;
                frame.atoms.mass[idx] = 1.0;
                frame.atoms.type[idx] = 0;
                frame.atoms.tag[idx] = idx + 1; // distinct, nonzero
                ++idx;
            }
    frame.box = Mat3::cubic(static_cast<double>(box_len));
    return frame;
}

} // namespace

/// @brief Every delivered AsyncBridge snapshot is single-step and correctly
///        stamped under a lagging worker (no tearing, no mis-stamp).
TEST(CudaAsyncBridgeTearing, DeliveredSnapshotsAreSingleStep) {
    cpu::Frame frame = make_tagged_cube();
    ConsistencyBridge bridge;

    cuda::detail::reset_sync_count();
    {
        cuda::md::MD sim;
        sim.on(md::Phase::force)
            .add(cuda::NeighborList{2.5}, -10)
            .add(cuda::PairLJCut{1.0, 1.0, 2.5});
        sim.add(cuda::md::VelocityVerlet{0.001});
        sim.add(cuda::md::InitVelocity{1.0, 12345ULL});
        sim.add(StampPositions{});
        sim.add(bridge);
        sim.build();
        sim.run(frame, 60); // hot loop races ahead of the 3 ms/frame worker
        // sim destructor drains the defer stream (AsyncBridge::destroy), so
        // all callbacks have fired by the time we read the flags below.
    }

    EXPECT_GE(bridge.handled(), 1) << "no snapshot was delivered";
    EXPECT_TRUE(bridge.consistent())
        << "a delivered snapshot was torn or mis-stamped "
           "(x_i != tag_i + snap.step)";
    // The device-ring freeze uses D2D + events only — never the sync wrapper.
    EXPECT_EQ(cuda::detail::sync_count(), 0u)
        << "AsyncBridge device-ring freeze routed a cuda::stream_synchronize "
           "on the hot path";
}

// ── AsyncRecordBase device-ring: no in-flight overwrite ──────────────

namespace {

/// @brief Per-step scalar snapshot for the AsyncRecordBase ring test.
struct StepVal {
    double val = 0.0;
};

/// @brief Stamp the device-staged snapshot with the step index.
__global__ void set_stepval_kernel(StepVal* s, double v) { s->val = v; }

/// @brief AsyncRecordBase recorder that stages `val = step` each step and, with
///        a lagging worker, checks every delivered snapshot still carries its
///        own step's value — i.e. the device-ring never lets a later step
///        overwrite an in-flight slot (the single-buffer predecessor would).
class StepValRecorder : public cuda::AsyncRecordBase<StepVal> {
  public:
    std::string name() const override { return "stepval_recorder"; }
    int frequency() const override { return 1; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, 20)};
    }

    void launch(const cuda::LaunchContext& ctx) override {
        set_stepval_kernel<<<1, 1, 0,
                             static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
            device_snapshot_, static_cast<double>(ctx.step));
        std::atomic<bool>* ok = &ok_;
        std::atomic<int>* handled = &handled_;
        submit_async(static_cast<std::int64_t>(ctx.step),
                     [ok, handled](const StepVal& s, std::int64_t st) {
                         std::this_thread::sleep_for(
                             std::chrono::milliseconds(3));
                         if (std::abs(s.val - static_cast<double>(st)) > 0.25)
                             ok->store(false, std::memory_order_relaxed);
                         handled->fetch_add(1, std::memory_order_relaxed);
                     });
    }

    bool ok() const { return ok_.load(std::memory_order_relaxed); }
    int handled() const { return handled_.load(std::memory_order_relaxed); }

  private:
    std::atomic<bool> ok_{true};
    std::atomic<int> handled_{0};
};

} // namespace

/// @brief AsyncRecordBase device-ring delivers each step's own snapshot under a
///        lagging worker (no in-flight slot overwrite).
TEST(CudaAsyncRecordRing, NoInFlightOverwrite) {
    cpu::Frame frame = make_tagged_cube();
    StepValRecorder rec;

    {
        cuda::md::MD sim;
        sim.on(md::Phase::force)
            .add(cuda::NeighborList{2.5}, -10)
            .add(cuda::PairLJCut{1.0, 1.0, 2.5});
        sim.add(cuda::md::VelocityVerlet{0.001});
        sim.add(cuda::md::InitVelocity{1.0, 12345ULL});
        sim.add(rec);
        sim.build();
        sim.run(frame, 60);
    }

    EXPECT_GE(rec.handled(), 1) << "no snapshot was delivered";
    EXPECT_TRUE(rec.ok())
        << "a delivered AsyncRecordBase snapshot carried a different step's "
           "value — a later step overwrote an in-flight ring slot";
}
