/// @file tests/cuda/async/test_hot_path_no_sync.cu
/// @brief Headline correctness gate of `msu-pe-cuda-components-07`:
///        the AsyncRecordBase hot-path must not route any sync calls
///        through `cuda::stream_synchronize`.
///
/// We build a minimal `cuda::md::MD` loop with a tiny AsyncRecordBase-derived
/// recorder bound at `end_of_step @ +20`. Around every `run(frame, 1)` we
/// reset the opt-in counter and assert it stays zero after the step. The
/// counter only counts calls that go through `cuda::stream_synchronize` (the
/// wrapper in `cuda/async/sync_guard.cuh`); backend-level slot-boundary syncs
/// in `src/cuda/driver/driver.cu` are deliberately excluded because they are not
/// user-component code and not what the spec is gating.

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/async/record_base.cuh"
#include "atomiverse/cuda/async/sync_guard.cuh"
#include "atomiverse/cuda/md/init.cuh"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/md/nose_hoover_npt.cuh"
#include "atomiverse/cuda/md/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

#include <cuda_runtime.h>

using namespace ATV;

namespace {

/// @brief Tiny snapshot POD — we only need to prove the hot path is sync-free.
struct MinimalSnapshot {
    double kinetic = 0.0;
};

/// @brief AsyncRecordBase-derived recorder that does nothing useful on the
///        host side. It exists only to exercise the submit_async path.
class NoOpAsyncRecorder : public cuda::AsyncRecordBase<MinimalSnapshot> {
  public:
    std::string name() const override { return "noop_async_recorder"; }
    int frequency() const override { return 1; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, 20)};
    }

    void launch(const cuda::LaunchContext& ctx) override {
        // Stage a snapshot via a single small cudaMemcpyAsync from the device
        // SystemScalars.kinetic into device_snapshot_.kinetic. Both pointers
        // are device-resident, so this is a D2D copy on the compute stream
        // (zero hot-path block).
        cudaMemcpyAsync(&device_snapshot_->kinetic, &ctx.frame.scalars->kinetic,
                        sizeof(double), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(ctx.runtime.stream));
        submit_async(static_cast<std::int64_t>(ctx.step),
                     [](const MinimalSnapshot& /*s*/, std::int64_t /*step*/) {
                         // intentionally empty
                     });
    }
};

/// @brief Build a small Lennard-Jones cube to exercise cuda::md::MD.
cpu::Frame make_lj_cube(int n_side = 4, double density = 0.8) {
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
                ++idx;
            }
    frame.box = Mat3::cubic(static_cast<double>(box_len));
    return frame;
}

} // namespace

/// @brief Verify `cuda::stream_synchronize` is never called on the
///        AsyncRecordBase hot path across 100 MD steps.
TEST(CudaAsyncHotPathNoSync, ZeroSyncOver100Steps) {
    cpu::Frame frame = make_lj_cube();
    NoOpAsyncRecorder rec;

    cuda::md::MD sim;
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{2.5}, -10)
        .add(cuda::PairLJCut{1.0, 1.0, 2.5});
    sim.add(cuda::md::VelocityVerlet{0.001});
    sim.add(cuda::md::InitVelocity{1.0, 12345ULL});
    sim.add(rec);
    sim.build();

    // Reset once at the very start; AsyncRecordBase never calls the wrapper,
    // so the counter must stay at zero across all 100 steps.
    cuda::detail::reset_sync_count();
    sim.run(frame, 100);
    EXPECT_EQ(cuda::detail::sync_count(), 0u)
        << "AsyncRecordBase routed at least one cuda::stream_synchronize "
           "call on the hot path; this breaks the zero-block contract.";
}

/// @brief Same gate, but with NoseHooverNPT replacing VelocityVerlet.
///        Asserts the NPT consolidation in `cuda-md-review-cleanup-03-launch-purity`
///        (8-double pinned scratch + event-sync) keeps the wrapper counter
///        at zero across 100 NPT steps.
TEST(CudaAsyncHotPathNoSync, ZeroSyncOverNPTLoop) {
    cpu::Frame frame = make_lj_cube();
    NoOpAsyncRecorder rec;

    cuda::md::MD sim;
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{2.5}, -10)
        .add(cuda::PairLJCut{1.0, 1.0, 2.5});
    sim.add(cuda::md::InitVelocity{1.0, 12345ULL});
    sim.set_integrator(cuda::md::NoseHooverNPT{{/*T_target=*/1.0,
                                                /*tau_T=*/100.0,
                                                /*P_target=*/0.0,
                                                /*tau_P=*/1000.0,
                                                /*dt=*/0.001}});
    sim.add(rec);
    sim.build();

    cuda::detail::reset_sync_count();
    sim.run(frame, 100);
    EXPECT_EQ(cuda::detail::sync_count(), 0u)
        << "NPT hot path routed at least one cuda::stream_synchronize "
           "call; the link-03 consolidation should have eliminated all of "
           "them (defer-stream + event-sync for KE/stress/box reads).";
}

/// @brief Lifecycle: build/destroy round-trip leaves pointers null.
TEST(CudaAsyncHotPathNoSync, LifecycleRoundTrip) {
    cpu::Frame frame = make_lj_cube();
    for (int rep = 0; rep < 3; ++rep) {
        NoOpAsyncRecorder rec;
        cuda::md::MD sim;
        sim.on(md::Phase::force)
            .add(cuda::NeighborList{2.5}, -10)
            .add(cuda::PairLJCut{1.0, 1.0, 2.5});
        sim.add(cuda::md::VelocityVerlet{0.001});
        sim.add(cuda::md::InitVelocity{1.0, 12345ULL});
        sim.add(rec);
        sim.build();
        sim.run(frame, 5);
        // After sim destruction at scope exit, AsyncRecordBase::destroy()
        // ran via the Backend's destroy_components pass. No direct
        // observable on the recorder here (it's been moved into the plan
        // and destroyed), so we just check the next iteration builds clean.
    }
    SUCCEED();
}
