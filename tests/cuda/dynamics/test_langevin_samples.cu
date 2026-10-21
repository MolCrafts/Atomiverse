/// @file cuda/dynamics/test_langevin_samples.cu
/// @brief Direct thermostat launches consume reproducible force samples.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/dynamics/fix_langevin.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include <gtest/gtest.h>

#include <vector>

namespace {
namespace gpu = ATV::cuda;

struct State {
    ATV::cpu::Frame host;
    gpu::Frame device;
    gpu::Context runtime;

    explicit State(int n) : host(n) {
        host.box = ATV::cpu::Box::cubic(10);
        for (int i = 0; i < n; ++i) {
            host.span(ATV::kAtom_mass)[i] = 12;
            host.span(ATV::kAtom_tag)[i] = i + 17;
            // Zero velocity isolates the noise from Stokes drag.
            host.span(ATV::kAtom_vx)[i] = 0;
            host.span(ATV::kAtom_vy)[i] = 0;
            host.span(ATV::kAtom_vz)[i] = 0;
        }
        device = ATV::cpu::to_cuda(host);
    }
    ~State() { gpu::destroy(device); }

    std::vector<gpu::real_t> sample(gpu::md::FixLangevin& fix, int step) {
        const auto n = host.nlocal();
        const auto bytes = static_cast<std::size_t>(n) * sizeof(gpu::real_t);
        if (n) {
            ATV_CUDA_CHECK(cudaMemset(device.atoms.fx, 0, bytes));
            ATV_CUDA_CHECK(cudaMemset(device.atoms.fy, 0, bytes));
            ATV_CUDA_CHECK(cudaMemset(device.atoms.fz, 0, bytes));
        }
        fix.launch(
            {runtime, device, ATV::slot_of(ATV::md::Phase::post_force), step});
        ATV_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<gpu::real_t> result(static_cast<std::size_t>(n) * 3);
        if (n) {
            ATV_CUDA_CHECK(cudaMemcpy(result.data(), device.atoms.fx, bytes,
                                      cudaMemcpyDeviceToHost));
            ATV_CUDA_CHECK(cudaMemcpy(result.data() + n, device.atoms.fy, bytes,
                                      cudaMemcpyDeviceToHost));
            ATV_CUDA_CHECK(cudaMemcpy(result.data() + 2 * n, device.atoms.fz,
                                      bytes, cudaMemcpyDeviceToHost));
        }
        return result;
    }
};
} // namespace

TEST(LangevinSamples, SameStepUsesDistinctNoiseAndBuildRestartsSequence) {
    State state(2);
    gpu::md::FixLangevin fix(200, 400, 100, 0.5, 12345);
    fix.set_run_span(0, 10);
    fix.build({state.runtime, state.device});
    const auto first = state.sample(fix, 7);
    const auto second = state.sample(fix, 7);
    EXPECT_NE(first, second);
    EXPECT_DOUBLE_EQ(fix.T_target(), 340);
    fix.build({state.runtime, state.device});
    EXPECT_EQ(state.sample(fix, 7), first);
    EXPECT_EQ(state.sample(fix, 7), second);
    EXPECT_DOUBLE_EQ(fix.T_target(), 340);
}

TEST(LangevinSamples, EmptyFrameLaunchConsumesOneForceSample) {
    State state(2), empty(0);
    gpu::md::FixLangevin reference(300, 300, 100, 0.5, 9876);
    reference.build({state.runtime, state.device});
    const auto first = state.sample(reference, 0);
    const auto second = state.sample(reference, 0);
    gpu::md::FixLangevin fix(300, 300, 100, 0.5, 9876);
    fix.build({empty.runtime, empty.device});
    EXPECT_TRUE(empty.sample(fix, 0).empty());
    const auto actual = state.sample(fix, 0);
    EXPECT_EQ(actual, second);
    EXPECT_NE(actual, first);
}
