/// @file cuda/minimize/test_motion_precision.cu
/// @brief Direct optimizer operations preserve physical coordinates and history.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/minimize/cg.cuh"
#include "atomiverse/cuda/minimize/conv_check.cuh"
#include "atomiverse/cuda/minimize/lbfgs.cuh"
#include "atomiverse/cuda/minimize/sd.cuh"
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {
namespace gpu = ATV::cuda;
constexpr double kLength = 512.000000123;
constexpr double kPosition = 100.0 + 1e-8;
constexpr double kMass = 12.0110000003;
constexpr double kTolerance = sizeof(gpu::motion_t) == 8 ? 2e-13 : 2e-5;

struct DeviceState {
    ATV::cpu::Frame host;
    gpu::Frame device;
    gpu::Context runtime;
    ATV::LoopMetrics metrics;
    cudaStream_t stream = nullptr;

    explicit DeviceState(int n = 1) : host(n) {
        host.box = ATV::cpu::Box::cubic(kLength);
        for (int i = 0; i < n; ++i) {
            host.span(ATV::kAtom_x)[i] = kPosition;
            host.span(ATV::kAtom_y)[i] = 1.0;
            host.span(ATV::kAtom_z)[i] = 1.0;
            host.span(ATV::kAtom_mass)[i] = kMass;
        }
        device = ATV::cpu::to_cuda(host);
        ATV_CUDA_CHECK(cudaStreamCreate(&stream));
        runtime.stream = stream;
        runtime.loop.metrics = &metrics;
        for (int d = 0; d < 3; ++d)
            runtime.host_motion_box.cell[d * 3 + d] = gpu::motion_t(kLength);
        runtime.host_motion_box.refresh();
        runtime.host_box = gpu::box_cast<gpu::real_t>(runtime.host_motion_box);
    }

    ~DeviceState() {
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        gpu::destroy(device);
        ATV_CUDA_CHECK(cudaStreamDestroy(stream));
    }

    gpu::LaunchContext search(int step = 0) {
        return {runtime, device,
                ATV::slot_of(ATV::minimizer::Phase::search_direction), step};
    }

    gpu::LaunchContext convergence(int step = 0) {
        return {runtime, device,
                ATV::slot_of(ATV::minimizer::Phase::convergence_check), step};
    }

    void force(gpu::real_t x, gpu::real_t y = 0, gpu::real_t z = 0) {
        ATV_CUDA_CHECK(cudaMemcpyAsync(device.atoms.fx, &x, sizeof(x),
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(device.atoms.fy, &y, sizeof(y),
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(device.atoms.fz, &z, sizeof(z),
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void energy(double value) {
        ATV_CUDA_CHECK(cudaMemcpyAsync(&device.scalars->energy, &value,
                                       sizeof(value), cudaMemcpyHostToDevice,
                                       stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    double x() {
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        gpu::to_cpu(device, host);
        return host.span(ATV::kAtom_x)[0];
    }
};

void expect_empty_search(gpu::Component& optimizer) {
    DeviceState state(0);
    optimizer.build({state.runtime, state.device});
    optimizer.launch(state.search());
    ATV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    optimizer.destroy({state.runtime, state.device});
}
} // namespace

TEST(MinimizerMotionPrecision, SteepestDescentLaunchRetainsResidualAndMass) {
    DeviceState state;
    state.force(1);
    constexpr double alpha = 0.25;
    const double initial = state.host.span(ATV::kAtom_x)[0];
    const double mass = state.host.span(ATV::kAtom_mass)[0];
    gpu::minimizer::SteepestDescent optimizer(alpha);
    optimizer.build({state.runtime, state.device});
    optimizer.launch(state.search());
    const double expected = initial + alpha / mass * ATV::units::force_to_accel;
    EXPECT_NEAR(state.x(), expected, kTolerance);
    optimizer.destroy({state.runtime, state.device});
}

TEST(MinimizerMotionPrecision, CgLaunchRetainsSubFloatUlpStep) {
    DeviceState state;
    state.force(1);
    constexpr double alpha = 1e-8;
    const double initial = state.host.span(ATV::kAtom_x)[0];
    gpu::minimizer::CG optimizer(alpha);
    optimizer.build({state.runtime, state.device});
    optimizer.launch(state.search());
    EXPECT_NEAR(state.x(), initial + alpha, kTolerance);
    // The second PR+ step has beta=2 and direction=4. It must start
    // from the first physical step, whose entire displacement is sub-ULP.
    state.force(2);
    optimizer.launch(state.search(1));
    EXPECT_NEAR(state.x(), initial + 5 * alpha, kTolerance);
    optimizer.destroy({state.runtime, state.device});
}

TEST(MinimizerMotionPrecision, LbfgsLaunchRetainsSubFloatUlpStep) {
    DeviceState state;
    state.force(1);
    constexpr double alpha = 1e-8;
    const double initial = state.host.span(ATV::kAtom_x)[0];
    gpu::minimizer::LBFGS optimizer(2, alpha);
    optimizer.build({state.runtime, state.device});
    optimizer.launch(state.search());
    EXPECT_NEAR(state.x(), initial + alpha, kTolerance);
    optimizer.destroy({state.runtime, state.device});
}

TEST(MinimizerMotionPrecision, LbfgsLaunchRollbackUsesFullPrecisionOrigin) {
    DeviceState state;
    state.force(1);
    constexpr double alpha = 1e-8;
    const double initial = state.host.span(ATV::kAtom_x)[0];
    gpu::minimizer::LBFGS optimizer(2, alpha);
    optimizer.line_search(1e-4, 0.5, 1e-14);
    optimizer.build({state.runtime, state.device});
    optimizer.launch(state.search());
    EXPECT_NEAR(state.x(), initial + alpha, kTolerance);
    // Inject a rejected trial's scalar energy. There is no force evaluator or
    // simulation loop: launch() alone owns the Armijo state and contraction.
    state.energy(1.0);
    optimizer.launch(state.convergence());
    optimizer.launch(state.search(1));
    EXPECT_NEAR(state.x(), initial + alpha * 0.5, kTolerance);
    optimizer.destroy({state.runtime, state.device});
}

TEST(MinimizerMotionPrecision, ConvCheckMeasureUsesResidualAndPrecisePbc) {
    DeviceState state;
    gpu::destroy(state.device);
    state.host.span(ATV::kAtom_x)[0] = 1e-8;
    state.host.span(ATV::kAtom_y)[0] = kPosition;
    state.device = ATV::cpu::to_cuda(state.host);
    state.force(2, -3, 6);
    state.energy(4.0);
    const gpu::motion_t previous[] = {gpu::motion_t(kLength - 2e-8),
                                      gpu::motion_t(100), gpu::motion_t(1)};
    gpu::motion_t* device_previous = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&device_previous, sizeof(previous)));
    ATV_CUDA_CHECK(cudaMemcpyAsync(device_previous, previous, sizeof(previous),
                                   cudaMemcpyHostToDevice, state.stream));
    gpu::minimizer::ConvCheck check;
    check.build(1);
    std::vector<gpu::minimizer::OptRecord> trace;
    check.measure(state.convergence(1), device_previous, 1, trace);
    EXPECT_DOUBLE_EQ(state.metrics.max_force, 7.0);
    EXPECT_NEAR(state.metrics.rms_force, std::sqrt(49.0 / 3.0), 1e-13);
    EXPECT_TRUE(state.metrics.has_displacement);
    if constexpr (sizeof(gpu::motion_t) == 8) {
        EXPECT_NEAR(state.metrics.max_displacement, 3e-8, 2e-13);
        EXPECT_NEAR(state.metrics.rms_displacement,
                    std::sqrt((9e-16 + 1e-16) / 3.0), 2e-13);
    } else {
        EXPECT_DOUBLE_EQ(state.metrics.max_displacement, 0.0);
        EXPECT_DOUBLE_EQ(state.metrics.rms_displacement, 0.0);
    }
    ASSERT_EQ(trace.size(), 1u);
    EXPECT_DOUBLE_EQ(trace[0].energy, 4.0);
    check.destroy();
    ATV_CUDA_CHECK(cudaFree(device_previous));
}

TEST(MinimizerMotionPrecision, ConvCheckMeasureEmptyPublishesFiniteMetrics) {
    DeviceState state(0);
    state.energy(2.5);
    gpu::minimizer::ConvCheck check;
    check.build(0);
    std::vector<gpu::minimizer::OptRecord> trace;
    check.measure(state.convergence(), nullptr, 0, trace);
    EXPECT_DOUBLE_EQ(state.metrics.energy, 2.5);
    EXPECT_DOUBLE_EQ(state.metrics.max_force, 0.0);
    EXPECT_DOUBLE_EQ(state.metrics.rms_force, 0.0);
    EXPECT_DOUBLE_EQ(state.metrics.max_displacement, 0.0);
    EXPECT_DOUBLE_EQ(state.metrics.rms_displacement, 0.0);
    EXPECT_FALSE(state.metrics.has_displacement);
    EXPECT_TRUE(std::isfinite(state.metrics.energy_change));
    ASSERT_EQ(trace.size(), 1u);
    check.destroy();
}

TEST(MinimizerMotionPrecision,
     SteepestDescentLaunchEmptyDoesNotLaunchGridZero) {
    gpu::minimizer::SteepestDescent optimizer(0.01);
    expect_empty_search(optimizer);
}

TEST(MinimizerMotionPrecision, CgLaunchEmptyDoesNotLaunchGridZero) {
    gpu::minimizer::CG optimizer;
    expect_empty_search(optimizer);
}

TEST(MinimizerMotionPrecision, LbfgsLaunchEmptyDoesNotLaunchGridZero) {
    gpu::minimizer::LBFGS optimizer;
    expect_empty_search(optimizer);
}
