/// @file cuda/dynamics/test_motion_precision.cu
/// @brief VelocityVerlet::launch preserves increments below force-position ULP.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include <gtest/gtest.h>

using namespace ATV;

TEST(MotionPrecision, InitialIntegrateRetainsSmallDriftAndWrapsWithFullCell) {
    cpu::Frame frame(2);
    constexpr double length = 512.000000123;
    constexpr double dt = 0.25;
    frame.box = cpu::Box::cubic(length);
    const double positions[] = {100.0 + 1e-8, length - 1e-8};
    const double velocities[] = {1e-7, 2e-7};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_x)[i] = positions[i];
        frame.span(kAtom_y)[i] = 1;
        frame.span(kAtom_z)[i] = 1;
        frame.span(kAtom_vx)[i] = velocities[i];
        frame.span(kAtom_mass)[i] = 12.011;
    }
    auto device = cpu::to_cuda(frame);
    cuda::Context runtime;
    for (int c = 0; c < 9; ++c) {
        runtime.host_box.cell[c] = cuda::real_t(c % 4 == 0 ? length : 0);
        runtime.host_motion_box.cell[c] =
            cuda::motion_t(c % 4 == 0 ? length : 0);
    }
    runtime.host_box.refresh();
    runtime.host_motion_box.refresh();
    cuda::md::VelocityVerlet integrator{dt};
    integrator.launch(
        {runtime, device, slot_of(md::Phase::initial_integrate), 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    cuda::to_cpu(device, frame);
    if constexpr (sizeof(cuda::motion_t) == 8) {
        EXPECT_NEAR(frame.span(kAtom_x)[0], positions[0] + dt * velocities[0],
                    1e-13);
        EXPECT_NEAR(frame.span(kAtom_x)[1],
                    positions[1] + dt * velocities[1] - length, 1e-13);
        EXPECT_NEAR(frame.span(kAtom_vx)[0], velocities[0], 1e-20);
    } else {
        EXPECT_FLOAT_EQ(float(frame.span(kAtom_x)[0]), 100.0f);
        EXPECT_FLOAT_EQ(float(frame.span(kAtom_x)[1]), 0.0f);
    }
    EXPECT_EQ(frame.span(kAtom_image)(1, 0), 1);
    cuda::destroy(device);
}
