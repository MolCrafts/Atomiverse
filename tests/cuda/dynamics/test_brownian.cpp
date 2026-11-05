/// @file tests/cuda/dynamics/test_brownian.cpp
/// @brief Host checks for the overdamped step the CUDA kernel calls.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/dynamics/brownian.cuh"
#include "atomiverse/cuda/dynamics/brownian_step.h"
#include "atomiverse/cuda/frame/frame.cuh"

using namespace ATV;

TEST(CudaBrownian, RejectsNonPositiveParameters) {
    EXPECT_THROW(cuda::md::Brownian(0.0, 0.01, 300.0), std::invalid_argument);
    EXPECT_THROW(cuda::md::Brownian(-1.0, 0.01, 300.0), std::invalid_argument);
    EXPECT_THROW(cuda::md::Brownian(1.0, 0.0, 300.0), std::invalid_argument);
    EXPECT_THROW(cuda::md::Brownian(1.0, 0.01, 0.0), std::invalid_argument);
    EXPECT_THROW(cuda::md::Brownian(std::numeric_limits<double>::quiet_NaN(),
                                    0.01, 300.0),
                 std::invalid_argument);
}

TEST(CudaBrownian, PublishesNameTimestepAndBindings) {
    cuda::md::Brownian bd(0.5, 0.02, 300.0, /*seed=*/7);
    EXPECT_EQ(bd.name(), "brownian");
    EXPECT_DOUBLE_EQ(bd.dt(), 0.5);
    EXPECT_DOUBLE_EQ(bd.D(), 0.02);
    const auto bindings = bd.default_bindings();
    ASSERT_EQ(bindings.size(), 2u);
    EXPECT_EQ(bindings[0].kind, PhaseKind::MD);
    EXPECT_EQ(bindings[1].kind, PhaseKind::MD);
    EXPECT_EQ(bindings[0].phase, slot_of(md::Phase::initial_integrate));
    EXPECT_EQ(bindings[1].phase, slot_of(md::Phase::final_integrate));
    EXPECT_EQ(bindings[0].order, 0);
    EXPECT_EQ(bindings[1].order, 0);
}

TEST(CudaBrownian, DriftMatchesMobility) {
    const double D = 0.02;
    const double temperature = 300.0;
    const double dt = 0.5;
    const double force = 0.1;
    const double mobility = cuda::md::brownian_mobility(D, temperature);
    const double noise = cuda::md::brownian_noise_amplitude(D, dt);
    EXPECT_DOUBLE_EQ(mobility, D / (units::md_boltz * temperature));
    EXPECT_DOUBLE_EQ(noise, std::sqrt(2.0 * D * dt));
    const double x =
        cuda::md::brownian_advance(1.25, force, true, mobility, dt, noise, 0.0);
    EXPECT_DOUBLE_EQ(x, 1.25 + mobility * force * dt);
}

TEST(CudaBrownian, NonFiniteForceSkipsDrift) {
    const double mobility = cuda::md::brownian_mobility(0.02, 300.0);
    const double noise = cuda::md::brownian_noise_amplitude(0.02, 0.5);
    EXPECT_FALSE(cuda::md::brownian_forces_finite(
        std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0));
    EXPECT_FALSE(cuda::md::brownian_forces_finite(
        0.0, std::numeric_limits<double>::infinity(), 0.0));
    const double x = cuda::md::brownian_advance(
        1.25, std::numeric_limits<double>::quiet_NaN(), false, mobility, 0.5,
        noise, 0.0);
    EXPECT_DOUBLE_EQ(x, 1.25);
}

TEST(CudaBrownian, WrapMatchesCpuSandboxRule) {
    EXPECT_DOUBLE_EQ(cuda::md::brownian_wrap(10.5, 10.0), 0.5);
    EXPECT_DOUBLE_EQ(cuda::md::brownian_wrap(-0.25, 10.0), 9.75);
    EXPECT_DOUBLE_EQ(cuda::md::brownian_wrap(10.0, 10.0), 0.0);
    EXPECT_DOUBLE_EQ(cuda::md::brownian_wrap(3.0, 1.0e6), 3.0);
    EXPECT_DOUBLE_EQ(cuda::md::brownian_wrap(3.0, 0.0), 3.0);
    EXPECT_TRUE(cuda::md::brownian_should_wrap(true, 20.0));
    EXPECT_FALSE(cuda::md::brownian_should_wrap(true, 1.0e6));
    EXPECT_FALSE(cuda::md::brownian_should_wrap(false, 20.0));
}

TEST(CudaBrownian, FreeParticleStepVariance) {
    const double D = 0.01;
    const double dt = 1.0;
    const double mobility = cuda::md::brownian_mobility(D, 300.0);
    const double noise = cuda::md::brownian_noise_amplitude(D, dt);
    std::mt19937_64 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);
    const int n = 20000;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x = cuda::md::brownian_advance(0.0, 0.0, true, mobility,
                                                    dt, noise, normal(rng));
        acc += x * x;
    }
    const double expect = 2.0 * D * dt;
    EXPECT_NEAR(acc / n, expect, 0.08 * expect);
}

#ifndef ATV_HAS_CUDA
TEST(CudaBrownian, CpuBuildNamesMissingKernel) {
    cuda::md::Brownian bd(1.0, 0.01, 300.0, 1);
    cuda::Context runtime;
    cuda::Frame frame;
    const cuda::LaunchContext final_phase{
        runtime, frame, slot_of(md::Phase::final_integrate), 0};
    EXPECT_NO_THROW(bd.launch(final_phase));

    const cuda::LaunchContext move{runtime, frame,
                                   slot_of(md::Phase::initial_integrate), 0};
    try {
        bd.launch(move);
        FAIL() << "missing kernel must throw";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("brownian_integrate"), std::string::npos)
            << message;
    }
}
#endif
