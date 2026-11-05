/// @file tests/cuda/dynamics/test_md_primitives.cu
/// @brief Single-component checks for tagged noise and fused scalar reductions.
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/dynamics/brownian.cuh"
#include "atomiverse/cuda/dynamics/fix_langevin.cuh"
#include "atomiverse/cuda/dynamics/init.cuh"
#include "atomiverse/cuda/dynamics/kinetic_energy.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

using namespace ATV;
namespace {

// Component-author fixture: a nondefault stream, no Driver or integration loop.
struct DeviceInput {
    ATV::cuda::Frame frame;
    ATV::cuda::Context context;
    cudaStream_t stream;
    explicit DeviceInput(const cpu::Frame& host) : frame(cpu::to_cuda(host)) {
        ATV_CUDA_CHECK(cudaStreamCreate(&stream));
        context.stream = stream;
        context.md.n_dof_global = md::degrees_of_freedom(host.nlocal());
    }
    ~DeviceInput() {
        cudaStreamSynchronize(stream);
        ATV::cuda::destroy(frame);
        cudaStreamDestroy(stream);
    }
};

class TaggedNoise : public testing::TestWithParam<bool> {
  protected:
    static constexpr int n = 8195; // partial final warp and block
    std::array<std::vector<double>, 3>
    sample(bool langevin, int step, bool reversed, unsigned seed = 42) {
        cpu::Frame host(n);
        for (int i = 0; i < n; ++i) {
            const int tag_index = reversed ? n - 1 - i : i;
            // Adjacent identities share their low 32 bits. Truncating a tag
            // would give each pair identical noise and fail the correlation check.
            host.span(kAtom_tag)[i] =
                (std::int64_t(1) << 40) * (1 + tag_index % 2)
                + 7 * (tag_index / 2);
            host.span(kAtom_mass)[i] = 1;
        }
        host.rebuild_tag_index(0, n);
        DeviceInput input(host);
        // Choose amplitudes equal to one in public units, so output is N(0,1).
        if (langevin) {
            const double T = units::md_mvv2e * units::md_ftm2v * units::md_ftm2v
                             / (2 * units::md_boltz);
            ATV::cuda::md::FixLangevin thermostat(T, T, 1, 1, seed);
            thermostat.launch({input.context, input.frame,
                               slot_of(md::Phase::post_force), step});
        } else {
            ATV::cuda::md::Brownian brownian(1, 0.5, 300, seed);
            brownian.launch({input.context, input.frame,
                             slot_of(md::Phase::initial_integrate), step});
        }
        ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
        ATV::cuda::to_cpu(input.frame, host);
        std::array<std::vector<double>, 3> result;
        for (auto& axis : result)
            axis.resize(n);
        const auto x = langevin ? host.span(kAtom_fx) : host.span(kAtom_x);
        const auto y = langevin ? host.span(kAtom_fy) : host.span(kAtom_y);
        const auto z = langevin ? host.span(kAtom_fz) : host.span(kAtom_z);
        for (int i = 0; i < n; ++i) {
            const int tag_index = reversed ? n - 1 - i : i;
            result[0][tag_index] = x[i];
            result[1][tag_index] = y[i];
            result[2][tag_index] = z[i];
        }
        return result;
    }
};

TEST_P(TaggedNoise, ReorderingPreservesFullWidthTagsAndRepeatedSteps) {
    const auto a = sample(GetParam(), 7, false);
    const auto b = sample(GetParam(), 7, true);
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n; ++i)
            EXPECT_DOUBLE_EQ(a[d][i], b[d][i]);
}

TEST_P(TaggedNoise, UnitVarianceAndIndependentAxesStepsAndSeeds) {
    const auto a = sample(GetParam(), 7, false);
    const auto next = sample(GetParam(), 8, false);
    const auto other_seed = sample(GetParam(), 7, false, 43);
    for (int d = 0; d < 3; ++d) {
        double sum = 0, square = 0, axes = 0, steps = 0, seeds = 0, tags = 0;
        for (int i = 0; i < n; ++i) {
            ASSERT_TRUE(std::isfinite(a[d][i]));
            sum += a[d][i];
            square += a[d][i] * a[d][i];
            axes += a[d][i] * a[(d + 1) % 3][i];
            steps += a[d][i] * next[d][i];
            seeds += a[d][i] * other_seed[d][i];
            if (i % 2 == 0 && i + 1 < n)
                tags += a[d][i] * a[d][i + 1];
        }
        EXPECT_NEAR(sum / n, 0, 0.05);
        EXPECT_NEAR(square / n, 1, 0.08);
        EXPECT_NEAR(axes / n, 0, 0.06);
        EXPECT_NEAR(steps / n, 0, 0.06);
        EXPECT_NEAR(seeds / n, 0, 0.06);
        EXPECT_NEAR(tags / (n / 2), 0, 0.08);
    }
}

TEST_P(TaggedNoise, ComponentsDoNotReuseNoiseAtTheSameSeedAndStep) {
    const auto own = sample(GetParam(), 0, false);
    const auto other = sample(!GetParam(), 0, false);
    for (int d = 0; d < 3; ++d) {
        double cross = 0;
        for (int i = 0; i < n; ++i)
            cross += own[d][i] * other[d][i];
        EXPECT_NEAR(cross / n, 0, 0.06);
    }
}

INSTANTIATE_TEST_SUITE_P(BrownianAndLangevin, TaggedNoise,
                         testing::Values(false, true));

TEST(CudaMdReductions, KineticEnergyAndSixStressComponentsHaveDistinctSums) {
    cpu::Frame host(259);
    double expected[7] = {};
    for (int i = 0; i < host.nlocal(); ++i) {
        const double m = 1 + i % 3, x = (i % 5 - 2) * 0.125;
        const double y = (i % 7 - 3) * 0.25, z = (i % 11 - 5) * 0.5;
        host.span(kAtom_mass)[i] = m;
        host.span(kAtom_vx)[i] = x;
        host.span(kAtom_vy)[i] = y;
        host.span(kAtom_vz)[i] = z;
        const double c = m * units::md_mvv2e;
        expected[0] += 0.5 * c * (x * x + y * y + z * z);
        expected[1] += c * x * x;
        expected[2] += c * y * y;
        expected[3] += c * z * z;
        expected[4] += c * x * y;
        expected[5] += c * x * z;
        expected[6] += c * y * z;
    }
    DeviceInput input(host);
    ATV::cuda::md::KineticEnergy kinetic;
    kinetic.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
    ATV::cuda::to_cpu(input.frame, host);
    constexpr double relative =
        std::is_same_v<ATV::cuda::mixed_t, float> ? 2e-6 : 2e-13;
    EXPECT_NEAR(host.meta.get(kKinetic), expected[0], relative * expected[0]);
    const auto stress = host.meta.get(kStressKinetic);
    for (int d = 0; d < 6; ++d)
        EXPECT_NEAR(stress[d], expected[d + 1],
                    relative * std::max(1.0, std::abs(expected[d + 1])));
}

TEST(CudaMdReductions, InitVelocityHasTargetTemperatureAndZeroMomentum) {
    constexpr int n = 8195;
    cpu::Frame host(n);
    for (int i = 0; i < n; ++i) {
        host.span(kAtom_mass)[i] = 1 + i % 3;
        host.span(kAtom_tag)[i] = (std::int64_t(1) << 40) + 7 * i;
    }
    host.rebuild_tag_index(0, n);
    DeviceInput input(host);
    ATV::cuda::md::InitVelocity initialize(300, 42);
    initialize.build({input.context, input.frame});
    ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
    ATV::cuda::to_cpu(input.frame, host);
    double momentum[3] = {}, mv2 = 0, mass = 0;
    for (int i = 0; i < n; ++i) {
        const double m = host.span(kAtom_mass)[i];
        const double v[3] = {host.span(kAtom_vx)[i], host.span(kAtom_vy)[i],
                             host.span(kAtom_vz)[i]};
        mass += m;
        for (int d = 0; d < 3; ++d) {
            momentum[d] += m * v[d];
            mv2 += m * v[d] * v[d];
        }
    }
    EXPECT_NEAR(units::md_mvv2e * mv2 / ((3 * n - 3) * units::md_boltz), 300,
                (std::is_same_v<ATV::cuda::mixed_t, float> ? 1e-3 : 1e-9));
    // The COM residual is roundoff on velocities of size sigma, so bound it
    // relative to sigma (unit mass). The bound is the old absolute one
    // divided by the metal-unit sigma, sqrt(kB 300 / mvv2e) = 0.0158 A/fs.
    const double sigma = std::sqrt(units::md_boltz * 300 / units::md_mvv2e);
    const double tolerance =
        (std::is_same_v<ATV::cuda::mixed_t, float> ? 6e-8 : 6e-15) * sigma;
    for (double p : momentum)
        EXPECT_NEAR(p / mass, 0, tolerance);
}

TEST(CudaMdNoise, LangevinDoesNotReuseInitializationSamples) {
    constexpr int n = 8195;
    constexpr double T = 300;
    cpu::Frame host(n);
    for (int i = 0; i < n; ++i) {
        host.span(kAtom_mass)[i] = 1;
        host.span(kAtom_tag)[i] = i;
    }
    host.rebuild_tag_index(0, n);
    DeviceInput input(host);
    ATV::cuda::md::InitVelocity initialize(T, 42);
    initialize.build({input.context, input.frame});
    ATV::cuda::md::FixLangevin thermostat(T, T, 1, 1, 42);
    thermostat.launch(
        {input.context, input.frame, slot_of(md::Phase::post_force), 0});
    ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
    ATV::cuda::to_cpu(input.frame, host);
    const double sigma_v = std::sqrt(units::md_boltz * T / units::md_mvv2e);
    const double sigma_f = std::sqrt(2.0) * sigma_v / units::md_ftm2v;
    double cross[3] = {};
    for (int i = 0; i < n; ++i) {
        const double v[3] = {host.span(kAtom_vx)[i], host.span(kAtom_vy)[i],
                             host.span(kAtom_vz)[i]};
        const double f[3] = {host.span(kAtom_fx)[i], host.span(kAtom_fy)[i],
                             host.span(kAtom_fz)[i]};
        for (int d = 0; d < 3; ++d)
            cross[d] +=
                (v[d] / sigma_v) * ((f[d] + v[d] / units::md_ftm2v) / sigma_f);
    }
    for (double value : cross)
        EXPECT_NEAR(value / n, 0, 0.06);
}

TEST(CudaMdReductions, NoseHooverUsesEveryBlockBeforeScalingVelocities) {
    cpu::Frame host(259);
    double ke = 0;
    for (int i = 0; i < host.nlocal(); ++i) {
        host.span(kAtom_mass)[i] = 1;
        const double v = 0.001 * (1 + i % 5);
        host.span(kAtom_vx)[i] = v;
        ke += 0.5 * units::md_mvv2e * v * v;
    }
    const double T = 300, tau = 10, dt = 0.25;
    const double target = (3 * host.nlocal() - 3) * units::md_boltz * T;
    const double xi = 0.5 * dt * (2 * ke - target) / (target * tau * tau);
    const double scale = std::exp(-0.5 * dt * xi);
    DeviceInput input(host);
    ATV::cuda::md::NoseHoover thermostat(T, tau, dt);
    thermostat.build({input.context, input.frame});
    thermostat.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
    ATV::cuda::to_cpu(input.frame, host);
    thermostat.destroy({input.context, input.frame});
    constexpr double tolerance =
        std::is_same_v<ATV::cuda::mixed_t, float> ? 2e-9 : 1e-14;
    for (int i = 0; i < host.nlocal(); ++i)
        EXPECT_NEAR(host.span(kAtom_vx)[i], scale * 0.001 * (1 + i % 5),
                    tolerance);
}

// Same timestep rule as cpu::md::NoseHoover: the integrator's pushed dt, not
// the constructor's, sets the half step.
TEST(CudaMdReductions, NoseHooverUsesThePushedIntegratorTimestep) {
    cpu::Frame host(64);
    double ke = 0;
    for (int i = 0; i < host.nlocal(); ++i) {
        host.span(kAtom_mass)[i] = 2;
        const double v = 0.002 * (1 + i % 3);
        host.span(kAtom_vy)[i] = v;
        ke += 0.5 * units::md_mvv2e * 2 * v * v;
    }
    const double T = 50, tau = 20, dt = 0.5;
    const double target = (3 * host.nlocal() - 3) * units::md_boltz * T;
    const double xi = 0.5 * dt * (2 * ke - target) / (target * tau * tau);
    const double scale = std::exp(-0.5 * dt * xi);
    DeviceInput input(host);
    ATV::cuda::md::NoseHoover thermostat(T, tau, /*dt=*/7.0);
    ASSERT_NE(thermostat.as_timestep_consumer(), nullptr);
    EXPECT_FALSE(thermostat.timestep_required());
    thermostat.set_timestep(dt);
    thermostat.build({input.context, input.frame});
    thermostat.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
    ATV::cuda::to_cpu(input.frame, host);
    thermostat.destroy({input.context, input.frame});
    for (int i = 0; i < host.nlocal(); ++i)
        EXPECT_NEAR(host.span(kAtom_vy)[i], scale * 0.002 * (1 + i % 3), 1e-12);
}

TEST(CudaMdReductions, NoseHooverRefusesNonPositiveParameters) {
    using ATV::cuda::md::NoseHoover;
    EXPECT_THROW(NoseHoover(0.0, 10.0, 1.0), std::invalid_argument);
    EXPECT_THROW(NoseHoover(300.0, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(NoseHoover(300.0, 10.0, -1.0), std::invalid_argument);
    NoseHoover ok(300.0, 10.0, 1.0);
    EXPECT_THROW(ok.set_timestep(0.0), std::invalid_argument);
    EXPECT_THROW(ok.set_timestep(std::nan("")), std::invalid_argument);
}

// One atom has no thermal DOF after COM removal (Q = 0): the CPU twin leaves
// the velocities alone, and so must this one, instead of writing NaN.
TEST(CudaMdReductions, NoseHooverLeavesAZeroDofSystemUntouched) {
    cpu::Frame host(1);
    host.span(kAtom_mass)[0] = 1;
    host.span(kAtom_vx)[0] = 0.01;
    DeviceInput input(host);
    ASSERT_EQ(input.context.md.n_dof_global, 0);
    ATV::cuda::md::NoseHoover thermostat(300, 10, 1);
    thermostat.build({input.context, input.frame});
    thermostat.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
    ATV::cuda::to_cpu(input.frame, host);
    thermostat.destroy({input.context, input.frame});
    EXPECT_EQ(host.span(kAtom_vx)[0], 0.01);
}

} // namespace

#include "atomiverse/cuda/dynamics/barostat.cuh"

TEST(CudaMdNoise, InitialForceAndFirstStepUseIndependentLangevinSamples) {
    constexpr int n = 8195;
    cpu::Frame host(n);
    for (int i = 0; i < n; ++i) {
        host.span(kAtom_mass)[i] = 1;
        host.span(kAtom_tag)[i] = (std::int64_t(1) << 40) + i;
    }
    DeviceInput input(host);
    ATV::cuda::md::FixLangevin thermostat(300, 300, 1, 1, 42);
    ATV::cuda::ZeroForces zero;
    const int phase = slot_of(md::Phase::post_force);
    thermostat.initialize_forces({input.context, input.frame, phase, 0, true});
    ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
    ATV::cuda::to_cpu(input.frame, host);
    std::vector<double> initial(n);
    for (int i = 0; i < n; ++i)
        initial[i] = host.span(kAtom_fx)[i];
    zero.launch({input.context, input.frame, phase, 0});
    thermostat.launch({input.context, input.frame, phase, 0});
    ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
    ATV::cuda::to_cpu(input.frame, host);
    double dot = 0, a2 = 0, b2 = 0;
    for (int i = 0; i < n; ++i) {
        const double b = host.span(kAtom_fx)[i];
        dot += initial[i] * b;
        a2 += initial[i] * initial[i];
        b2 += b * b;
    }
    EXPECT_LT(std::abs(dot / std::sqrt(a2 * b2)), 0.05);
}

TEST(CudaMdPressure, BerendsenUsesKineticVirialVolumeAndModulus) {
    constexpr int n = 8195; // multiple blocks must use the same dilation
    cpu::Frame host(n);
    host.box = cpu::Box::cubic(10);
    host.meta.get(kStress) = {1000, 2000, 3000, 0, 0, 0};
    host.meta.get(kStressKinetic) = {500, 1000, 1500, 0, 0, 0};
    for (int i = 0; i < n; ++i)
        host.span(kAtom_x)[i] = 1.0;
    DeviceInput input(host);
    ATV::cuda::md::Barostat barostat(1, 10, 1, 2);
    barostat.build({input.context, input.frame});
    barostat.launch(
        {input.context, input.frame, slot_of(md::Phase::end_of_step), 0});
    ATV_CUDA_CHECK(cudaStreamSynchronize(input.stream));
    ATV::cuda::to_cpu(input.frame, host);
    const double mu = md::berendsen_dilation(3.0, 1.0, 1.0, 10.0, 2.0);
    constexpr double tolerance =
        std::is_same_v<ATV::cuda::mixed_t, float> ? 1e-5 : 1e-12;
    EXPECT_NEAR(host.box.length(0), 10 * mu, tolerance);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(host.span(kAtom_x)[i], mu, tolerance);
    barostat.destroy({input.context, input.frame});
}
