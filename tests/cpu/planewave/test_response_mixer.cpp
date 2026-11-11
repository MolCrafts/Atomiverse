#include "atomiverse/cpu/planewave/internal/response_mixer.h"

#include <array>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

namespace {
using Real = ATV::AccReal;
using Cx = std::complex<Real>;
using ATV::cpu::pw::internal::ResponseMixer;
} // namespace

TEST(PwResponseMixer, RealSingleChannelStartsWithTheOriginalLinearUpdate) {
    std::array<Real, 3> density{.2, -.3, .7}, output{.8, .1, -.2};
    const auto initial = density;
    ResponseMixer mixer(density.size(), 1, .4, 5);
    mixer.mix<Real>({{density, output}});
    for (std::size_t i = 0; i < density.size(); ++i)
        EXPECT_DOUBLE_EQ(density[i],
                         initial[i] + .4 * (output[i] - initial[i]));
}

TEST(PwResponseMixer, ComplexSingleChannelRetainsRealAndImaginaryDensity) {
    std::array<Cx, 2> density{Cx(.3, -.8), Cx(-.2, .1)};
    const std::array<Cx, 2> output{Cx(-.4, .2), Cx(.6, -.5)};
    const auto initial = density;
    ResponseMixer mixer(density.size(), 1, .5, 0);
    mixer.mix<Cx>({{density, output}});
    for (std::size_t i = 0; i < density.size(); ++i)
        EXPECT_EQ(density[i], initial[i] + .5 * (output[i] - initial[i]));
    EXPECT_GT(std::abs(density[0].imag() + density[1].imag()), .1);
}

TEST(PwResponseMixer, JointOppositeSpinResidualsRemainIndependentInformation) {
    // The total-density residual vanishes at every step. The joint residual
    // must still solve the nonzero magnetic response instead of cancelling it.
    std::array<Real, 1> up{0}, down{0}, new_up{1}, new_down{-1};
    ResponseMixer mixer(1, 2, .5, 5);
    mixer.mix<Real>({{up, new_up}, {down, new_down}});
    EXPECT_DOUBLE_EQ(up[0], .5);
    EXPECT_DOUBLE_EQ(down[0], -.5);
    new_up[0] = .8 * up[0] + 1;
    new_down[0] = .8 * down[0] - 1;
    mixer.mix<Real>({{up, new_up}, {down, new_down}});
    EXPECT_NEAR(up[0], 5, 1e-11);
    EXPECT_NEAR(down[0], -5, 1e-11);
}

TEST(PwResponseMixer, PulaySolvesAStronglyCoupledRealLinearMap) {
    // Eigenvalues .97 and .73 make ordinary beta=.5 mixing slow, while two
    // independent residual directions determine this affine fixed point.
    const std::array<Real, 2> target{1, -2};
    const auto evaluate = [&](const auto& x) {
        return std::array<Real, 2>{
            target[0] + .85 * (x[0] - target[0]) + .12 * (x[1] - target[1]),
            target[1] + .12 * (x[0] - target[0]) + .85 * (x[1] - target[1])};
    };
    std::array<Real, 2> density{0, 0}, linear{0, 0};
    ResponseMixer mixer(2, 1, .5, 5);
    int iterations = 0;
    for (; iterations < 8; ++iterations) {
        const auto output = evaluate(density);
        if (std::hypot(output[0] - density[0], output[1] - density[1]) < 1e-11)
            break;
        mixer.mix<Real>({{density, output}});
    }
    EXPECT_LT(iterations, 8);
    EXPECT_NEAR(density[0], target[0], 1e-9);
    EXPECT_NEAR(density[1], target[1], 1e-9);
    for (int i = 0; i < 8; ++i) {
        const auto output = evaluate(linear);
        for (int j = 0; j < 2; ++j)
            linear[j] += .5 * (output[j] - linear[j]);
    }
    EXPECT_GT(std::hypot(linear[0] - target[0], linear[1] - target[1]), .1);
}

TEST(PwResponseMixer, ComplexJointMixPreservesChargeAndTheImaginaryResponse) {
    const Cx charge_up(1, .2), charge_down(-.5, .3);
    const Cx target_up(.7, .2), target_down(-.3, .8);
    std::array<Cx, 2> up{charge_up, charge_up};
    std::array<Cx, 2> down{charge_down, charge_down};
    ResponseMixer mixer(2, 2, .5, 5);
    int iterations = 0;
    for (; iterations < 8; ++iterations) {
        const Cx error_up = .5 * (up[0] - up[1]) - target_up;
        const Cx error_down = .5 * (down[0] - down[1]) - target_down;
        const Cx next_up = target_up + .25 * error_up + .7 * error_down;
        const Cx next_down = target_down + .7 * error_up + .25 * error_down;
        const std::array<Cx, 2> output_up{charge_up + next_up,
                                          charge_up - next_up};
        const std::array<Cx, 2> output_down{charge_down + next_down,
                                            charge_down - next_down};
        if (std::abs(output_up[0] - up[0]) + std::abs(output_down[0] - down[0])
            < 1e-11)
            break;
        mixer.mix<Cx>({{up, output_up}, {down, output_down}});
        EXPECT_NEAR(std::abs(up[0] + up[1] - Real(2) * charge_up), 0, 1e-12);
        EXPECT_NEAR(std::abs(down[0] + down[1] - Real(2) * charge_down), 0,
                    1e-12);
    }
    EXPECT_LT(iterations, 8);
    EXPECT_NEAR(std::abs(.5 * (up[0] - up[1]) - target_up), 0, 1e-9);
    EXPECT_NEAR(std::abs(.5 * (down[0] - down[1]) - target_down), 0, 1e-9);
}

TEST(PwResponseMixer, ResetStartsAnIndependentPerturbationHistory) {
    ResponseMixer reused(2, 1, .4, 5), fresh(2, 1, .4, 5);
    std::array<Real, 2> density{0, 0}, output{1, -1};
    reused.mix<Real>({{density, output}});
    output = {1.2, -.9};
    reused.mix<Real>({{density, output}});
    reused.reset();
    density = {.1, -.2};
    output = {-.7, .8};
    auto independent = density;
    reused.mix<Real>({{density, output}});
    fresh.mix<Real>({{independent, output}});
    EXPECT_EQ(density, independent);
}
