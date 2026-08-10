/// @file tests/cpu/gpw/test_cpu_gpw_vol_forces.cpp
/// @brief Fixed-volume (Hirshfeld-scaled) TS/MBD force dual-gates (geometry-only).
///
/// Density-chain ∂v/∂R is out of scope: forces hold \(v_A\) fixed. Geometry-
/// only FD duals stay unit-test class. Product-path Hirshfeld fold through
/// `GpwDft::run` lives in `reg_gpw-hirshfeld-forces` /
/// `reg_gpw-hirshfeld-kpoint-forces` (e2e-in-tests wave-2).

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"

using ATV::cpu::gpw::mbd_rsscs_pbe;
using ATV::cpu::gpw::ts_pbe;

namespace {

constexpr double kAngToBohr = 1.0 / 0.52917720859;

std::vector<std::array<double, 3>> water_dimer_bohr() {
    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    std::vector<std::array<double, 3>> pos(6);
    for (int i = 0; i < 6; ++i)
        for (int d = 0; d < 3; ++d)
            pos[static_cast<std::size_t>(i)][d] = ang[i][d] * kAngToBohr;
    return pos;
}

const std::vector<int> kZ = {8, 1, 1, 8, 1, 1};

} // namespace

TEST(GpwVolForces, VolumeScaledMbdForcesMatchFd) {
    auto pos = water_dimer_bohr();
    const double cell0[3] = {0.0, 0.0, 0.0};
    const std::vector<double> v = {0.74, 0.65, 0.65, 0.72, 0.58, 0.58};
    const auto r = mbd_rsscs_pbe(kZ, pos, cell0, /*periodic=*/false, 0.83, 6.0,
                                 v);
    ASSERT_EQ(r.forces.size(), 6u);
    EXPECT_TRUE(std::isfinite(r.energy));
    EXPECT_LT(r.energy, 0.0);
    // Distinct from free-atom (v=1) water-dimer pin ~ -1.493e-3 Ha.
    EXPECT_GT(r.energy, -1.4e-3);
    EXPECT_LT(r.energy, -0.8e-3);

    const double h = 1.0e-5;
    double max_err = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    auto pos_fd = pos;
    for (std::size_t i = 0; i < 6; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos_fd[i][d];
            pos_fd[i][d] = save + h;
            const double ep =
                mbd_rsscs_pbe(kZ, pos_fd, cell0, false, 0.83, 6.0, v).energy;
            pos_fd[i][d] = save - h;
            const double em =
                mbd_rsscs_pbe(kZ, pos_fd, cell0, false, 0.83, 6.0, v).energy;
            pos_fd[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - r.forces[i][d]));
            sum[d] += r.forces[i][d];
        }
    }
    const double sum_norm =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    std::printf("[Vol F MBD] E=%.6e max|F-F_fd|=%.2e |sum F|=%.2e\n", r.energy,
                max_err, sum_norm);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(sum_norm, 1e-8);
}

// Same dual-gate for volume-scaled TS.
TEST(GpwVolForces, VolumeScaledTsForcesMatchFd) {
    auto pos = water_dimer_bohr();
    const double cell0[3] = {0.0, 0.0, 0.0};
    const std::vector<double> v = {0.74, 0.65, 0.65, 0.72, 0.58, 0.58};
    const auto r = ts_pbe(kZ, pos, cell0, /*periodic=*/false, 94.2, 0.94, 20.0,
                          v);
    ASSERT_EQ(r.forces.size(), 6u);
    EXPECT_TRUE(std::isfinite(r.energy));
    EXPECT_LT(r.energy, 0.0);

    const double h = 1.0e-5;
    double max_err = 0.0;
    auto pos_fd = pos;
    for (std::size_t i = 0; i < 6; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos_fd[i][d];
            pos_fd[i][d] = save + h;
            const double ep =
                ts_pbe(kZ, pos_fd, cell0, false, 94.2, 0.94, 20.0, v).energy;
            pos_fd[i][d] = save - h;
            const double em =
                ts_pbe(kZ, pos_fd, cell0, false, 94.2, 0.94, 20.0, v).energy;
            pos_fd[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - r.forces[i][d]));
        }
    }
    std::printf("[Vol F TS] E=%.6e max|F-F_fd|=%.2e\n", r.energy, max_err);
    EXPECT_LT(max_err, 1e-6);
}
