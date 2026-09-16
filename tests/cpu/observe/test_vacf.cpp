/// @file tests/cpu/observer/test_vacf.cpp
/// @brief VACF via the molrs FFI bridge (cpu::compute::VACF).
///
/// Validates the molrs-first analysis chain: flat velocity buffer -> molrs
/// compute::VACF -> curve back to C++. molrs owns the FFT-ACF numerics (its own
/// Rust tests); here we assert the FFI round-trip shape + degenerate handling.

#include "atomiverse/cpu/observe/trajectory_metrics.h"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

using ATV::cpu::compute::VACF;

// n_frames x n_dof row-major velocity buffer; a varying signal has positive
// zero-lag autocorrelation. Curve length == resolution + 1.
TEST(CpuVACF, ValidBufferShapeAndZeroLagPositive) {
    const int n_frames = 8, n_dof = 3, res = 4;
    std::vector<double> vel(static_cast<std::size_t>(n_frames * n_dof), 0.0);
    for (int t = 0; t < n_frames; ++t) {
        vel[static_cast<std::size_t>(t * n_dof + 0)] =
            (t % 2 == 0) ? 1.0 : -1.0;
        vel[static_cast<std::size_t>(t * n_dof + 1)] = (t % 2 == 0) ? 2.0 : 0.0;
        // dof 2 stays zero (no fluctuation) -> contributes 0 to the DOF average
    }
    std::vector<double> vacf =
        VACF{/*dt=*/1.0, res}.compute(vel, n_frames, n_dof);
    ASSERT_EQ(vacf.size(), static_cast<std::size_t>(res + 1));
    EXPECT_TRUE(std::isfinite(vacf[0]));
    EXPECT_GT(vacf[0], 0.0)
        << "zero-lag VACF of a fluctuating signal is positive";
}

TEST(CpuVACF, FewerThanTwoFramesEmpty) {
    std::vector<double> vel{1.0, 2.0, 3.0}; // 1 frame, 3 dof
    EXPECT_TRUE((VACF{1.0, 1}.compute(vel, 1, 3).empty()));
}

TEST(CpuVACF, ShapeMismatchEmpty) {
    std::vector<double> vel{1.0, 2.0, 3.0, 4.0}; // 4 != n_frames*n_dof (2*3)
    EXPECT_TRUE((VACF{1.0, 1}.compute(vel, 2, 3).empty()));
}
