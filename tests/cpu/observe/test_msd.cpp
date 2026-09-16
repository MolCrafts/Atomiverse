/// @file tests/cpu/observer/test_msd.cpp
/// @brief MSD via the molrs FFI bridge (cpu::compute::MSD / EinsteinDiffusion).
///
/// Validates the molrs-first analysis chain: a flat `[n_frames, 3*n_atoms]`
/// position buffer -> molrs compute::MSD -> curve back to C++. All analysis math
/// is in molrs; this asserts the FFI round-trip reproduces the analytic MSD.

#include "atomiverse/cpu/observe/trajectory_metrics.h"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

using ATV::cpu::compute::EinsteinDiffusion;
using ATV::cpu::compute::MSD;

namespace {
// Append one frame's positions (blocked x|y|z) to a flat [n_frames, 3*n_atoms]
// buffer.
void push_frame(std::vector<double>& pos, const std::vector<double>& x,
                const std::vector<double>& y, const std::vector<double>& z) {
    pos.insert(pos.end(), x.begin(), x.end());
    pos.insert(pos.end(), y.begin(), y.end());
    pos.insert(pos.end(), z.begin(), z.end());
}
} // namespace

// Two atoms with known linear displacement: atom 0 moves +a per frame along x,
// atom 1 moves +b per frame along y. Direct-mode MSD relative to frame 0:
//   MSD(t) = <|r(t) - r(0)|^2>_i = ( (a t)^2 + (b t)^2 ) / 2 = t^2 (a^2+b^2)/2.
TEST(CpuMSD, LinearDisplacementMatchesAnalytic) {
    const double a = 1.0, b = 2.0;
    const int n_frames = 4, n_dof = 6; // 2 atoms
    std::vector<double> pos;
    for (int t = 0; t < n_frames; ++t)
        push_frame(pos, {a * t, 0.0}, {0.0, b * t}, {0.0, 0.0});

    std::vector<double> msd = MSD{}.compute(pos, n_frames, n_dof);
    ASSERT_EQ(msd.size(), static_cast<std::size_t>(n_frames));

    for (int t = 0; t < n_frames; ++t) {
        const double expected =
            static_cast<double>(t) * t * (a * a + b * b) / 2.0;
        EXPECT_NEAR(msd[t], expected, 1e-9) << "frame t=" << t;
    }
    EXPECT_DOUBLE_EQ(msd[0], 0.0); // reference frame
}

// Fewer than two frames -> empty curve, never a panic/abort across the FFI.
TEST(CpuMSD, SingleFrameReturnsEmpty) {
    std::vector<double> pos{0.0, 0.0, 0.0}; // 1 frame, 1 atom
    EXPECT_TRUE(MSD{}.compute(pos, 1, 3).empty());
}

TEST(CpuMSD, NoFramesReturnsEmpty) {
    EXPECT_TRUE(MSD{}.compute({}, 0, 6).empty());
}

// Einstein-diffusion bridge (molrs EinsteinDiffusion + LinearFit). molrs owns
// the numerical correctness (its own Rust tests); here we validate the FFI
// round-trip: a valid trajectory yields a finite D, degenerate input yields NaN.
TEST(CpuMSD, DiffusionBridgeFiniteForValidTrajectory) {
    const int n_frames = 6, n_dof = 6;
    std::vector<double> pos;
    for (int t = 0; t < n_frames; ++t)
        push_frame(pos, {1.0 * t, 0.0}, {0.0, 2.0 * t}, {0.0, 0.0});
    const double d =
        EinsteinDiffusion{/*dt=*/1.0, /*dims=*/3, 0.2, 0.8}.compute(
            pos, n_frames, n_dof);
    EXPECT_TRUE(std::isfinite(d)) << "D must be finite for a valid trajectory";
}

TEST(CpuMSD, DiffusionEmptyIsNaN) {
    EXPECT_TRUE(
        std::isnan(EinsteinDiffusion{1.0, 3, 0.2, 0.8}.compute({}, 0, 6)));
}
