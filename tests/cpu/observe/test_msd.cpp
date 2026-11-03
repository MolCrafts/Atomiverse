/// @file tests/cpu/observe/test_msd.cpp
/// @brief MSD via the molpy FFI bridge (cpu::compute::MSD + LinearFit) and
/// the ATV::MSD observer that streams snapshots into it.
///
/// Validates the molpy-first analysis chain: blocked x|y|z position frames
/// pushed one at a time into the molpy MSD accumulator -> curve back to C++.
/// All analysis math is in molpy; this asserts the FFI round-trip reproduces
/// the analytic MSD and that the Einstein composition (LinearFit slope /
/// 2·dims) is wired correctly.

#include "atomiverse/cpu/observe/msd.h"
#include "atomiverse/cpu/observe/trajectory_metrics.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using ATV::cpu::compute::LinearFit;
using ATV::cpu::compute::MSD;
using ATV::cpu::compute::MsdOrigins;

namespace {
// Two atoms with known linear displacement: atom 0 moves +a per frame along x,
// atom 1 moves +b per frame along y. One frame, blocked x|y|z.
std::vector<double> ballistic_frame(double a, double b, int t) {
    return {a * t, 0.0, 0.0, b * t, 0.0, 0.0};
}

// The MSD curve of `n_frames` ballistic frames `dt` apart.
ATV::cpu::compute::TimeCorrelation
ballistic_curve(const MSD& msd, double a, double b, int n_frames, double dt) {
    auto acc = msd.accumulator();
    for (int t = 0; t < n_frames; ++t)
        acc.push(ballistic_frame(a, b, t), dt);
    return acc.finish();
}
} // namespace

// MSD(t) = <|r(τ+t) - r(τ)|^2>_i = ((a t)^2 + (b t)^2) / 2 = t^2 (a^2+b^2)/2.
// Constant velocities make every origin equivalent, so the frame-0 estimator
// (LAMMPS `compute msd`) and the all-origins (windowed) estimator agree.
TEST(CpuMSD, LinearDisplacementMatchesAnalytic) {
    const double a = 1.0, b = 2.0, dt = 0.5;
    const int n_frames = 4, max_lag = n_frames - 1;

    for (const MsdOrigins origins : {MsdOrigins::First, MsdOrigins::All}) {
        const auto curve =
            ballistic_curve(MSD{max_lag, origins}, a, b, n_frames, dt);
        const std::vector<double> msd = curve.values();
        const std::vector<double> lag = curve.lag_times();
        ASSERT_EQ(msd.size(), static_cast<std::size_t>(max_lag + 1));
        ASSERT_EQ(lag.size(), msd.size());
        for (int t = 0; t <= max_lag; ++t) {
            const double expected =
                static_cast<double>(t) * t * (a * a + b * b) / 2.0;
            EXPECT_NEAR(msd[t], expected, 1e-9) << "lag t=" << t;
            EXPECT_DOUBLE_EQ(lag[t], dt * t);
        }
        EXPECT_DOUBLE_EQ(msd[0], 0.0);
    }
}

// Fewer frames than max_lag + 1 (including a single frame or none) is a
// molpy refusal, never an empty or clamped curve.
TEST(CpuMSD, TooFewFramesThrows) {
    EXPECT_THROW(ballistic_curve(MSD{1}, 1.0, 2.0, 1, 1.0), std::runtime_error);
    EXPECT_THROW(ballistic_curve(MSD{1}, 1.0, 2.0, 0, 1.0), std::runtime_error);
    EXPECT_THROW(ballistic_curve(MSD{5}, 1.0, 2.0, 4, 1.0), std::runtime_error);
}

TEST(CpuMSD, RaggedFrameThrows) {
    // 10 values do not split into blocked x|y|z.
    auto acc = MSD{1}.accumulator();
    EXPECT_THROW(acc.push(std::vector<double>(10, 0.0), 1.0),
                 std::runtime_error);
    EXPECT_EQ(acc.n_frames(), 0);
}

// Einstein composition: D = slope / (2·dims) of a molpy LinearFit over the
// MSD curve. For MSD(τ) = c·i² at lags τ = i·dt, i = s..e consecutive, the
// least-squares slope of i² on i is (s + e), so slope = c·(s + e)/dt — an
// exact check that the curve, its lag times and the fit window reach molpy
// unchanged.
TEST(CpuMSD, EinsteinSlopeOfABallisticCurve) {
    const double a = 1.0, b = 2.0, dt = 2.0;
    const int n_frames = 11, max_lag = 10;
    const double c = (a * a + b * b) / 2.0;
    const auto fit = LinearFit{0.2, 0.8}.compute(
        ballistic_curve(MSD{max_lag}, a, b, n_frames, dt));
    EXPECT_EQ(fit.fit_start, 2);
    EXPECT_EQ(fit.fit_end, 8);
    EXPECT_NEAR(fit.slope, c * (2 + 8) / dt, 1e-9);
    EXPECT_NEAR(fit.slope / (2.0 * 3), c * 10.0 / dt / 6.0, 1e-9);
}

// ── ATV::MSD observer: snapshot packing + the Einstein composition ─────────

namespace {
/// @brief Storage behind one hand-built snapshot view.
struct BallisticFrame {
    std::vector<std::int64_t> tag{1, 2};
    std::vector<double> x, y, z;
    ATV::cpu::FrameSnapshot snap;
    BallisticFrame(double a, double b, int t, double box)
        : x{a * t, 0.0}, y{0.0, b * t}, z{0.0, 0.0} {
        snap.step = t;
        snap.n_owned = 2;
        snap.tag = tag;
        snap.box = ATV::cpu::Box::cubic(box);
        snap.x = x;
        snap.y = y;
        snap.z = z;
    }
};

std::string scratch_path(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}
} // namespace

// The observer streams blocked x|y|z + cell into molpy and composes
// D = LinearFit slope / (2·3); for MSD = c·i² the slope over consecutive lags
// s..e is c·(s + e)/dt (see EinsteinSlopeOfABallisticCurve).
TEST(CpuMSDObserver, CurveAndEinsteinDFromStreamedSnapshots) {
    const double a = 1.0, b = 2.0, dt = 2.0, c = (a * a + b * b) / 2.0;
    const int window = 4, n_frames = 7;
    const std::string out = scratch_path("atv_test_msd_observer.dat");
    std::filesystem::remove(out);

    ATV::MSD msd{dt, /*freq=*/0, out, window};
    for (int t = 0; t < n_frames; ++t)
        msd.on_snapshot(BallisticFrame(a, b, t, /*box=*/100.0).snap);
    msd.finalize();

    EXPECT_EQ(msd.n_frames(), n_frames);
    const std::vector<double>& curve = msd.curve();
    ASSERT_EQ(curve.size(), static_cast<std::size_t>(window + 1));
    for (int t = 0; t <= window; ++t)
        EXPECT_NEAR(curve[t], c * t * t, 1e-9) << "lag " << t;

    const auto fit = LinearFit{0.2, 0.8}.compute(
        ballistic_curve(MSD{window}, a, b, n_frames, dt));
    EXPECT_NEAR(msd.diffusion(),
                c * (fit.fit_start + fit.fit_end) / dt / (2.0 * 3), 1e-9);
    EXPECT_TRUE(std::filesystem::exists(out));
    std::filesystem::remove(out);
}

// Fewer than window + 1 frames: molpy refuses the unresolved lags, so the
// observer writes nothing and leaves curve empty / D NaN (finalize never
// throws — it runs on the Runtime's shutdown path).
TEST(CpuMSDObserver, TooShortRunWritesNothing) {
    const std::string out = scratch_path("atv_test_msd_observer_short.dat");
    std::filesystem::remove(out);
    ATV::MSD msd{/*dt=*/1.0, /*freq=*/0, out, /*window=*/5};
    for (int t = 0; t < 3; ++t)
        msd.on_snapshot(BallisticFrame(1.0, 2.0, t, 100.0).snap);
    EXPECT_NO_THROW(msd.finalize());
    EXPECT_TRUE(msd.curve().empty());
    EXPECT_TRUE(std::isnan(msd.diffusion()));
    EXPECT_FALSE(std::filesystem::exists(out));
}

// A step longer than half the cell height is a wrapped trajectory: molpy
// refuses that snapshot and the observer counts only accepted frames.
TEST(CpuMSDObserver, WrappedStepIsRefusedNotAccumulated) {
    ATV::MSD msd{/*dt=*/1.0, /*freq=*/0, "unused_msd.dat", /*window=*/1};
    msd.on_snapshot(BallisticFrame(1.0, 2.0, 0, /*box=*/10.0).snap);
    msd.on_snapshot(BallisticFrame(6.0, 0.0, 1, /*box=*/10.0).snap); // 6 Å > 5
    EXPECT_EQ(msd.n_frames(), 1);
}

TEST(CpuMSDObserver, ConstructorRefusesBadArguments) {
    EXPECT_THROW((ATV::MSD{0.0, 0, "unused_msd.dat", 4}),
                 std::invalid_argument);
    EXPECT_THROW((ATV::MSD{1.0, 0, "unused_msd.dat", 0}),
                 std::invalid_argument);
}

TEST(CorrelationInput, AtomPermutationPreservesPackedIdentity) {
    using V = std::vector<double>;
    using T = std::vector<std::int64_t>;
    ATV::cpu::CorrelationInput input;
    std::vector<double> packed;
    input.pack(packed, T{20, 10}, V{2, 1}, V{4, 3}, V{6, 5}, 2, 1, 1);
    EXPECT_EQ(packed, (std::vector<double>{1, 2, 3, 4, 5, 6}));
    input.pack(packed, T{10, 20}, V{1, 2}, V{3, 4}, V{5, 6}, 2, 2, 1);
    EXPECT_EQ(packed, (std::vector<double>{1, 2, 3, 4, 5, 6}));
    EXPECT_THROW(
        input.pack(packed, T{10, 30}, V{1, 2}, V{3, 4}, V{5, 6}, 2, 3, 1),
        std::runtime_error);
}

TEST(CorrelationInput, MissingCaptureDoesNotCompressTimeAxis) {
    using V = std::vector<double>;
    using T = std::vector<std::int64_t>;
    ATV::cpu::CorrelationInput input;
    std::vector<double> packed;
    input.pack(packed, T{1}, V{0}, V{0}, V{0}, 1, 10, 10);
    EXPECT_THROW(input.pack(packed, T{1}, V{0}, V{0}, V{0}, 1, 30, 10),
                 std::runtime_error);
}
