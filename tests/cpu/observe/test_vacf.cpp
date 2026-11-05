/// @file tests/cpu/observe/test_vacf.cpp
/// @brief VACF via the molpy FFI bridge (cpu::compute::VACF) and the ATV::VACF
/// observer that streams snapshots into it.
///
/// Validates the molpy-first analysis chain: velocity frames pushed one at a
/// time into the molpy Vacf accumulator -> curve back to C++. molpy owns the
/// correlator numerics (its own Rust tests); here we assert the FFI round-trip
/// against a hand-computed curve and the refusal of degenerate input.

#include "atomiverse/cpu/observe/trajectory_metrics.h"
#include "atomiverse/cpu/observe/vacf.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using ATV::cpu::compute::VACF;

// One velocity frame of n_dof = 3. molpy Vacf defaults (module doc table +
// `Correlator` docs): each DOF mean-subtracted, unbiased 1/(T - t) origin
// average, mean over DOF.
//   dof 0 alternates +1/-1 (mean 0)  -> fluctuation ±1
//   dof 1 alternates  2/ 0 (mean 1)  -> fluctuation ±1
//   dof 2 stays zero                 -> contributes 0
// so C(k) = (1 + 1 + 0)/3 · (-1)^k = (2/3)(-1)^k at every lag.
TEST(CpuVACF, AlternatingSignalMatchesHandComputedCurve) {
    const int n_frames = 8, max_lag = 4;
    auto acc = VACF{max_lag}.accumulator();
    for (int t = 0; t < n_frames; ++t)
        acc.push({(t % 2 == 0) ? 1.0 : -1.0, (t % 2 == 0) ? 2.0 : 0.0, 0.0},
                 /*dt=*/1.0);
    EXPECT_EQ(acc.n_frames(), n_frames);
    const auto curve = acc.finish();
    const std::vector<double> vacf = curve.values();
    ASSERT_EQ(vacf.size(), static_cast<std::size_t>(max_lag + 1));
    for (int k = 0; k <= max_lag; ++k) {
        EXPECT_NEAR(vacf[k], (2.0 / 3.0) * ((k % 2 == 0) ? 1.0 : -1.0), 1e-12)
            << "lag " << k;
        EXPECT_DOUBLE_EQ(curve.lag_times()[k], static_cast<double>(k));
    }
}

TEST(CpuVACF, FewerThanMaxLagPlusOneFramesThrows) {
    auto one = VACF{1}.accumulator();
    one.push({1.0, 2.0, 3.0}, 1.0); // 1 frame, 3 dof
    EXPECT_THROW(one.finish(), std::runtime_error);
    auto two = VACF{2}.accumulator();
    two.push({1.0, 2.0, 3.0}, 1.0);
    two.push({4.0, 5.0, 6.0}, 1.0); // 2 frames
    EXPECT_THROW(two.finish(), std::runtime_error);
    EXPECT_THROW(VACF{1}.accumulator().finish(), std::runtime_error);
}

TEST(CpuVACF, ChangedWidthOrIntervalThrows) {
    auto acc = VACF{1}.accumulator();
    acc.push({1.0, 2.0}, 1.0);
    EXPECT_THROW(acc.push({1.0, 2.0, 3.0}, 1.0), std::runtime_error);
    EXPECT_THROW(acc.push({1.0, 2.0}, 2.0), std::runtime_error);
    EXPECT_EQ(acc.n_frames(), 1);
}

// ── ATV::VACF observer: snapshot packing into the molpy accumulator ────────

// One atom whose vx alternates ±1 and vy alternates 2/0 (vz = 0): packed
// blocked vx|vy|vz it is the same 3-DOF signal as above, so the streamed
// curve is (2/3)(-1)^k.
TEST(CpuVACFObserver, StreamedSnapshotsMatchHandComputedCurve) {
    const std::string out =
        (std::filesystem::temp_directory_path() / "atv_test_vacf_observer.dat")
            .string();
    std::filesystem::remove(out);
    const int max_lag = 3;
    ATV::VACF vacf{/*dt=*/0.5, max_lag, /*freq=*/0, out};
    for (int t = 0; t < 8; ++t) {
        ATV::cpu::FrameSnapshot snap;
        snap.n_owned = 1;
        snap.tag = {1};
        snap.step = t;
        snap.vx = {(t % 2 == 0) ? 1.0 : -1.0};
        snap.vy = {(t % 2 == 0) ? 2.0 : 0.0};
        snap.vz = {0.0};
        vacf.on_snapshot(snap);
    }
    vacf.finalize();
    EXPECT_EQ(vacf.n_frames(), 8);
    const std::vector<double>& c = vacf.curve();
    ASSERT_EQ(c.size(), static_cast<std::size_t>(max_lag + 1));
    for (int k = 0; k <= max_lag; ++k)
        EXPECT_NEAR(c[k], (2.0 / 3.0) * ((k % 2 == 0) ? 1.0 : -1.0), 1e-12)
            << "lag " << k;
    EXPECT_TRUE(std::filesystem::exists(out));
    std::filesystem::remove(out);
}

TEST(CpuVACFObserver, ConstructorRefusesBadArguments) {
    EXPECT_THROW((ATV::VACF{0.0, 4, 0, "unused_vacf.dat"}),
                 std::invalid_argument);
    EXPECT_THROW((ATV::VACF{1.0, 0, 0, "unused_vacf.dat"}),
                 std::invalid_argument);
}
