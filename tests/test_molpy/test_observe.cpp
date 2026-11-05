/// @file tests/test_molpy/test_observe.cpp
/// @brief Drift fence for molpy trajectory-analysis FFI via ATV wrappers
/// (accumulator → push → finish; molpy has no batch form).

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/observe/trajectory_metrics.h"

using ATV::cpu::compute::LinearFit;
using ATV::cpu::compute::MSD;
using ATV::cpu::compute::RDF;
using ATV::cpu::compute::VACF;

namespace {

// One atom, displaced by 1 Å between two frames: MSD(1) = 1 Å², at lag
// times 0 and dt.
TEST(MolpyObserve, MsdAccumulatorTwoFrames) {
    auto acc = MSD{/*max_lag=*/1}.accumulator();
    const std::array<double, 9> cell = {20, 0, 0, 0, 20, 0, 0, 0, 20};
    acc.push({0, 0, 0}, /*dt=*/0.5, cell);
    acc.push({1, 0, 0}, /*dt=*/0.5, cell);
    EXPECT_EQ(acc.n_frames(), 2);
    const auto curve = acc.finish();
    const auto msd = curve.values();
    ASSERT_EQ(msd.size(), 2u);
    EXPECT_DOUBLE_EQ(msd[0], 0.0);
    EXPECT_NEAR(msd[1], 1.0, 1e-12);
    EXPECT_EQ(curve.lag_times(), (std::vector<double>{0.0, 0.5}));
}

// A lag with no origin is refused, never clamped (molpy Correlator docs:
// "max_lag ≤ T − 1 (a lag with no origin is OutOfRange, never clamped)").
TEST(MolpyObserve, MsdFinishThrowsWithFewerThanMaxLagPlusOneFrames) {
    auto acc = MSD{/*max_lag=*/3}.accumulator();
    acc.push({0, 0, 0}, 1.0);
    acc.push({1, 0, 0}, 1.0);
    EXPECT_THROW(acc.finish(), std::runtime_error);
}

// A step longer than half the cell height is a wrapped trajectory; the frame
// is refused and the state is unchanged.
TEST(MolpyObserve, MsdAccumulatorRefusesAWrappedStep) {
    auto acc = MSD{/*max_lag=*/1}.accumulator();
    const std::array<double, 9> cell = {4, 0, 0, 0, 4, 0, 0, 0, 4};
    acc.push({0.1, 0, 0}, 1.0, cell);
    EXPECT_THROW(acc.push({3.9, 0, 0}, 1.0, cell), std::runtime_error);
    EXPECT_EQ(acc.n_frames(), 1);
}

// molpy `compute::time_series` module doc example: two DOF over four frames,
// v = (1,0) (0,1) (-1,0) (0,-1) — each DOF has zero mean, so the unbiased
// C(0) = <v_d²> = 0.5, C(1) = 0, C(2) = -0.5.
TEST(MolpyObserve, VacfAccumulatorMatchesMolpyDocExample) {
    const std::vector<double> vel = {1, 0, 0, 1, -1, 0, 0, -1};
    auto acc = VACF{/*max_lag=*/2}.accumulator();
    for (std::size_t t = 0; t < 4; ++t)
        acc.push({vel[2 * t], vel[2 * t + 1]}, /*dt=*/1.0);
    const auto acf = acc.finish().values();
    ASSERT_EQ(acf.size(), 3u);
    EXPECT_NEAR(acf[0], 0.5, 1e-12);
    EXPECT_NEAR(acf[1], 0.0, 1e-12);
    EXPECT_NEAR(acf[2], -0.5, 1e-12);
}

TEST(MolpyObserve, VacfAccumulatorRefusesAWidthChange) {
    auto acc = VACF{/*max_lag=*/1}.accumulator();
    acc.push({1.0, 2.0, 3.0}, 1.0);
    EXPECT_THROW(acc.push({1.0, 2.0}, 1.0), std::runtime_error);
    EXPECT_EQ(acc.n_frames(), 1);
}

TEST(MolpyObserve, LinearFitOfAnMsdCurve) {
    // One atom at unit speed: MSD(t) = t² over lags 0..9, whose least-squares
    // line has slope 9 and intercept -12. (blocked x|y|z with one atom is
    // just (x, y, z) per frame)
    auto acc = MSD{9}.accumulator();
    for (int t = 0; t < 10; ++t)
        acc.push({static_cast<double>(t), 0, 0}, 1.0);
    const auto fit = LinearFit{0.0, 1.0}.compute(acc.finish());
    EXPECT_NEAR(fit.slope, 9.0, 1e-9);
    EXPECT_NEAR(fit.intercept, -12.0, 1e-9);
    EXPECT_EQ(fit.fit_start, 0);
    EXPECT_EQ(fit.fit_end, 9);
}

TEST(MolpyObserve, ConstructorsRefuseOutOfRangeConfiguration) {
    EXPECT_THROW(MSD{0}, std::invalid_argument);
    EXPECT_THROW((MSD{2, ATV::cpu::compute::MsdOrigins::All, 0}),
                 std::invalid_argument);
    EXPECT_THROW((MSD{2, ATV::cpu::compute::MsdOrigins::First, 2}),
                 std::invalid_argument);
    EXPECT_THROW(VACF{0}, std::invalid_argument);
    EXPECT_THROW((LinearFit{0.8, 0.2}), std::invalid_argument);
    EXPECT_THROW((RDF{0, 4.0, 0.0}), std::invalid_argument);
    EXPECT_THROW((RDF{10, 1.0, 2.0}), std::invalid_argument);
}

TEST(MolpyObserve, RdfAccumulatorOnePairInOneBin) {
    // Two particles 1.5 Å apart in a 20 Å cube (blocked x|y|z: (0,0,0) and
    // (0,1.5,0)): the one pair lands in bin [1, 2), whose ideal-gas shell
    // volume 4π/3·(2³ − 1³) gives g = 2·1 / (ρ·V_shell·N) = 4000 / V_shell
    // with ρ = 2 / 8000 Å⁻³ and N = 2.
    auto acc = RDF(/*n_bins=*/5, /*r_max=*/5.0, /*r_min=*/0.0).accumulator();
    const std::array<double, 9> cell = {20, 0, 0, 0, 20, 0, 0, 0, 20};
    acc.push({0, 0, 0, 1.5, 0, 0}, cell);
    const auto g = acc.finish();
    ASSERT_EQ(g.rdf.size(), 5u);
    ASSERT_EQ(g.bin_centers.size(), 5u);
    ASSERT_EQ(g.bin_edges.size(), 6u);
    EXPECT_EQ(g.n_frames, 1);
    EXPECT_DOUBLE_EQ(g.volume, 8000.0);
    const double v_shell = 4.0 / 3.0 * M_PI * (8.0 - 1.0);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(g.bin_centers[i], static_cast<double>(i) + 0.5, 1e-12);
        EXPECT_DOUBLE_EQ(g.n_r[i], i == 1 ? 1.0 : 0.0) << "bin " << i;
        EXPECT_NEAR(g.rdf[i], i == 1 ? 4000.0 / v_shell : 0.0, 1e-9)
            << "bin " << i;
    }
}

TEST(MolpyObserve, RdfAccumulatorFinish) {
    auto acc = RDF(/*n_bins=*/4, /*r_max=*/4.0, /*r_min=*/0.0).accumulator();
    EXPECT_THROW(acc.finish(), std::runtime_error); // no frame yet
    const std::array<double, 9> cell = {20, 0, 0, 0, 20, 0, 0, 0, 20};
    acc.push({0, 1.0, 0, 0, 0, 0}, cell);
    EXPECT_EQ(acc.n_frames(), 1);
    EXPECT_EQ(acc.finish().rdf.size(), 4u);
    const std::array<double, 9> singular{};
    EXPECT_THROW(acc.push({0, 1.0, 0, 0, 0, 0}, singular), std::runtime_error);
    EXPECT_EQ(acc.n_frames(), 1);
}

} // namespace
