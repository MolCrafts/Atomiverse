/// @file tests/test_molrs/test_wham.cpp
/// @brief Drift fence for molrs::wham_compute_new via ATV's thin wrapper.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/observe/wham.h"

#if !defined(ATV_HAS_MOLPY_WHAM)
#error "test_wham.cpp requires ATV_HAS_MOLPY_WHAM"
#endif

namespace {

TEST(MolrsWham, FreeEnergyReturnsBinCount) {
    constexpr int n_bins = 8;
    constexpr int n_win = 2;
    std::vector<double> hist(static_cast<std::size_t>(n_win * n_bins), 10.0);
    std::vector<double> n_samples = {80.0, 80.0};
    std::vector<double> k = {4.0, 4.0};
    std::vector<double> centers = {1.0, 2.0};
    ATV::cpu::compute::Wham wham(/*beta=*/1.0);
    auto F = wham.free_energy(hist, n_samples, k, centers, /*s_min=*/0.0,
                              /*s_max=*/4.0, n_bins, /*max_iter=*/200,
                              /*tol=*/1e-6);
    ASSERT_EQ(static_cast<int>(F.size()), n_bins);
    EXPECT_GT(wham.last_iterations(), 0);
}

} // namespace
