/// @file tests/cpu/multiscale/test_qmmm_polarizable.cpp
/// @brief Unit tests for induced-dipole SCC math.
///
/// Full QmmmPolarizable RHF launch energy pin: regressions/multiscale/qmmm_pol_h2o
/// (e2e-in-tests wave-2).
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/multiscale/qmmm_polarizable.h"

using namespace ATV;

TEST(InducedDipole, SingleSiteUniformField) {
    const double alpha = 1.5; // Å³
    const double e_perm[3] = {0.2, -0.1, 0.05};
    const double pos[3] = {0, 0, 0};
    std::vector<double> mu;
    const double e_pol =
        cpu::solve_induced_dipoles(1, pos, &alpha, e_perm, /*thole=*/0.0,
                                   /*tol=*/1e-12, /*max_iter=*/10, mu);
    ASSERT_EQ(mu.size(), 3u);
    EXPECT_NEAR(mu[0], alpha * e_perm[0], 1e-14);
    EXPECT_NEAR(mu[1], alpha * e_perm[1], 1e-14);
    EXPECT_NEAR(mu[2], alpha * e_perm[2], 1e-14);
    const double e_expect =
        units::coulomb
        * (-0.5 * (mu[0] * e_perm[0] + mu[1] * e_perm[1] + mu[2] * e_perm[2]));
    EXPECT_NEAR(e_pol, e_expect, 1e-12);
    EXPECT_LT(e_pol, 0.0);
}

TEST(InducedDipole, SccThrowsWhenUnconverged) {
    const double pos[6] = {0, 0, 0, 0.5, 0, 0};
    const double alpha[2] = {50.0, 50.0};
    const double e_perm[6] = {1, 0, 0, -1, 0, 0};
    std::vector<double> mu;
    EXPECT_THROW(cpu::solve_induced_dipoles(2, pos, alpha, e_perm, 0.39, 1e-16,
                                            /*max_iter=*/1, mu),
                 std::runtime_error);
}
