/// @file tests/cpu/observer/test_esp_charges.cpp
/// @brief Unit: MK / CHELPG grid generation + constrained ESP fit on
///        synthetic point-charge potentials — no SCF run, no determinant.
///
/// The fit path is exercised through the public `fit_potential(V, grid,
/// frame, q_tot)` member: place known point charges at the H2O geometry,
/// evaluate their ANALYTIC Coulomb potential on the generated grids, and
/// the constrained LSQ must recover the charges (the model is exact, so
/// recovery is limited only by conditioning).

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observer/esp_charges.h"

using namespace ATV;
using cpu::ChelpgCharges;
using cpu::MKCharges;

namespace {

/// H2O geometry [Å]: O(0,0,0.1173), H(0,±0.7572,−0.4692).
cpu::Frame h2o_frame() {
    cpu::Frame f(3);
    const double pos[3][3] = {{0.0, 0.0, 0.1173},
                              {0.0, 0.7572, -0.4692},
                              {0.0, -0.7572, -0.4692}};
    const int z[3] = {8, 1, 1};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_x)[i] = pos[i][0];
        f.span(kAtom_y)[i] = pos[i][1];
        f.span(kAtom_z)[i] = pos[i][2];
        f.span(kAtom_type)[i] = z[i];
    }
    f.simbox = cpu::Box::cubic(50.0);
    return f;
}

/// Analytic Coulomb potential [Hartree/e] of point charges q at the frame's
/// atom positions, evaluated at grid points [Å].
std::vector<double>
point_charge_potential(const cpu::Frame& f, const std::vector<double>& q,
                       const std::vector<std::array<double, 3>>& grid) {
    std::vector<double> V(grid.size(), 0.0);
    for (std::size_t k = 0; k < grid.size(); ++k) {
        for (std::size_t a = 0; a < q.size(); ++a) {
            const double dx = grid[k][0] - f.span(kAtom_x)[static_cast<int>(a)];
            const double dy = grid[k][1] - f.span(kAtom_y)[static_cast<int>(a)];
            const double dz = grid[k][2] - f.span(kAtom_z)[static_cast<int>(a)];
            const double r_bohr = std::sqrt(dx * dx + dy * dy + dz * dz)
                                  * units::kAngstromInBohr;
            V[k] += q[a] / r_bohr;
        }
    }
    return V;
}

const std::vector<double> kTrueQ = {-0.8, 0.4, 0.4}; // O, H, H [e]

} // namespace

/// MK grid: exact recovery of the synthetic charges, exact Σq, and
/// symmetric H equality (the exact model has a unique full-rank minimizer).
TEST(EspCharges, RecoverPointChargesOnMkGrid) {
    auto f = h2o_frame();
    MKCharges mk;
    const auto grid = mk.grid(f);
    ASSERT_GT(grid.size(), 100u); // 4 shells x 3 atoms at 1 pt/A^2
    const auto V = point_charge_potential(f, kTrueQ, grid);
    const auto q = mk.fit_potential(V, grid, f, 0.0);
    ASSERT_EQ(q.size(), 3u);
    for (int a = 0; a < 3; ++a)
        EXPECT_NEAR(q[static_cast<std::size_t>(a)],
                    kTrueQ[static_cast<std::size_t>(a)], 1e-6);
    EXPECT_NEAR(q[0] + q[1] + q[2], 0.0, 1e-12); // constraint row is exact
    EXPECT_NEAR(q[1], q[2], 1e-8);               // symmetric H equal
}

/// CHELPG grid: same recovery contract on the cubic grid.
TEST(EspCharges, RecoverPointChargesOnChelpgGrid) {
    auto f = h2o_frame();
    ChelpgCharges chelpg;
    const auto grid = chelpg.grid(f);
    ASSERT_GT(grid.size(), 1000u); // 0.3 A spacing, 2.8 A band
    const auto V = point_charge_potential(f, kTrueQ, grid);
    const auto q = chelpg.fit_potential(V, grid, f, 0.0);
    ASSERT_EQ(q.size(), 3u);
    for (int a = 0; a < 3; ++a)
        EXPECT_NEAR(q[static_cast<std::size_t>(a)],
                    kTrueQ[static_cast<std::size_t>(a)], 1e-6);
    EXPECT_NEAR(q[0] + q[1] + q[2], 0.0, 1e-12);
    EXPECT_NEAR(q[1], q[2], 1e-8);
}

/// Nonzero total-charge constraint holds exactly through the KKT row.
TEST(EspCharges, TotalChargeConstraintExact) {
    auto f = h2o_frame();
    MKCharges mk;
    const auto grid = mk.grid(f);
    const std::vector<double> q_cation = {-0.2, 0.6, 0.6}; // sums to +1
    const auto V = point_charge_potential(f, q_cation, grid);
    const auto q = mk.fit_potential(V, grid, f, 1.0);
    EXPECT_NEAR(q[0] + q[1] + q[2], 1.0, 1e-12);
    for (int a = 0; a < 3; ++a)
        EXPECT_NEAR(q[static_cast<std::size_t>(a)],
                    q_cation[static_cast<std::size_t>(a)], 1e-6);
}

/// Buried-atom pathology: two nearly coincident sites make the design
/// matrix rank-deficient; the SVD minimum-norm branch must return finite
/// charges that still satisfy the constraint — never NaN/Inf.
TEST(EspCharges, IllConditionedMinimumNormIsFinite) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 1e-9; // effectively coincident hydrogens
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.simbox = cpu::Box::cubic(50.0);

    ChelpgCharges chelpg;
    const auto grid = chelpg.grid(f);
    ASSERT_GT(grid.size(), 100u);
    const std::vector<double> q_true = {0.3, 0.3};
    const auto V = point_charge_potential(f, q_true, grid);
    const auto q = chelpg.fit_potential(V, grid, f, 0.6);
    ASSERT_EQ(q.size(), 2u);
    for (const double v : q)
        EXPECT_TRUE(std::isfinite(v));
    EXPECT_NEAR(q[0] + q[1], 0.6, 1e-6);
    // Minimum-norm resolution of the degeneracy: the equal split.
    EXPECT_NEAR(q[0], 0.3, 1e-3);
    EXPECT_NEAR(q[1], 0.3, 1e-3);
}

/// Unsupported element: the Bondi lookup throws (no fallback radius).
TEST(EspCharges, UnsupportedElementThrows) {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[0] = 79; // Au — not in the Bondi table
    f.simbox = cpu::Box::cubic(50.0);

    EXPECT_THROW(MKCharges{}.grid(f), std::invalid_argument);
    EXPECT_THROW(ChelpgCharges{}.grid(f), std::invalid_argument);
}
