/// @file tests/cpu/planewave/test_nc_potential.cpp
/// @brief Radial local value/derivative and immutable NC provider contract.
#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include "atomiverse/cpu/planewave/norm_conserving_potential.h"
#include <array>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

using namespace ATV;

TEST(NcPotential, GthOriginTailAndMatchedLocalDerivative) {
    cpu::pw::GthPseudopotential potential(molrs::Element::Si, "GTH-PBE-q4");
    const std::array<AccReal, 6> r{0, 1e-8, 0.13, 0.44, 1.2, 20};
    std::array<AccReal, 6> v, dv;
    potential.local_r(r, v, dv);
    // Krack's GTH-PBE Si parameters: rloc=.44, C1=-6.26928833.
    EXPECT_NEAR(v[0], -4 * std::sqrt(2 / M_PI) / .44 - 6.26928833, 1e-12);
    EXPECT_EQ(dv[0], 0);
    EXPECT_NEAR(v.back(), -4.0 / r.back(), 1e-15);
    EXPECT_NEAR(dv.back(), 4.0 / (r.back() * r.back()), 1e-15);
    for (std::size_t i = 2; i + 1 < r.size(); ++i) {
        const std::array<AccReal, 2> displaced{r[i] - 1e-6, r[i] + 1e-6};
        std::array<AccReal, 2> vp, dp;
        potential.local_r(displaced, vp, dp);
        EXPECT_NEAR(dv[i], (vp[1] - vp[0]) / 2e-6, 2e-8);
    }
}

TEST(NcPotential, LocalShapeErrorsPreserveOutput) {
    cpu::pw::GthPseudopotential potential(molrs::Element::H, "GTH-PBE-q1");
    const std::array<AccReal, 2> r{0, -1};
    std::array<AccReal, 2> v{9, 9}, dv{8, 8};
    EXPECT_THROW(potential.local_r(r, v, dv), std::invalid_argument);
    EXPECT_EQ(v[0], 9);
    EXPECT_EQ(dv[1], 8);
    const std::array<AccReal, 1> valid{0};
    EXPECT_THROW(potential.local_r(valid, v, dv), std::invalid_argument);
    EXPECT_EQ(v[1], 9);
    EXPECT_EQ(dv[0], 8);
    std::array<AccReal, 3> storage{0, 1, 2};
    const auto before = storage;
    EXPECT_THROW(potential.local_r(std::span(storage).first(2),
                                   std::span(storage).subspan(1), dv),
                 std::invalid_argument);
    EXPECT_EQ(storage, before);
    const std::array<AccReal, 2> disjoint{0, 1};
    EXPECT_THROW(potential.local_r(disjoint, v, v), std::invalid_argument);
    EXPECT_EQ(v[0], 9);
}

TEST(NcPotential, UpfLocalTableInterpolationPreservesSamplesAndDerivative) {
    const auto path = std::filesystem::path(__FILE__)
                          .parent_path()
                          .parent_path()
                          .parent_path()
                      / "data/pw/Si_lda_nc.upf";
    cpu::pw::UpfPseudopotential potential(path.string());
    const auto view = potential.projectors();
    std::vector<AccReal> v(view.radius.size()), dv(v.size());
    potential.local_r(view.radius, v, dv);
    // Snapshot is in physical Ha units and owns its data independently.
    cpu::pw::internal::NcPotentialData snapshot(potential);
    EXPECT_EQ(snapshot.v_loc, v);
    EXPECT_EQ(potential.atomic_number(), 14);
    EXPECT_EQ(potential.valence(), 4);
    for (AccReal r : {AccReal(.31), AccReal(.75), AccReal(1.15)}) {
        const std::array<AccReal, 3> points{r - 1e-7, r, r + 1e-7};
        std::array<AccReal, 3> vp, dp;
        potential.local_r(points, vp, dp);
        EXPECT_NEAR(dp[1], (vp[2] - vp[0]) / 2e-7, 2e-7);
    }
    const AccReal r = view.radius.back();
    const std::array<AccReal, 3> boundary{r - 1e-7, r, r + 1e-7};
    std::array<AccReal, 3> vb, db;
    potential.local_r(boundary, vb, db);
    EXPECT_NEAR(db[1], (vb[2] - vb[0]) / 2e-7, 2e-7);
    const AccReal h = r - view.radius[view.radius.size() - 2];
    const std::array<AccReal, 1> tail{r + 2 * h};
    std::array<AccReal, 1> vt, dt;
    potential.local_r(tail, vt, dt);
    EXPECT_DOUBLE_EQ(vt[0], -AccReal(potential.valence()) / tail[0]);
}
