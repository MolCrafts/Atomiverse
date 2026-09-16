/// @file tests/cpu/vibration/test_phonon_dispersion.cpp
/// @brief Unit tests for cpu::vib::PhononDispersion (Fourier ω(q)).

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/vibration/phonon_dispersion.h"

using namespace ATV;
using namespace ATV::cpu::vib;

namespace {

/// @brief 1-atom cubic cell (mass via Z=14).
cpu::Frame mono_cubic(double a) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 14;
    f.span(kAtom_mass)[0] = 28.085;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(a);
    return f;
}

/// @brief Translationally invariant on-site spring IFC for a 1×1×1 monoatomic
///        cell: Φ = k * (3I − 11ᵀ) style acoustic ASR — actually for 1 atom
///        with only self terms, acoustic modes require row-sum 0, so
///        Φ_αβ = 0 for pure acoustic. Use a finite optical-like isotropic
///        spring that still has 3 zero modes via Φ = 0 (trivial).
///
///        Better: 2×1×1 supercell of 1-atom unit with nearest-neighbour
///        springs along a that obey ASR.
///
///        Unit: 1 atom. Supercell 2×1×1 → N_sc=2.
///        Φ between 0 and 1: -k on diagonal Cartesian (same α);
///        self: +k (ASR).
xt::xtensor<double, 2> nn_spring_hessian_2x1x1(double k) {
    // N_sc = 2, dim = 6
    xt::xtensor<double, 2> H = xt::zeros<double>({6, 6});
    for (int a = 0; a < 3; ++a) {
        // self
        H(a, a) = k;
        H(3 + a, 3 + a) = k;
        // cross (atom 0 ↔ atom 1)
        H(a, 3 + a) = -k;
        H(3 + a, a) = -k;
    }
    return H;
}

} // namespace

TEST(PhononDispersion, DiamondPathLength) {
    PhononDispersion path;
    path.set_diamond_path(3);
    // 5 segments × (3-1) + 1 start = 5*2 + 1 = 11 points
    EXPECT_EQ(path.qfrac.size(), 11u);
    EXPECT_NEAR(path.qfrac.front()[0], 0.0, 1e-15);
    EXPECT_NEAR(path.qfrac.front()[1], 0.0, 1e-15);
    EXPECT_NEAR(path.qfrac.front()[2], 0.0, 1e-15);
}

TEST(PhononDispersion, RejectsBadShape) {
    cpu::Frame unit = mono_cubic(5.0);
    xt::xtensor<double, 2> bad = xt::zeros<double>({3, 3});
    PhononDispersion path;
    EXPECT_THROW(path.compute(bad, unit, 2, 1, 1), std::invalid_argument);
    EXPECT_THROW(path.compute(bad, unit, 0, 1, 1), std::invalid_argument);
}

TEST(PhononDispersion, GammaAcousticFromNnSpring) {
    // 1-atom unit, 2×1×1 supercell, NN springs → acoustic branch at Γ is 0,
    // and finite frequency at the zone-boundary q = (0.5, 0, 0).
    cpu::Frame unit = mono_cubic(4.0);
    const double k = 10.0; // eV/Å²
    auto H = nn_spring_hessian_2x1x1(k);

    PhononDispersion path;
    auto wG = path.frequencies_at_q(H, unit, 2, 1, 1, {0.0, 0.0, 0.0});
    ASSERT_EQ(wG.size(), 3u);
    for (double w : wG)
        EXPECT_NEAR(w, 0.0, 1e-4) << "Gamma acoustic must vanish";

    auto wX = path.frequencies_at_q(H, unit, 2, 1, 1, {0.5, 0.0, 0.0});
    ASSERT_EQ(wX.size(), 3u);
    // All three Cartesian springs equal → three equal finite modes at X.
    for (double w : wX) {
        EXPECT_GT(w, 50.0) << "zone-boundary mode should be finite";
        EXPECT_TRUE(std::isfinite(w));
    }
    EXPECT_NEAR(wX[0], wX[1], 1e-6);
    EXPECT_NEAR(wX[1], wX[2], 1e-6);
}

TEST(PhononDispersion, DefaultPathWhenQfracEmpty) {
    cpu::Frame unit = mono_cubic(4.0);
    auto H = nn_spring_hessian_2x1x1(5.0);
    PhononDispersion path; // qfrac empty
    auto r = path.compute(H, unit, 2, 1, 1);
    EXPECT_GE(r.qfrac.size(), 5u);
    EXPECT_EQ(r.n_atoms_unit, 1);
    EXPECT_EQ(r.n_branches, 3);
    EXPECT_EQ(r.frequencies_cm.size(), r.qfrac.size());
    for (const auto& br : r.frequencies_cm) {
        ASSERT_EQ(br.size(), 3u);
        for (double w : br)
            EXPECT_TRUE(std::isfinite(w));
    }
}
