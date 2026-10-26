/// @file tests/cpu/vibration/test_phonon_dispersion.cpp
/// @brief Unit tests for cpu::vib::PhononDispersion (Fourier ω(q)).

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/vibration/phonon_dispersion.h"
#include "atomiverse/cpu/vibration/supercell.h"

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

TEST(PhononDispersion, SupercellReplicatesColumnsOfNonSymmetricCell) {
    auto unit = mono_cubic(4.0);
    Mat3 h = Mat3::diagonal(4.0, 5.0, 6.0);
    h(0, 1) = 1.5;
    h(2, 0) = 0.8;
    unit.box.set_cell(h);
    auto sc = expand_supercell(unit, 2, 3, 1);
    ASSERT_EQ(sc.nlocal(), 6);
    const auto& hs = sc.box.cell();
    for (int d = 0; d < 3; ++d) {
        EXPECT_DOUBLE_EQ(hs(d, 0), 2 * h(d, 0));
        EXPECT_DOUBLE_EQ(hs(d, 1), 3 * h(d, 1));
        EXPECT_DOUBLE_EQ(hs(d, 2), h(d, 2));
    }
    // Enumeration order ia/ib/ic: atom 1 is b; atom 3 is a.
    EXPECT_DOUBLE_EQ(sc.span(kAtom_x)[1], 1.5);
    EXPECT_DOUBLE_EQ(sc.span(kAtom_y)[1], 5.0);
    EXPECT_DOUBLE_EQ(sc.span(kAtom_z)[1], 0.0);
    EXPECT_DOUBLE_EQ(sc.span(kAtom_x)[3], 4.0);
    EXPECT_DOUBLE_EQ(sc.span(kAtom_y)[3], 0.0);
    EXPECT_DOUBLE_EQ(sc.span(kAtom_z)[3], 0.8);
}

// Analytic monatomic nearest-neighbor chain: C(q)=2k[1−cos(2πq)].
// The raw on-site supercell block is 2k, which is NOT C(X)=4k.
TEST(PhononDispersion, RawFourierSpringChainMatrixHasTrueQDependence) {
    const auto unit = mono_cubic(4.0);
    constexpr int cells = 4;
    constexpr double spring = 7.5; // eV/Å²; output must remain unweighted
    xt::xtensor<double, 2> H = xt::zeros<double>({12, 12});
    for (int cell = 0; cell < cells; ++cell)
        for (int axis = 0; axis < 3; ++axis) {
            const int a = 3 * cell + axis, b = 3 * ((cell + 1) % cells) + axis;
            H(a, a) += spring;
            H(b, b) += spring;
            H(a, b) -= spring;
            H(b, a) -= spring;
        }
    PhononDispersion path;
    for (double q : {0.0, 0.25, 0.5}) {
        double asym = 0;
        const auto C =
            path.force_constants_at_q(H, unit, cells, 1, 1, {q, 0, 0}, &asym);
        const double want =
            2 * spring * (1 - std::cos(2 * std::numbers::pi * q));
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                EXPECT_NEAR(C(i, j).real(), i == j ? want : 0.0, 2e-13);
                EXPECT_NEAR(C(i, j).imag(), 0.0, 2e-13);
            }
        EXPECT_NEAR(asym, 0.0, 2e-13);
    }
}

// Alternating A/B chain: C_AB(q)=−k(1+exp(−2πiq)),
// C_BA=conj(C_AB), while C_AA=C_BB=2k. This also checks the imaginary
// observable, which cannot be obtained from the top-left primitive block.
TEST(PhononDispersion, RawFourierAlternatingChainRetainsComplexPhase) {
    cpu::Frame unit(2);
    unit.span(kAtom_type)[0] = unit.span(kAtom_type)[1] = 14;
    unit.box = cpu::Box::cubic(4.0);
    constexpr int cells = 4;
    constexpr double spring = 3.25;
    xt::xtensor<double, 2> H = xt::zeros<double>({24, 24});
    auto bond = [&](int a, int b) {
        H(3 * a, 3 * a) += spring;
        H(3 * b, 3 * b) += spring;
        H(3 * a, 3 * b) -= spring;
        H(3 * b, 3 * a) -= spring;
    };
    for (int cell = 0; cell < cells; ++cell) {
        bond(2 * cell, 2 * cell + 1);
        bond(2 * cell + 1, 2 * ((cell + 1) % cells));
    }
    PhononDispersion path;
    const auto C =
        path.force_constants_at_q(H, unit, cells, 1, 1, {0.25, 0, 0});
    EXPECT_NEAR(C(0, 0).real(), 2 * spring, 2e-13);
    EXPECT_NEAR(C(3, 3).real(), 2 * spring, 2e-13);
    EXPECT_NEAR(std::abs(C(0, 3) - std::complex<double>(-spring, spring)), 0.0,
                2e-13);
    EXPECT_NEAR(std::abs(C(3, 0) - std::conj(C(0, 3))), 0.0, 2e-13);
    // A deliberately nonsymmetric IFC proves publication is raw, with the
    // asymmetry diagnostic retaining the discarded-information magnitude.
    H(0, 3) += 0.02;
    double asym = 0;
    const auto raw =
        path.force_constants_at_q(H, unit, cells, 1, 1, {0.25, 0, 0}, &asym);
    EXPECT_NEAR((raw(0, 3) - C(0, 3)).real(), 0.02, 2e-13);
    EXPECT_NEAR(asym, 0.01, 2e-13);
}
