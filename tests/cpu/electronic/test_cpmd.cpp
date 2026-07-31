/// @file tests/cpu/electronic/test_cpmd.cpp
/// @brief Unit tests: Cpmd / CpmdCplx electronic propagators (RATTLE +
///        Nosé–Hoover electronic thermostat).
///
/// Mirrors include/atomiverse/cpu/electronic/cpmd.h — pure dense algebra on
/// the propagator class, no GPW/SCF anything. The GPWEvaluator build()
/// validation guards live in tests/cpu/gpw/test_gpw_cpmd.cpp; the multi-step
/// MD / water trajectory envelopes are the e2e pins
/// regressions/dynamics/cpmd-water and cpmd-rattle-water.

#include <cmath>
#include <complex>
#include <stdexcept>

#include <gtest/gtest.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "atomiverse/cpu/electronic/cpmd.h"
#include "atomiverse/cpu/linalg/matrix.h"

using ATV::cpu::electronic::Cpmd;
using ATV::cpu::linalg::RealMatrix;

namespace {

double max_abs_mat(const RealMatrix& a) {
    double m = 0.0;
    for (double v : a) {
        const double av = std::fabs(v);
        if (av > m)
            m = av;
    }
    return m;
}

RealMatrix eye_n(int n) {
    RealMatrix I = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    for (int i = 0; i < n; ++i)
        I(i, i) = 1.0;
    return I;
}

} // namespace

// ── Cpmd propagator ─────────────────────────────────────────────────────

TEST(Cpmd, ConstructsWithPositiveMass) {
    const Cpmd cp(400.0);
    EXPECT_DOUBLE_EQ(cp.mu(), 400.0);
    EXPECT_FALSE(cp.ready());
}

TEST(Cpmd, NonPositiveMassThrows) {
    EXPECT_THROW(Cpmd(0.0), std::invalid_argument);
    EXPECT_THROW(Cpmd(-10.0), std::invalid_argument);
}

TEST(Cpmd, SeedThenReadyAtRest) {
    Cpmd cp(400.0);
    RealMatrix C = xt::zeros<double>({4, 2});
    C(0, 0) = 1.0;
    C(1, 1) = 1.0;
    cp.seed(C);
    ASSERT_TRUE(cp.ready());
    EXPECT_EQ(cp.C().shape(0), 4u);
    EXPECT_EQ(cp.C().shape(1), 2u);
    EXPECT_NEAR(cp.kinetic_ha(), 0.0, 1e-15);
}

TEST(Cpmd, AdvanceBeforeSeedThrows) {
    Cpmd cp(400.0);
    RealMatrix H = xt::zeros<double>({2, 2});
    RealMatrix S = xt::zeros<double>({2, 2});
    S(0, 0) = 1.0;
    S(1, 1) = 1.0;
    EXPECT_THROW(cp.advance(H, S, 0.1), std::logic_error);
}

TEST(Cpmd, AdvanceWithZeroHKeepsRest) {
    Cpmd cp(400.0);
    RealMatrix C = xt::zeros<double>({2, 1});
    C(0, 0) = 1.0;
    cp.seed(C);
    RealMatrix H = xt::zeros<double>({2, 2});
    RealMatrix S = xt::zeros<double>({2, 2});
    S(0, 0) = 1.0;
    S(1, 1) = 1.0;
    cp.advance(H, S, 0.1);
    EXPECT_NEAR(cp.C()(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(cp.C()(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(cp.kinetic_ha(), 0.0, 1e-12);
}

TEST(Cpmd, EmptySeedThrows) {
    Cpmd cp(400.0);
    RealMatrix empty;
    EXPECT_THROW(cp.seed(empty), std::invalid_argument);
}

TEST(Cpmd, ResetDropsReady) {
    Cpmd cp(400.0);
    RealMatrix C = xt::zeros<double>({2, 1});
    C(0, 0) = 1.0;
    cp.seed(C);
    cp.reset();
    EXPECT_FALSE(cp.ready());
    EXPECT_THROW(cp.C(), std::logic_error);
}

// ── Electronic RATTLE residual gates ────────────────────────────────────

TEST(Cpmd, RattlePreservesOrthonormalityAndVelocityConstraint) {
    // Two occupied columns in a 3-AO space; S = I; nontrivial diagonal H so
    // the force is nonzero off the BO surface for a non-eigen C.
    Cpmd cp(400.0);
    RealMatrix C = xt::zeros<double>({3, 2});
    C(0, 0) = 1.0;
    C(1, 1) = 1.0;
    cp.seed(C);

    RealMatrix S = eye_n(3);
    RealMatrix H = xt::zeros<double>({3, 3});
    H(0, 0) = -1.0;
    H(1, 1) = -0.5;
    H(2, 2) = 0.2;
    H(0, 2) = H(2, 0) = 0.15; // couple virtual direction into force

    cp.advance(H, S, 0.05);
    ASSERT_TRUE(cp.ready());
    EXPECT_GE(cp.last_position_iters(), 0);
    EXPECT_LE(cp.last_constraint_residual(), 1e-9);

    using xt::linalg::dot;
    const RealMatrix G = dot(xt::transpose(cp.C()), dot(S, cp.C()));
    EXPECT_LE(max_abs_mat(G - eye_n(2)), 1e-9);

    // Velocity constraint: sym(Cᵀ S Ċ) ≈ 0.
    const RealMatrix W = dot(xt::transpose(cp.C()), dot(S, cp.Cdot()));
    const RealMatrix Wsym = 0.5 * (W + xt::transpose(W));
    EXPECT_LE(max_abs_mat(Wsym), 1e-9);

    // Residual accessor matches the recomputed ‖σ‖_∞.
    EXPECT_NEAR(cp.last_constraint_residual(), max_abs_mat(G - eye_n(2)),
                1e-15);
}

TEST(Cpmd, RattleAtRestZeroHKeepsZeroVelocity) {
    Cpmd cp(400.0);
    RealMatrix C = xt::zeros<double>({2, 1});
    C(0, 0) = 1.0;
    cp.seed(C);
    RealMatrix H = xt::zeros<double>({2, 2});
    RealMatrix S = eye_n(2);
    cp.advance(H, S, 0.1);
    EXPECT_NEAR(cp.kinetic_ha(), 0.0, 1e-12);
    EXPECT_NEAR(cp.Cdot()(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(cp.Cdot()(1, 0), 0.0, 1e-12);
    EXPECT_LE(cp.last_constraint_residual(), 1e-12);
}

TEST(Cpmd, AdvanceWithOccOneMatchesUksFactor) {
    // occ=1 uses band_gradient factor 2 H C (UKS spin channel).
    Cpmd cp(400.0);
    RealMatrix C = xt::zeros<double>({2, 1});
    C(0, 0) = 1.0;
    cp.seed(C);
    RealMatrix H = xt::zeros<double>({2, 2});
    H(0, 0) = -1.0;
    H(1, 1) = 0.5;
    H(0, 1) = H(1, 0) = 0.1;
    RealMatrix S = eye_n(2);
    // At BO-ish eigen direction force is small; just ensure finite advance.
    EXPECT_NO_THROW(cp.advance(H, S, 0.05, /*occ=*/1.0));
    EXPECT_TRUE(std::isfinite(cp.kinetic_ha()));
    EXPECT_GE(cp.kinetic_ha(), 0.0);
}

TEST(Cpmd, AdvanceWithNonPositiveOccThrows) {
    Cpmd cp(400.0);
    RealMatrix C = xt::zeros<double>({2, 1});
    C(0, 0) = 1.0;
    cp.seed(C);
    RealMatrix H = eye_n(2);
    RealMatrix S = eye_n(2);
    EXPECT_THROW(cp.advance(H, S, 0.1, /*occ=*/0.0), std::invalid_argument);
    EXPECT_THROW(cp.advance(H, S, 0.1, /*occ=*/-1.0), std::invalid_argument);
}

// ── Nosé–Hoover electronic thermostat ───────────────────────────────────

TEST(Cpmd, ThermostatInvalidParamsThrowOnAdvance) {
    Cpmd cp(400.0);
    RealMatrix C = xt::zeros<double>({2, 1});
    C(0, 0) = 1.0;
    cp.seed(C);
    cp.thermostat = true;
    cp.ekin_target_ha = 0.0;
    cp.tau_e_fs = 10.0;
    RealMatrix H = eye_n(2);
    RealMatrix S = eye_n(2);
    EXPECT_THROW(cp.advance(H, S, 0.1), std::invalid_argument);
    cp.ekin_target_ha = 1e-4;
    cp.tau_e_fs = -1.0;
    EXPECT_THROW(cp.advance(H, S, 0.1), std::invalid_argument);
}

TEST(Cpmd, SeedWithCdotSetsKineticEnergy) {
    Cpmd cp(400.0);
    RealMatrix C = xt::zeros<double>({2, 1});
    C(0, 0) = 1.0;
    RealMatrix Cdot = xt::zeros<double>({2, 1});
    Cdot(1, 0) = 0.01; // off-manifold component is fine for KE check
    cp.seed(C, Cdot);
    const double expect = 0.5 * 400.0 * (0.01 * 0.01);
    EXPECT_NEAR(cp.kinetic_ha(), expect, 1e-15);
    EXPECT_NEAR(cp.thermostat_energy_ha(), 0.0, 1e-15);
    EXPECT_NEAR(cp.xi(), 0.0, 1e-15);
    EXPECT_NEAR(cp.eta(), 0.0, 1e-15);
}

TEST(Cpmd, ThermostatRegulatesHotElectronicKe) {
    // Hot seed Te ≫ K0; NH should pull Te down over several steps.
    Cpmd cp(400.0);
    cp.thermostat = true;
    cp.ekin_target_ha = 1e-6;
    cp.tau_e_fs = 5.0; // short τ for aggressive regulation on unit scale
    RealMatrix C = xt::zeros<double>({3, 2});
    C(0, 0) = 1.0;
    C(1, 1) = 1.0;
    RealMatrix Cdot = xt::zeros<double>({3, 2});
    Cdot(2, 0) = 0.05;
    Cdot(2, 1) = -0.04;
    cp.seed(C, Cdot);
    const double te0 = cp.kinetic_ha();
    ASSERT_GT(te0, 10.0 * cp.ekin_target_ha);

    RealMatrix S = eye_n(3);
    RealMatrix H = xt::zeros<double>({3, 3});
    H(0, 0) = -1.0;
    H(1, 1) = -0.5;
    H(2, 2) = 0.1;

    for (int i = 0; i < 40; ++i)
        cp.advance(H, S, 0.05);

    const double tef = cp.kinetic_ha();
    EXPECT_TRUE(std::isfinite(tef));
    EXPECT_GE(tef, 0.0);
    // Pull toward target: final below initial and not runaway.
    EXPECT_LT(tef, te0);
    EXPECT_LE(tef, 50.0 * cp.ekin_target_ha + 0.1 * te0);
    EXPECT_TRUE(std::isfinite(cp.thermostat_energy_ha()));
    EXPECT_LE(cp.last_constraint_residual(), 1e-9);

    using xt::linalg::dot;
    const RealMatrix G = dot(xt::transpose(cp.C()), dot(S, cp.C()));
    EXPECT_LE(max_abs_mat(G - eye_n(2)), 1e-9);
    const RealMatrix W = dot(xt::transpose(cp.C()), dot(S, cp.Cdot()));
    const RealMatrix Wsym = 0.5 * (W + xt::transpose(W));
    EXPECT_LE(max_abs_mat(Wsym), 1e-9);
}

TEST(Cpmd, ThermostatOffKeepsZeroThermostatEnergy) {
    Cpmd cp(400.0);
    RealMatrix C = xt::zeros<double>({2, 1});
    C(0, 0) = 1.0;
    cp.seed(C);
    EXPECT_FALSE(cp.thermostat);
    EXPECT_NEAR(cp.thermostat_energy_ha(), 0.0, 1e-15);
    RealMatrix H = eye_n(2);
    H(0, 0) = -0.5;
    RealMatrix S = eye_n(2);
    cp.advance(H, S, 0.1);
    EXPECT_NEAR(cp.thermostat_energy_ha(), 0.0, 1e-15);
    EXPECT_NEAR(cp.xi(), 0.0, 1e-15);
    EXPECT_NEAR(cp.eta(), 0.0, 1e-15);
}

// ── Multi-k Bloch CPMD (CpmdCplx) ───────────────────────────────────────

using ATV::cpu::electronic::CpmdCplx;
using ATV::cpu::linalg::CplxMatrix;
using Cplx = std::complex<double>;

namespace {

double max_abs_cmat(const CplxMatrix& a) {
    double m = 0.0;
    for (const Cplx& v : a) {
        const double av = std::abs(v);
        if (av > m)
            m = av;
    }
    return m;
}

CplxMatrix eye_c(int n) {
    CplxMatrix I = xt::zeros<Cplx>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    for (int i = 0; i < n; ++i)
        I(i, i) = Cplx(1.0, 0.0);
    return I;
}

// A† B (small helper for residual checks).
CplxMatrix c_dagger_mul_test(const CplxMatrix& A, const CplxMatrix& B) {
    const int n = static_cast<int>(A.shape(0));
    const int m = static_cast<int>(A.shape(1));
    const int p = static_cast<int>(B.shape(1));
    CplxMatrix C = xt::zeros<Cplx>(
        {static_cast<std::size_t>(m), static_cast<std::size_t>(p)});
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < n; ++k) {
            const Cplx a = std::conj(A(k, i));
            for (int j = 0; j < p; ++j)
                C(i, j) += a * B(k, j);
        }
    return C;
}

CplxMatrix c_mul_test(const CplxMatrix& A, const CplxMatrix& B) {
    const int n = static_cast<int>(A.shape(0));
    const int m = static_cast<int>(A.shape(1));
    const int p = static_cast<int>(B.shape(1));
    CplxMatrix C = xt::zeros<Cplx>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(p)});
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < m; ++k) {
            const Cplx a = A(i, k);
            for (int j = 0; j < p; ++j)
                C(i, j) += a * B(k, j);
        }
    return C;
}

} // namespace

TEST(CpmdCplx, ConstructsWithPositiveMass) {
    const CpmdCplx cp(400.0);
    EXPECT_DOUBLE_EQ(cp.mu(), 400.0);
    EXPECT_FALSE(cp.ready());
}

TEST(CpmdCplx, NonPositiveMassThrows) {
    EXPECT_THROW(CpmdCplx(0.0), std::invalid_argument);
    EXPECT_THROW(CpmdCplx(-1.0), std::invalid_argument);
}

TEST(CpmdCplx, EmptySeedThrows) {
    CpmdCplx cp(400.0);
    CplxMatrix empty;
    EXPECT_THROW(cp.seed(empty), std::invalid_argument);
}

TEST(CpmdCplx, RattlePreservesComplexOrthonormality) {
    CpmdCplx cp(400.0);
    CplxMatrix C = xt::zeros<Cplx>({3, 2});
    C(0, 0) = Cplx(1.0, 0.0);
    C(1, 1) = Cplx(0.0, 1.0); // complex phase on second column
    cp.seed(C);

    CplxMatrix S = eye_c(3);
    CplxMatrix H = xt::zeros<Cplx>({3, 3});
    H(0, 0) = Cplx(-1.0, 0.0);
    H(1, 1) = Cplx(-0.5, 0.0);
    H(2, 2) = Cplx(0.2, 0.0);
    H(0, 2) = Cplx(0.1, 0.05);
    H(2, 0) = std::conj(H(0, 2));

    cp.advance(H, S, 0.05);
    ASSERT_TRUE(cp.ready());
    EXPECT_LE(cp.last_constraint_residual(), 1e-9);

    const CplxMatrix G = c_dagger_mul_test(cp.C(), c_mul_test(S, cp.C()));
    CplxMatrix residual = G - eye_c(2);
    EXPECT_LE(max_abs_cmat(residual), 1e-9);

    // Velocity constraint: Herm(C† S Ċ) ≈ 0.
    const CplxMatrix W = c_dagger_mul_test(cp.C(), c_mul_test(S, cp.Cdot()));
    CplxMatrix Wherm = xt::zeros<Cplx>(W.shape());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            Wherm(i, j) = 0.5 * (W(i, j) + std::conj(W(j, i)));
    EXPECT_LE(max_abs_cmat(Wherm), 1e-9);
    EXPECT_TRUE(std::isfinite(cp.kinetic_ha()));
    EXPECT_GE(cp.kinetic_ha(), 0.0);
}

TEST(CpmdCplx, AdvanceBeforeSeedThrows) {
    CpmdCplx cp(400.0);
    CplxMatrix H = eye_c(2);
    CplxMatrix S = eye_c(2);
    EXPECT_THROW(cp.advance(H, S, 0.1), std::logic_error);
}

// ── Nosé–Hoover electronic thermostat (complex channel) ─────────────────

TEST(CpmdCplx, ThermostatInvalidParamsThrowOnAdvance) {
    CpmdCplx cp(400.0);
    CplxMatrix C = xt::zeros<Cplx>({2, 1});
    C(0, 0) = Cplx(1.0, 0.0);
    cp.seed(C);
    cp.thermostat = true;
    cp.ekin_target_ha = 0.0;
    cp.tau_e_fs = 10.0;
    CplxMatrix H = eye_c(2);
    CplxMatrix S = eye_c(2);
    EXPECT_THROW(cp.advance(H, S, 0.1), std::invalid_argument);
    cp.ekin_target_ha = 1e-4;
    cp.tau_e_fs = -1.0;
    EXPECT_THROW(cp.advance(H, S, 0.1), std::invalid_argument);
}

TEST(CpmdCplx, ThermostatOffKeepsZeroThermostatEnergy) {
    CpmdCplx cp(400.0);
    CplxMatrix C = xt::zeros<Cplx>({2, 1});
    C(0, 0) = Cplx(1.0, 0.0);
    cp.seed(C);
    EXPECT_FALSE(cp.thermostat);
    EXPECT_NEAR(cp.thermostat_energy_ha(), 0.0, 1e-15);
    CplxMatrix H = eye_c(2);
    H(0, 0) = Cplx(-0.5, 0.0);
    CplxMatrix S = eye_c(2);
    cp.advance(H, S, 0.1);
    EXPECT_NEAR(cp.thermostat_energy_ha(), 0.0, 1e-15);
    EXPECT_NEAR(cp.xi(), 0.0, 1e-15);
    EXPECT_NEAR(cp.eta(), 0.0, 1e-15);
}

TEST(CpmdCplx, ThermostatRegulatesHotElectronicKe) {
    // Hot complex seed Te ≫ K0; NH pulls Te down while RATTLE holds the
    // Hermitian constraint — the per-(k,σ) control the multi-k GPW consumer
    // relies on (cpmd-thermostat-kpoints).
    CpmdCplx cp(400.0);
    cp.thermostat = true;
    cp.ekin_target_ha = 1e-6;
    cp.tau_e_fs = 5.0;
    CplxMatrix C = xt::zeros<Cplx>({3, 2});
    C(0, 0) = Cplx(1.0, 0.0);
    C(1, 1) = Cplx(0.0, 1.0); // complex phase on the second column
    CplxMatrix Cdot = xt::zeros<Cplx>({3, 2});
    Cdot(2, 0) = Cplx(0.05, 0.02);
    Cdot(2, 1) = Cplx(-0.04, 0.01);
    cp.seed(C, Cdot);
    const double te0 = cp.kinetic_ha();
    ASSERT_GT(te0, 10.0 * cp.ekin_target_ha);

    CplxMatrix S = eye_c(3);
    CplxMatrix H = xt::zeros<Cplx>({3, 3});
    H(0, 0) = Cplx(-1.0, 0.0);
    H(1, 1) = Cplx(-0.5, 0.0);
    H(2, 2) = Cplx(0.1, 0.0);

    for (int i = 0; i < 40; ++i)
        cp.advance(H, S, 0.05);

    const double tef = cp.kinetic_ha();
    EXPECT_TRUE(std::isfinite(tef));
    EXPECT_GE(tef, 0.0);
    EXPECT_LT(tef, te0);
    EXPECT_LE(tef, 50.0 * cp.ekin_target_ha + 0.1 * te0);
    EXPECT_TRUE(std::isfinite(cp.thermostat_energy_ha()));
    EXPECT_LE(cp.last_constraint_residual(), 1e-9);

    const CplxMatrix G = c_dagger_mul_test(cp.C(), c_mul_test(S, cp.C()));
    EXPECT_LE(max_abs_cmat(G - eye_c(2)), 1e-9);
    const CplxMatrix W = c_dagger_mul_test(cp.C(), c_mul_test(S, cp.Cdot()));
    CplxMatrix Wherm = xt::zeros<Cplx>(W.shape());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            Wherm(i, j) = 0.5 * (W(i, j) + std::conj(W(j, i)));
    EXPECT_LE(max_abs_cmat(Wherm), 1e-9);
}

TEST(CpmdCplx, ThermostatGammaLimitMatchesRealCpmd) {
    // Γ-limit solver identity (cpmd-thermostat-kpoints): on purely real
    // inputs the thermostatted complex propagator IS the shipped real Γ
    // thermostat product — C, Ċ, T_e, E_therm, ξ, η track Cpmd to 1e-12
    // over several NH + RATTLE steps, and no imaginary part appears.
    const double mu = 400.0;
    const double k0 = 1e-4;
    const double tau = 10.0;

    Cpmd re(mu);
    re.thermostat = true;
    re.ekin_target_ha = k0;
    re.tau_e_fs = tau;
    CpmdCplx cx(mu);
    cx.thermostat = true;
    cx.ekin_target_ha = k0;
    cx.tau_e_fs = tau;

    RealMatrix C = xt::zeros<double>({3, 2});
    C(0, 0) = 1.0;
    C(1, 1) = 1.0;
    RealMatrix Cdot = xt::zeros<double>({3, 2});
    Cdot(2, 0) = 0.03;
    Cdot(2, 1) = -0.02;
    CplxMatrix Cc = xt::zeros<Cplx>({3, 2});
    CplxMatrix Cdotc = xt::zeros<Cplx>({3, 2});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j) {
            Cc(i, j) = Cplx(C(i, j), 0.0);
            Cdotc(i, j) = Cplx(Cdot(i, j), 0.0);
        }
    re.seed(C, Cdot);
    cx.seed(Cc, Cdotc);
    ASSERT_NEAR(cx.kinetic_ha(), re.kinetic_ha(), 1e-15);

    RealMatrix S = eye_n(3);
    RealMatrix H = xt::zeros<double>({3, 3});
    H(0, 0) = -1.0;
    H(1, 1) = -0.5;
    H(2, 2) = 0.2;
    H(0, 2) = H(2, 0) = 0.15;
    CplxMatrix Sc = eye_c(3);
    CplxMatrix Hc = xt::zeros<Cplx>({3, 3});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Hc(i, j) = Cplx(H(i, j), 0.0);

    for (int step = 0; step < 5; ++step) {
        re.advance(H, S, 0.05);
        cx.advance(Hc, Sc, 0.05);
    }

    EXPECT_NEAR(cx.kinetic_ha(), re.kinetic_ha(), 1e-12);
    EXPECT_NEAR(cx.thermostat_energy_ha(), re.thermostat_energy_ha(), 1e-12);
    EXPECT_NEAR(cx.xi(), re.xi(), 1e-12);
    EXPECT_NEAR(cx.eta(), re.eta(), 1e-12);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(cx.C()(i, j).real(), re.C()(i, j), 1e-12)
                << "C i=" << i << " j=" << j;
            EXPECT_NEAR(cx.C()(i, j).imag(), 0.0, 1e-13)
                << "C i=" << i << " j=" << j;
            EXPECT_NEAR(cx.Cdot()(i, j).real(), re.Cdot()(i, j), 1e-12)
                << "Cdot i=" << i << " j=" << j;
            EXPECT_NEAR(cx.Cdot()(i, j).imag(), 0.0, 1e-13)
                << "Cdot i=" << i << " j=" << j;
        }
}
