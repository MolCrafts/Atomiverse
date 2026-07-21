/// @file tests/cpu/scf/test_accelerators.cpp
/// @brief Isolated unit tests for DIIS extrapolator and ASPC predictor.
///
/// These SCF accelerators are otherwise exercised only transitively through
/// the concrete methods' SCF loop; this file adds direct, isolated
/// verification of their internal mechanics.

#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/scf/aspc_predictor.h"
#include "atomiverse/cpu/scf/internal/diis_mixer.h"
#include "atomiverse/cpu/scf/internal/level_shift.h"

using namespace ATV::cpu::scf;
using ATV::cpu::scf::internal::DIISMixer;

// ── DIIS (internal::DIISMixer) ───────────────────────────────────────

TEST(DIIS, SingleEntry_ReturnsSameMatrix) {
    DIISMixer diis(6);
    RealMatrix F = {{1.0, 0.5}, {0.5, 2.0}};
    RealMatrix e = {{0.01, 0.0}, {0.0, -0.01}};
    auto result = diis.step(F, e);
    EXPECT_NEAR(result(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(result(0, 1), 0.5, 1e-12);
    EXPECT_NEAR(result(1, 1), 2.0, 1e-12);
}

TEST(DIIS, TwoEntries_WeightsSmallError) {
    DIISMixer diis(6);
    RealMatrix F1 = {{1.0, 0.0}, {0.0, 2.0}};
    RealMatrix e1 = {{0.1, 0.0}, {0.0, -0.1}};
    RealMatrix F2 = {{1.1, 0.0}, {0.0, 1.9}};
    RealMatrix e2 = {{0.01, 0.0}, {0.0, -0.01}};
    diis.step(F1, e1);               // push first pair
    auto result = diis.step(F2, e2); // push second + extrapolate
    EXPECT_NE(result(0, 0), F1(0, 0));
    EXPECT_NE(result(0, 0), F2(0, 0));
}

TEST(DIIS, DualChannel_BothExtracted) {
    DIISMixer diis(6);
    RealMatrix Fa = {{1.0}}, Fb = {{2.0}};
    RealMatrix ea = {{0.01}}, eb = {{0.02}};
    auto [ra, rb] = diis.step(Fa, ea, Fb, eb);
    EXPECT_NEAR(ra(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(rb(0, 0), 2.0, 1e-12);
}

TEST(DIIS, ResetClearsHistory) {
    DIISMixer diis(6);
    RealMatrix F = {{1.0}}, e = {{0.01}};
    diis.step(F, e);
    EXPECT_EQ(diis.size(), 1);
    diis.reset();
    EXPECT_EQ(diis.size(), 0);
}

// ── ASPC ─────────────────────────────────────────────────────────────

TEST(ASPC, PredictReturnsFalseBeforeHistory) {
    ASPCPredictor aspc(3);
    xt::xtensor<double, 2> pred = xt::zeros<double>({2, 2});
    bool ok = aspc.predict(pred);
    EXPECT_FALSE(ok) << "predict() should return false with empty history";
}

TEST(ASPC, RecordAndPredict) {
    ASPCPredictor aspc(2); // order 2 needs 3 history points
    xt::xtensor<double, 2> P1 = {{1.0, 0.0}, {0.0, 1.0}};
    xt::xtensor<double, 2> P2 = {{1.1, 0.0}, {0.0, 1.1}};
    xt::xtensor<double, 2> P3 = {{1.2, 0.0}, {0.0, 1.2}};
    aspc.record(P1);
    aspc.record(P2);
    aspc.record(P3);
    xt::xtensor<double, 2> pred = xt::zeros<double>({2, 2});
    bool ok = aspc.predict(pred);
    EXPECT_TRUE(ok)
        << "predict() should succeed with 3 history points at order 2";
    // Linear trend: each step adds 0.1 → prediction should be ≥ 1.2
    EXPECT_GT(pred(0, 0), 1.15)
        << "ASPC should extrapolate past last value for linear trend";
}

TEST(ASPC, CorrectBlends) {
    ASPCPredictor aspc(3);
    // K=3: alpha_scf = (K+2)/(2K+3) = 5/9, alpha_pred = (K+1)/(2K+3) = 4/9
    xt::xtensor<double, 2> P_scf = {{1.0, 0.0}, {0.0, 1.0}};
    xt::xtensor<double, 2> P_pred = {{2.0, 0.0}, {0.0, 2.0}};
    xt::xtensor<double, 2> P_out = xt::zeros<double>({2, 2});
    aspc.correct(P_scf, P_pred, P_out);
    double expected = (5.0 / 9.0) * 1.0 + (4.0 / 9.0) * 2.0;
    EXPECT_NEAR(P_out(0, 0), expected, 1e-12);
}

TEST(ASPC, ResetClearsHistory) {
    ASPCPredictor aspc(3);
    xt::xtensor<double, 2> P = {{5.0, 0.0}, {0.0, 5.0}};
    aspc.record(P);
    EXPECT_EQ(aspc.history_size(), 1);
    aspc.reset();
    EXPECT_EQ(aspc.history_size(), 0);
}

// ── Level shift (internal::apply_level_shift) ────────────────────────
// hf-robust-03 ac-002: spectral contract of the Saunders–Hillier shift.
// Synthetic generalized-eigenproblem fixture: pick SPD S and target
// eigenvalues ε, build C = L^{-T} Q (so CᵀSC = I) and F = S C diag(ε) Cᵀ S;
// then F C = S C ε holds by construction and the fixture is self-checking.

namespace {

struct LevelShiftFixture {
    RealMatrix F;
    RealMatrix S;
    RealMatrix D_occ; ///< C_occ C_occᵀ over the lowest n_occ orbitals.
    xt::xtensor<double, 1> eps;
    int n_occ;
};

LevelShiftFixture make_level_shift_fixture() {
    LevelShiftFixture fx;
    fx.S = {{1.0, 0.2, 0.0}, {0.2, 1.1, 0.1}, {0.0, 0.1, 0.9}};
    fx.eps = {-1.0, 0.3, 0.8};
    fx.n_occ = 1;

    // C = L^{-T} Q with S = L Lᵀ and a fixed orthogonal Q (3-4-5 Givens).
    RealMatrix L = xt::linalg::cholesky(fx.S);
    RealMatrix Q = {{0.8, 0.6, 0.0}, {-0.6, 0.8, 0.0}, {0.0, 0.0, 1.0}};
    RealMatrix Linv_t(xt::linalg::inv(xt::transpose(L)));
    RealMatrix C(xt::linalg::dot(Linv_t, Q));

    RealMatrix E = xt::zeros<double>({3, 3});
    for (int i = 0; i < 3; ++i)
        E(i, i) = fx.eps(i);
    RealMatrix SC(xt::linalg::dot(fx.S, C));
    RealMatrix SCE(xt::linalg::dot(SC, E));
    fx.F = xt::linalg::dot(SCE, xt::transpose(SC));

    auto C_occ = xt::view(C, xt::all(), xt::range(0, fx.n_occ));
    fx.D_occ = xt::linalg::dot(C_occ, xt::transpose(C_occ));
    return fx;
}

} // namespace

// Virtual eigenvalues shift by exactly λ; occupied ones are unchanged
// (tol 1e-12; generalized eigensolve verification).
TEST(LevelShift, VirtualsShiftByLambdaOccupiedFixed) {
    auto fx = make_level_shift_fixture();

    // Fixture self-check: unshifted spectrum reproduces ε.
    auto [eig0, C0] =
        ATV::cpu::linalg::diagonalize_fock_generalized(fx.F, fx.S);
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(eig0(i), fx.eps(i), 1e-12);

    const double lambda = 0.37;
    RealMatrix F_shift = ATV::cpu::scf::internal::apply_level_shift(
        fx.F, fx.S, fx.D_occ, lambda);
    auto [eig1, C1] =
        ATV::cpu::linalg::diagonalize_fock_generalized(F_shift, fx.S);

    EXPECT_NEAR(eig1(0), fx.eps(0), 1e-12) << "occupied ε must not move";
    EXPECT_NEAR(eig1(1), fx.eps(1) + lambda, 1e-12) << "virtual ε + λ";
    EXPECT_NEAR(eig1(2), fx.eps(2) + lambda, 1e-12) << "virtual ε + λ";
}

// |λ| <= 1e-4 is a strict element-wise no-op (PySCF skip rule).
TEST(LevelShift, SkipThresholdIsElementwiseIdentity) {
    auto fx = make_level_shift_fixture();
    RealMatrix F_out =
        ATV::cpu::scf::internal::apply_level_shift(fx.F, fx.S, fx.D_occ, 1e-4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(F_out(i, j), fx.F(i, j));
}

#endif // ATV_HAS_HF
