/// @file tests/cpu/electronic/test_aspc_predictor.cpp
/// @brief Unit tests: ASPCPredictor / ASPCPredictorCplx — Kolafa
///        always-stable predictor-corrector density extrapolation.
///
/// Mirrors include/atomiverse/cpu/electronic/aspc_predictor.h — pure dense
/// algebra on the predictor class (record / predict / correct / reset), no SCF
/// method driven here. The RHF-with-predictor integration guard lives in
/// tests/cpu/scf/test_scf_evaluator.cpp; the DIIS / level-shift accelerators in
/// tests/cpu/scf/test_accelerators.cpp.

#include <complex>

#include <gtest/gtest.h>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "atomiverse/cpu/electronic/aspc_predictor.h"
#include "atomiverse/cpu/linalg/matrix.h"

using ATV::cpu::electronic::ASPCPredictor;
using ATV::cpu::electronic::ASPCPredictorCplx;
using ATV::cpu::linalg::CplxMatrix;

// ── ASPC predictor (real Γ density) ──────────────────────────────────

class ASPCPredictorTest : public ::testing::Test {};

TEST_F(ASPCPredictorTest, BasicFunctionality) {
    ASPCPredictor pred(3);

    xt::xtensor<double, 2> P1 = {{1.0, 0.0}, {0.0, 1.0}};
    xt::xtensor<double, 2> P2 = {{1.1, 0.0}, {0.0, 1.1}};
    xt::xtensor<double, 2> P3 = {{1.2, 0.0}, {0.0, 1.2}};
    xt::xtensor<double, 2> P4 = {{1.3, 0.0}, {0.0, 1.3}};

    xt::xtensor<double, 2> P_pred;

    pred.record(P1);
    EXPECT_FALSE(pred.predict(P_pred));
    pred.record(P2);
    EXPECT_FALSE(pred.predict(P_pred));
    pred.record(P3);
    EXPECT_FALSE(pred.predict(P_pred));
    pred.record(P4);
    EXPECT_TRUE(pred.predict(P_pred));

    EXPECT_NEAR(static_cast<double>(P_pred(0, 0)), 1.4, 0.01);
}

TEST_F(ASPCPredictorTest, CorrectorBlending) {
    ASPCPredictor pred(3);

    xt::xtensor<double, 2> P_scf = {{2.0}};
    xt::xtensor<double, 2> P_pred = {{1.8}};
    xt::xtensor<double, 2> P_corr;

    pred.correct(P_scf, P_pred, P_corr);

    EXPECT_NEAR(static_cast<double>(P_corr(0, 0)),
                5.0 / 9.0 * 2.0 + 4.0 / 9.0 * 1.8, 1e-8);
}

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

// ── ASPC complex (multi-k P_k) ───────────────────────────────────────

TEST(ASPCCplx, MatchesRealOnRealValuedMatrices) {
    // Same Kolafa weights as real ASPC when imag parts are zero.
    ASPCPredictor real_aspc(2);
    ASPCPredictorCplx cplx_aspc(2);
    xt::xtensor<double, 2> R1 = {{1.0, 0.1}, {0.1, 1.0}};
    xt::xtensor<double, 2> R2 = {{1.1, 0.1}, {0.1, 1.1}};
    xt::xtensor<double, 2> R3 = {{1.2, 0.1}, {0.1, 1.2}};
    using Cplx = std::complex<double>;
    CplxMatrix C1 = xt::zeros<Cplx>({2, 2});
    CplxMatrix C2 = xt::zeros<Cplx>({2, 2});
    CplxMatrix C3 = xt::zeros<Cplx>({2, 2});
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            C1(i, j) = Cplx(R1(i, j), 0.0);
            C2(i, j) = Cplx(R2(i, j), 0.0);
            C3(i, j) = Cplx(R3(i, j), 0.0);
        }
    real_aspc.record(R1);
    real_aspc.record(R2);
    real_aspc.record(R3);
    cplx_aspc.record(C1);
    cplx_aspc.record(C2);
    cplx_aspc.record(C3);
    xt::xtensor<double, 2> pred_r;
    CplxMatrix pred_c;
    ASSERT_TRUE(real_aspc.predict(pred_r));
    ASSERT_TRUE(cplx_aspc.predict(pred_c));
    EXPECT_NEAR(pred_c(0, 0).real(), pred_r(0, 0), 1e-12);
    EXPECT_NEAR(pred_c(0, 0).imag(), 0.0, 1e-12);
    EXPECT_NEAR(pred_c(1, 1).real(), pred_r(1, 1), 1e-12);

    xt::xtensor<double, 2> corr_r;
    CplxMatrix corr_c;
    real_aspc.correct(R3, pred_r, corr_r);
    cplx_aspc.correct(C3, pred_c, corr_c);
    EXPECT_NEAR(corr_c(0, 0).real(), corr_r(0, 0), 1e-12);
}

TEST(ASPCCplx, ExtrapolatesComplexLinearTrend) {
    ASPCPredictorCplx aspc(2);
    using Cplx = std::complex<double>;
    auto make_P = [](Cplx v) {
        CplxMatrix P = xt::zeros<Cplx>({2, 2});
        P(0, 0) = v;
        P(1, 1) = v;
        return P;
    };
    aspc.record(make_P(Cplx(1.0, 0.5)));
    aspc.record(make_P(Cplx(1.1, 0.6)));
    aspc.record(make_P(Cplx(1.2, 0.7)));
    CplxMatrix pred;
    ASSERT_TRUE(aspc.predict(pred));
    // Linear trend +0.1 real / +0.1 imag per step → predict past last.
    EXPECT_GT(pred(0, 0).real(), 1.15);
    EXPECT_GT(pred(0, 0).imag(), 0.65);
}

TEST(ASPCCplx, PredictFalseBeforeHistory) {
    ASPCPredictorCplx aspc(3);
    CplxMatrix pred = xt::zeros<std::complex<double>>({2, 2});
    EXPECT_FALSE(aspc.predict(pred));
}
