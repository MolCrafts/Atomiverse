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
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "atomiverse/cpu/electronic/aspc_predictor.h"
#include "atomiverse/cpu/linalg/matrix.h"

using ATV::cpu::electronic::ASPCPredictor;
using ATV::cpu::electronic::ASPCPredictorCplx;
using ATV::cpu::electronic::kolafa_aspc_weights;
using ATV::cpu::linalg::CplxMatrix;

// ── Kolafa weights ───────────────────────────────────────────────────
//
// Kolafa, J. Comput. Chem. 25, 335 (2004): order k uses the k+2 newest
// snapshots with B_j = (-1)^(j+1) j C(2k+4, k+2-j) / C(2k+2, k+1), j = 1
// (newest) … k+2. Same formula as CP2K qs_wf_history_methods.F
// (wfi_aspc_nr, nvec = k+2); the six-vector row is MetalWalls'
// src/electrode_charge.F90 constant set.

TEST(KolafaAspcWeights, MatchPublishedRows) {
    const std::vector<std::vector<double>> rows = {
        {2.0, -1.0},                                     // k = 0
        {2.5, -2.0, 0.5},                                // k = 1
        {2.8, -2.8, 1.2, -0.2},                          // k = 2
        {3.0, -24.0 / 7, 27.0 / 14, -4.0 / 7, 1.0 / 14}, // k = 3
        {22.0 / 7, -55.0 / 14, 55.0 / 21, -22.0 / 21, 5.0 / 21,
         -1.0 / 42}, // MetalWalls
    };
    for (const auto& row : rows) {
        const auto w = kolafa_aspc_weights(static_cast<int>(row.size()));
        ASSERT_EQ(w.size(), row.size());
        for (std::size_t j = 0; j < row.size(); ++j)
            EXPECT_NEAR(w[j], row[j], 1e-14)
                << "n=" << row.size() << " j=" << j;
    }
}

TEST(KolafaAspcWeights, PreserveConstantsAndLinearTrends) {
    for (int n = 1; n <= 8; ++n) {
        const auto w = kolafa_aspc_weights(n);
        double sum = 0, first_moment = 0;
        for (int j = 1; j <= n; ++j) {
            sum += w[static_cast<std::size_t>(j - 1)];
            first_moment += j * w[static_cast<std::size_t>(j - 1)];
        }
        EXPECT_NEAR(sum, 1.0, 1e-12) << "n=" << n;
        if (n >= 2)
            EXPECT_NEAR(first_moment, 0.0, 1e-12) << "n=" << n;
    }
    EXPECT_THROW(kolafa_aspc_weights(0), std::invalid_argument);
}

// ── ASPC predictor (real Γ density) ──────────────────────────────────

TEST(ASPC, OrderKNeedsKPlus2Snapshots) {
    ASPCPredictor aspc(3);
    xt::xtensor<double, 2> pred;
    for (int i = 0; i < 4; ++i) {
        aspc.record(xt::xtensor<double, 2>{{1.0 + 0.1 * i}});
        EXPECT_FALSE(aspc.predict(pred)) << "after " << i + 1 << " snapshots";
    }
    aspc.record(xt::xtensor<double, 2>{{1.4}});
    ASSERT_TRUE(aspc.predict(pred));
    EXPECT_EQ(aspc.history_size(), 5);
}

TEST(ASPC, PredictionAppliesKolafaWeightsNewestFirst) {
    // Impulse history: only snapshot j (1 = newest) is nonzero, so the
    // prediction reads back B_j.
    constexpr int k = 2;
    const auto w = kolafa_aspc_weights(k + 2);
    for (int j = 1; j <= k + 2; ++j) {
        ASPCPredictor aspc(k);
        for (int age = k + 2; age >= 1; --age)
            aspc.record(xt::xtensor<double, 2>{{age == j ? 1.0 : 0.0}});
        xt::xtensor<double, 2> pred;
        ASSERT_TRUE(aspc.predict(pred));
        EXPECT_NEAR(pred(0, 0), w[static_cast<std::size_t>(j - 1)], 1e-14)
            << "j=" << j;
    }
}

TEST(ASPC, LinearTrendIsExtrapolatedExactly) {
    ASPCPredictor aspc(3);
    for (int i = 0; i < 5; ++i)
        aspc.record(
            xt::xtensor<double, 2>{{1.0 + 0.1 * i, 0.0}, {0.0, 2.0 - 0.2 * i}});
    xt::xtensor<double, 2> pred;
    ASSERT_TRUE(aspc.predict(pred));
    EXPECT_NEAR(pred(0, 0), 1.5, 1e-12);
    EXPECT_NEAR(pred(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(pred(0, 1), 0.0, 1e-15);
}

TEST(ASPC, HistoryKeepsOnlyKPlus2Snapshots) {
    ASPCPredictor aspc(1);
    for (int i = 0; i < 10; ++i)
        aspc.record(xt::xtensor<double, 2>{{double(i)}});
    EXPECT_EQ(aspc.history_size(), 3);
}

TEST(ASPC, PredictReturnsFalseBeforeHistory) {
    ASPCPredictor aspc(3);
    xt::xtensor<double, 2> pred = xt::zeros<double>({2, 2});
    EXPECT_FALSE(aspc.predict(pred))
        << "predict() should return false with empty history";
}

TEST(ASPC, CorrectBlends) {
    ASPCPredictor aspc(3);
    // k=3: w_scf = (k+2)/(2k+3) = 5/9, w_pred = (k+1)/(2k+3) = 4/9
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
    xt::xtensor<double, 2> R4 = {{1.35, 0.1}, {0.1, 1.3}};
    using Cplx = std::complex<double>;
    for (const auto* R : {&R1, &R2, &R3, &R4}) {
        CplxMatrix C = xt::zeros<Cplx>({2, 2});
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                C(i, j) = Cplx((*R)(i, j), 0.0);
        real_aspc.record(*R);
        cplx_aspc.record(C);
    }
    xt::xtensor<double, 2> pred_r;
    CplxMatrix pred_c;
    ASSERT_TRUE(real_aspc.predict(pred_r));
    ASSERT_TRUE(cplx_aspc.predict(pred_c));
    EXPECT_NEAR(pred_c(0, 0).real(), pred_r(0, 0), 1e-12);
    EXPECT_NEAR(pred_c(0, 0).imag(), 0.0, 1e-12);
    EXPECT_NEAR(pred_c(1, 1).real(), pred_r(1, 1), 1e-12);

    xt::xtensor<double, 2> corr_r;
    CplxMatrix corr_c;
    CplxMatrix C4 = xt::zeros<Cplx>({2, 2});
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C4(i, j) = Cplx(R4(i, j), 0.0);
    real_aspc.correct(R4, pred_r, corr_r);
    cplx_aspc.correct(C4, pred_c, corr_c);
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
    aspc.record(make_P(Cplx(1.3, 0.8)));
    CplxMatrix pred;
    ASSERT_TRUE(aspc.predict(pred));
    // Kolafa weights reproduce a linear trend exactly.
    EXPECT_NEAR(pred(0, 0).real(), 1.4, 1e-12);
    EXPECT_NEAR(pred(0, 0).imag(), 0.9, 1e-12);
}

TEST(ASPCCplx, PredictFalseBeforeHistory) {
    ASPCPredictorCplx aspc(3);
    CplxMatrix pred = xt::zeros<std::complex<double>>({2, 2});
    EXPECT_FALSE(aspc.predict(pred));
}
