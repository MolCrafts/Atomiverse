#include "atomiverse/cpu/linalg/internal/diis_mixer.h"
#include <gtest/gtest.h>
using ATV::cpu::linalg::RealMatrix;
using ATV::cpu::linalg::internal::DIISMixer;

TEST(DIIS, CoupledSpinsUseOneCoefficientVector) {
    DIISMixer mixer(2);
    const RealMatrix first_a = {{1, 0}, {0, 1}};
    const RealMatrix first_b = {{10, 0}, {0, 10}};
    const RealMatrix second_a = {{3, 0}, {0, 3}};
    const RealMatrix second_b = {{30, 0}, {0, 30}};
    const RealMatrix e1 = {{1, 0}, {0, 0}};
    const RealMatrix e2 = {{0, 0}, {0, 2}};
    mixer.step(first_a, e1, first_b, e2);
    const auto [a, b] = mixer.step(second_a, e2, second_b, e1);
    // The combined Gram matrix is diag(5,5): both weights are 1/2.
    // Separate alpha/beta minimizations would instead use 4/5 and 1/5.
    EXPECT_NEAR(a(0, 0), 2, 1e-14);
    EXPECT_NEAR(b(0, 0), 20, 1e-14);
}

TEST(DIIS, RejectsNonpositiveHistoryDepth) {
    EXPECT_THROW(DIISMixer(0), std::invalid_argument);
    EXPECT_THROW(DIISMixer(-1), std::invalid_argument);
}

// ── DIIS (linalg::internal::DIISMixer) ───────────────────────────────────────

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

TEST(DIIS, ExtrapolatesFromIndependentResiduals) {
    DIISMixer diis(2);
    const RealMatrix H1 = {{1.0, 0.0}, {0.0, 2.0}};
    const RealMatrix H2 = {{3.0, 0.0}, {0.0, 4.0}};
    diis.step(H1, RealMatrix{{2.0, 0.0}, {0.0, 0.0}});
    const auto mixed = diis.step(H2, RealMatrix{{0.0, 0.0}, {0.0, 1.0}});
    EXPECT_NEAR(mixed(0, 0), 2.6, 1e-14);
    EXPECT_NEAR(mixed(1, 1), 3.6, 1e-14);
    diis.reset();
    EXPECT_EQ(diis.step(H1, H1)(0, 0), H1(0, 0));
}
