#include "atomiverse/cpu/gpw/internal/diis.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include <gtest/gtest.h>

using ATV::cpu::gpw::internal::GpwDiis;
using ATV::cpu::linalg::RealMatrix;

TEST(GpwDiis, RejectsNonpositiveHistoryDepth) {
    EXPECT_THROW(GpwDiis(0), std::invalid_argument);
    EXPECT_THROW(GpwDiis(-1), std::invalid_argument);
}

TEST(GpwDiis, ExtrapolatesFromIndependentResiduals) {
    GpwDiis diis(2);
    const RealMatrix H1 = {{1.0, 0.0}, {0.0, 2.0}};
    const RealMatrix H2 = {{3.0, 0.0}, {0.0, 4.0}};
    diis.step(H1, RealMatrix{{2.0, 0.0}, {0.0, 0.0}});
    const auto mixed = diis.step(H2, RealMatrix{{0.0, 0.0}, {0.0, 1.0}});
    EXPECT_NEAR(mixed(0, 0), 2.6, 1e-14);
    EXPECT_NEAR(mixed(1, 1), 3.6, 1e-14);
    diis.clear();
    EXPECT_EQ(diis.step(H1, H1)(0, 0), H1(0, 0));
}

TEST(GpwDiis, KPointHistoryDepthMustBePositive) {
    using ATV::cpu::gpw::internal::KDiis;
    EXPECT_THROW((KDiis(0, {1.0})), std::invalid_argument);
    EXPECT_THROW((KDiis(-1, {1.0})), std::invalid_argument);
}

TEST(GpwDiis, KPointReductionIsExplicitAndChangesCoefficients) {
    using ATV::cpu::gpw::internal::KDiis;
    using ATV::cpu::linalg::CplxMatrix;
    int reductions = 0;
    KDiis local(2, {1.0});
    KDiis distributed(2, {1.0}, [&](double* block, int count) {
        ++reductions;
        ASSERT_EQ(count, 4); // residual block only, no constraint row
        EXPECT_DOUBLE_EQ(block[0], 4.0);
        EXPECT_DOUBLE_EQ(block[3], 1.0);
        block[3] += 3.0; // another owner's independent residual contribution
    });
    const CplxMatrix h1 = {{1.0, 0.0}, {0.0, 2.0}};
    const CplxMatrix h2 = {{3.0, 0.0}, {0.0, 4.0}};
    const CplxMatrix e1 = {{2.0, 0.0}, {0.0, 0.0}};
    const CplxMatrix e2 = {{0.0, 0.0}, {0.0, 1.0}};
    local.step({h1}, {e1});
    distributed.step({h1}, {e1});
    EXPECT_EQ(reductions, 0);
    const auto a = local.step({h2}, {e2});
    const auto b = distributed.step({h2}, {e2});
    EXPECT_EQ(reductions, 1);
    EXPECT_NEAR(a[0](0, 0).real(), 2.6, 1e-14);
    EXPECT_NEAR(b[0](0, 0).real(), 2.0, 1e-14);
}
