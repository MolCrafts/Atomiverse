/// @file tests/cpu/scf/test_eom_ccsd_common.cpp
/// @brief Unit checks for the hoisted `make_tau` (private header
///        internal/eom_ccsd_common.h, namespace ATV::cpu::scf matching
///        internal/ccsd_pack.h).
///
/// scf-review-03-posthf-dedup collapsed the four verbatim `make_tau` copies
/// (ccsd.cpp + eom_{ee,ea,ip}_ccsd.cpp) into the single canonical definition in
/// include/atomiverse/cpu/scf/internal/eom_ccsd_common.h. This test pins that
/// one implementation against a hand-computed tau (written RED-first; landed
/// and GREEN).
///
/// Gauss-Stanton (PySCF gintermediates.make_tau):
///   tau_ij^ab = t2_ijab + 0.5*fac * P(ij)P(ab) t1_ia t1_jb
/// The verbatim kernel expands the antisymmetrizer into four sign terms; with
///   c = 0.5*fac  it adds  c[(t1_ia t1_jb - t1_ja t1_ib)
///                          - (t1_ib t1_ja - t1_jb t1_ia)]
/// which collapses (equal products) to  fac*(t1_ia t1_jb - t1_ib t1_ja).
///
/// NOTE: EomImds construction is intentionally NOT unit-tested here. Its
/// hoisted form is the computing constructor `EomImds(const CCSDSolution&)`,
/// which needs a full converged CCSDSolution (e2e territory). That path is
/// gated by the eom-ee / eom-ea / eom-ip regressions, not by this unit tier.

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/internal/eom_ccsd_common.h"

namespace {

namespace scf = ATV::cpu::scf;

// Hand case: no = nv = 2. Distinct t1 so every t1t1 product is distinct:
//   t1(0,0)=0.1  t1(0,1)=0.2  t1(1,0)=0.3  t1(1,1)=0.4
xt::xtensor<double, 2> make_t1() {
    xt::xtensor<double, 2> t1 = xt::zeros<double>({2, 2});
    t1(0, 0) = 0.1;
    t1(0, 1) = 0.2;
    t1(1, 0) = 0.3;
    t1(1, 1) = 0.4;
    return t1;
}

// t2 base (make_tau adds the correction onto a copy of t2, element-wise):
//   (0,0,0,0)=1.0  (0,1,0,1)=0.5  (1,0,1,0)=2.0
//   (0,1,1,0)=0.6  (1,0,0,1)=0.9  (0,1,0,0)=0.4
xt::xtensor<double, 4> make_t2() {
    xt::xtensor<double, 4> t2 = xt::zeros<double>({2, 2, 2, 2});
    t2(0, 0, 0, 0) = 1.0; // i==j  -> P(ij) kills the correction
    t2(0, 1, 0, 1) = 0.5;
    t2(1, 0, 1, 0) = 2.0;
    t2(0, 1, 1, 0) = 0.6;
    t2(1, 0, 0, 1) = 0.9;
    t2(0, 1, 0, 0) = 0.4; // a==b  -> P(ab) kills the correction
    return t2;
}

} // namespace

// fac = 1.0 -> tau (full amplitude).
TEST(TestEomCcsdCommon, MakeTauFacOne) {
    const auto tau = scf::make_tau(make_t2(), make_t1(), 1.0);

    // Hand computation, correction = fac*(t1_ia t1_jb - t1_ib t1_ja):
    //  (0,0,0,0): i==j             -> +0                  -> 1.0
    //  (0,1,0,1): 0.1*0.4-0.2*0.3 = -0.02  (*fac=1)       -> 0.5 - 0.02 = 0.48
    //  (1,0,1,0): 0.4*0.1-0.3*0.2 = -0.02                 -> 2.0 - 0.02 = 1.98
    //  (0,1,1,0): 0.2*0.3-0.1*0.4 = +0.02                 -> 0.6 + 0.02 = 0.62
    //  (1,0,0,1): 0.3*0.2-0.4*0.1 = +0.02                 -> 0.9 + 0.02 = 0.92
    //  (0,1,0,0): a==b             -> +0                  -> 0.4
    EXPECT_NEAR(tau(0, 0, 0, 0), 1.00, 1e-12);
    EXPECT_NEAR(tau(0, 1, 0, 1), 0.48, 1e-12);
    EXPECT_NEAR(tau(1, 0, 1, 0), 1.98, 1e-12);
    EXPECT_NEAR(tau(0, 1, 1, 0), 0.62, 1e-12);
    EXPECT_NEAR(tau(1, 0, 0, 1), 0.92, 1e-12);
    EXPECT_NEAR(tau(0, 1, 0, 0), 0.40, 1e-12);
}

// fac = 0.5 -> tau-tilde (half correction).
TEST(TestEomCcsdCommon, MakeTauFacHalf) {
    const auto tau = scf::make_tau(make_t2(), make_t1(), 0.5);

    // Same correction scaled by fac=0.5 -> +/- 0.01.
    EXPECT_NEAR(tau(0, 0, 0, 0), 1.00, 1e-12);
    EXPECT_NEAR(tau(0, 1, 0, 1), 0.49, 1e-12);
    EXPECT_NEAR(tau(1, 0, 1, 0), 1.99, 1e-12);
    EXPECT_NEAR(tau(0, 1, 1, 0), 0.61, 1e-12);
    EXPECT_NEAR(tau(1, 0, 0, 1), 0.91, 1e-12);
    EXPECT_NEAR(tau(0, 1, 0, 0), 0.40, 1e-12);
}

#endif // ATV_HAS_HF
