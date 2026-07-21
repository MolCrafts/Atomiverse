/// @file tests/cpu/potentials/test_pair_kernel.cpp
/// @brief force-infra-03 P3.1/P3.2: the backend-agnostic PairKernel seam.
///
/// Proves the kernels are (a) correct vs closed-form LJ / erfc-Coulomb, (b)
/// generic — ONE definition instantiates for `double` AND `xsimd::batch<double>`
/// and the SIMD lanes agree with the scalar, (c) composable — SumKernel<LJ,Coul>
/// equals the sum of the two run separately (the lj/cut/coul/long pattern), and
/// (d) cutoff-masked per term (contribution is exactly 0 beyond each cutoff).

#include <gtest/gtest.h>

#include <cmath>
#include <xsimd/xsimd.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/potentials/pair_kernel.h"

using namespace ATV;
using potentials::CoulKernel;
using potentials::LJKernel;
using potentials::SumKernel;

namespace {

LJKernel make_lj(double eps, double sigma, double cutoff) {
    const double sr = sigma / cutoff, sr6 = std::pow(sr, 6);
    const double v_shift = 4.0 * eps * (sr6 * sr6 - sr6);
    return LJKernel{eps, sigma * sigma, cutoff * cutoff, v_shift};
}

} // namespace

TEST(PairKernel, LJMatchesClosedForm) {
    const double eps = 0.0104, sigma = 3.4, cutoff = 10.0;
    auto lj = make_lj(eps, sigma, cutoff);
    const double r = 3.9, r2 = r * r;

    auto c = lj.operator()<double>(r2, 0.0, 0.0);

    const double sr6 = std::pow(sigma / r, 6), sr12 = sr6 * sr6;
    const double e_ref =
        4 * eps * (sr12 - sr6)
        - 4 * eps
              * (std::pow(sigma / cutoff, 12) - std::pow(sigma / cutoff, 6));
    const double f_ref = 24 * eps * (2 * sr12 - sr6) / r2;
    EXPECT_NEAR(c.energy, e_ref, 1e-12);
    EXPECT_NEAR(c.fpair, f_ref, 1e-12);
}

TEST(PairKernel, CoulMatchesClosedForm) {
    const double alpha = 0.30, cutoff = 10.0, qi = 1.0, qj = -1.0;
    CoulKernel coul{units::coulomb, alpha, cutoff * cutoff};
    const double r = 4.2, r2 = r * r;

    auto c = coul.operator()<double>(r2, qi, qj);

    const double pref = units::coulomb * qi * qj / r;
    const double e_ref = pref * std::erfc(alpha * r);
    const double f_ref =
        pref
        * (std::erfc(alpha * r)
           + CoulKernel::kEwaldF * (alpha * r) * std::exp(-alpha * alpha * r2))
        / r2;
    EXPECT_NEAR(c.energy, e_ref, 1e-12);
    EXPECT_NEAR(c.fpair, f_ref, 1e-12);
}

TEST(PairKernel, BatchLanesAgreeWithScalar) {
    using batch = xsimd::batch<double>;
    auto lj = make_lj(0.0104, 3.4, 10.0);
    CoulKernel coul{units::coulomb, 0.30, 100.0};

    // Distinct r per lane, all inside cutoff.
    alignas(batch::arch_type::alignment()) double r2v[batch::size];
    for (std::size_t k = 0; k < batch::size; ++k) {
        const double r = 3.5 + 0.3 * static_cast<double>(k);
        r2v[k] = r * r;
    }
    const batch r2 = batch::load_aligned(r2v);
    const batch qi(1.0), qj(-1.0);

    auto clj = lj.operator()<batch>(r2, qi, qj);
    auto cco = coul.operator()<batch>(r2, qi, qj);

    double ljf[batch::size], lje[batch::size], cof[batch::size],
        coe[batch::size];
    clj.fpair.store_unaligned(ljf);
    clj.energy.store_unaligned(lje);
    cco.fpair.store_unaligned(cof);
    cco.energy.store_unaligned(coe);
    for (std::size_t k = 0; k < batch::size; ++k) {
        auto slj = lj.operator()<double>(r2v[k], 1.0, -1.0);
        auto sco = coul.operator()<double>(r2v[k], 1.0, -1.0);
        EXPECT_NEAR(ljf[k], slj.fpair, 1e-12) << "lj lane " << k;
        EXPECT_NEAR(lje[k], slj.energy, 1e-12) << "lj lane " << k;
        EXPECT_NEAR(cof[k], sco.fpair, 1e-12) << "coul lane " << k;
        EXPECT_NEAR(coe[k], sco.energy, 1e-12) << "coul lane " << k;
    }
}

TEST(PairKernel, SumComposesLJAndCoul) {
    auto lj = make_lj(0.0104, 3.4, 10.0);
    CoulKernel coul{units::coulomb, 0.30, 100.0};
    SumKernel<LJKernel, CoulKernel> both{lj, coul};
    const double r2 = 4.0 * 4.0;

    auto cs = both.operator()<double>(r2, 1.0, -1.0);
    auto cl = lj.operator()<double>(r2, 1.0, -1.0);
    auto cc = coul.operator()<double>(r2, 1.0, -1.0);
    EXPECT_NEAR(cs.fpair, cl.fpair + cc.fpair, 1e-12);
    EXPECT_NEAR(cs.energy, cl.energy + cc.energy, 1e-12);
}

TEST(PairKernel, CutoffMaskedPerTerm) {
    auto lj = make_lj(0.0104, 3.4, 5.0);         // LJ cutoff 5
    CoulKernel coul{units::coulomb, 0.30, 49.0}; // Coulomb cutoff 7
    const double r2 = 6.0 * 6.0;                 // inside Coulomb, outside LJ

    auto cl = lj.operator()<double>(r2, 1.0, -1.0);
    auto cc = coul.operator()<double>(r2, 1.0, -1.0);
    EXPECT_EQ(cl.fpair, 0.0);
    EXPECT_EQ(cl.energy, 0.0);
    EXPECT_NE(cc.fpair, 0.0); // Coulomb still active at r=6 < 7
}
