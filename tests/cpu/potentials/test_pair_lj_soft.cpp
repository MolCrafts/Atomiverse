/// @file tests/cpu/potentials/test_pair_lj_soft.cpp
/// @brief Unit tests for cpu::PairLJSoft (single-function cases).

#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_soft.h"

using namespace ATV;

namespace {

cpu::Frame dimer(double r) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(50.0);
    return f;
}

cpu::md::PairList half_list() {
    cpu::md::PairList pairs;
    pairs.inum = 1;
    pairs.ilist = {0};
    pairs.numneigh = {1, 0};
    pairs.firstneigh = {0, 0};
    pairs.neighbors = {1};
    pairs.total_pairs = 1;
    pairs.has_shifts = false;
    pairs.version = 1;
    return pairs;
}

void zero_ef(cpu::Frame& f) {
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyPair) = 0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
}

double soft_energy_closed(double eps, double sigma, double rc, double r,
                          double lam, double alpha, double p) {
    const double sigma6 = std::pow(sigma, 6);
    const double r_lam = std::pow(
        alpha * sigma6 * std::pow(1.0 - lam, p) + std::pow(r, 6), 1.0 / 6.0);
    const double sr = sigma / r_lam;
    const double sr6 = std::pow(sr, 6);
    const double sr12 = sr6 * sr6;
    const double vlj = 4.0 * eps * (sr12 - sr6);
    const double src = sigma / rc;
    const double src6 = std::pow(src, 6);
    const double vshift = 4.0 * eps * (src6 * src6 - src6);
    return lam * (vlj - vshift);
}

} // namespace

TEST(PairLJSoft, NameAndDefaults) {
    cpu::PairLJSoft soft(0.5, 3.0, 8.0);
    EXPECT_EQ(soft.name(), "lj/soft");
    EXPECT_DOUBLE_EQ(soft.lambda, 1.0);
    EXPECT_DOUBLE_EQ(soft.pair_cutoff(), 8.0);
}

TEST(PairLJSoft, LambdaOneMatchesPairLJCut) {
    const double eps = 0.5, sigma = 3.0, rc = 8.0, r = sigma;
    auto pairs = half_list();

    cpu::Frame f_hard = dimer(r);
    cpu::Frame f_soft = dimer(r);
    zero_ef(f_hard);
    zero_ef(f_soft);

    cpu::Context ctx;
    ctx.md.pair_list = &pairs;

    cpu::PairLJCut hard(eps, sigma, rc);
    hard.build({ctx, f_hard});
    hard.launch({ctx, f_hard, 0, 0});
    hard.destroy({ctx, f_hard});

    cpu::PairLJSoft soft(eps, sigma, rc);
    soft.lambda = 1.0;
    soft.build({ctx, f_soft});
    soft.launch({ctx, f_soft, 0, 0});
    soft.destroy({ctx, f_soft});

    EXPECT_NEAR(f_soft.meta.get(kEnergy), f_hard.meta.get(kEnergy), 1e-12);
    EXPECT_NEAR(f_soft.span(kAtom_fx)[0], f_hard.span(kAtom_fx)[0], 1e-12);
    EXPECT_NEAR(f_soft.span(kAtom_fx)[1], f_hard.span(kAtom_fx)[1], 1e-12);
}

TEST(PairLJSoft, FiniteEnergyNearContact) {
    const double eps = 0.5, sigma = 3.0, rc = 8.0;
    const double r = 1e-8, lam = 0.5, alpha = 0.5, p = 1.0;
    auto pairs = half_list();
    auto f = dimer(r);
    zero_ef(f);

    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::PairLJSoft soft(eps, sigma, rc, alpha, p);
    soft.lambda = lam;
    soft.build({ctx, f});
    soft.launch({ctx, f, 0, 0});
    soft.destroy({ctx, f});

    const double E = f.meta.get(kEnergy);
    EXPECT_TRUE(std::isfinite(E));
    const double E_ref = soft_energy_closed(eps, sigma, rc, r, lam, alpha, p);
    EXPECT_NEAR(E, E_ref, 1e-10);
}

TEST(PairLJSoft, LambdaZeroIsZero) {
    const double eps = 0.5, sigma = 3.0, rc = 8.0, r = 3.0;
    auto pairs = half_list();
    auto f = dimer(r);
    zero_ef(f);

    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::PairLJSoft soft(eps, sigma, rc);
    soft.lambda = 0.0;
    soft.build({ctx, f});
    soft.launch({ctx, f, 0, 0});
    soft.destroy({ctx, f});

    EXPECT_NEAR(f.meta.get(kEnergy), 0.0, 1e-14);
    EXPECT_NEAR(f.span(kAtom_fx)[0], 0.0, 1e-14);
    EXPECT_NEAR(f.span(kAtom_fx)[1], 0.0, 1e-14);
}

TEST(PairLJSoft, ForceMatchesFiniteDifference) {
    const double eps = 0.5, sigma = 3.0, rc = 8.0;
    const double r0 = sigma, lam = 0.5, alpha = 0.5, p = 1.0;
    const double h = 1e-5;
    auto pairs = half_list();

    auto energy_at = [&](double r) {
        auto f = dimer(r);
        zero_ef(f);
        cpu::Context ctx;
        ctx.md.pair_list = &pairs;
        cpu::PairLJSoft soft(eps, sigma, rc, alpha, p);
        soft.lambda = lam;
        soft.build({ctx, f});
        soft.launch({ctx, f, 0, 0});
        soft.destroy({ctx, f});
        return f.meta.get(kEnergy);
    };

    auto f = dimer(r0);
    zero_ef(f);
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::PairLJSoft soft(eps, sigma, rc, alpha, p);
    soft.lambda = lam;
    soft.build({ctx, f});
    soft.launch({ctx, f, 0, 0});
    soft.destroy({ctx, f});

    // Move atom 1 along +x by central difference of total energy.
    // F_x on atom 1 ≈ −dE/dx1; atom 0 is fixed at origin.
    const double e_p = energy_at(r0 + h);
    const double e_m = energy_at(r0 - h);
    const double f_fd = -(e_p - e_m) / (2.0 * h);
    EXPECT_NEAR(f.span(kAtom_fx)[1], f_fd, 1e-6);
    EXPECT_NEAR(f.span(kAtom_fx)[0] + f.span(kAtom_fx)[1], 0.0, 1e-12);
}
