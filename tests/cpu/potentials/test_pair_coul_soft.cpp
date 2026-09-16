/// @file tests/cpu/potentials/test_pair_coul_soft.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_coul_soft.h"

using namespace ATV;

namespace {

cpu::Frame ion_dimer(double r, double q0, double q1) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_charge)[0] = q0;
    f.span(kAtom_charge)[1] = q1;
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
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
}

} // namespace

TEST(PairCoulSoft, NameAndDefaults) {
    cpu::PairCoulSoft soft(10.0, 1.0);
    EXPECT_EQ(soft.name(), "coul/soft");
    EXPECT_DOUBLE_EQ(soft.lambda, 1.0);
}

TEST(PairCoulSoft, LambdaOneMatchesShiftedHardCoulomb) {
    // PairCoulCut is unshifted; λ=1 soft uses shifted hard Coulomb:
    // V = k qi qj (1/r − 1/rc), F_r = k |qi qj| / r²
    const double r = 3.0, rc = 10.0;
    const double qq = units::coulomb * (1.0) * (-1.0);
    const double E_ref = qq * (1.0 / r - 1.0 / rc);
    const double Fr_ref = std::fabs(qq) / (r * r);

    auto pairs = half_list();
    cpu::Frame f_soft = ion_dimer(r, 1.0, -1.0);
    zero_ef(f_soft);
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;

    cpu::PairCoulSoft soft(rc, 1.0);
    soft.lambda = 1.0;
    soft.build({ctx, f_soft});
    soft.launch({ctx, f_soft, 0, 0});
    soft.destroy({ctx, f_soft});

    EXPECT_NEAR(f_soft.meta.get(kEnergy), E_ref, 1e-12);
    EXPECT_NEAR(std::fabs(f_soft.span(kAtom_fx)[0]), Fr_ref, 1e-12);
}

TEST(PairCoulSoft, IntermediateLambdaClosedForm) {
    // Goldens from Beutler map (python3, 2026-07-23).
    const double r = 3.0, rc = 10.0, lam = 0.5, alpha = 0.5, p = 1.0, sc = 1.0;
    const double E_ref = -1.6798214399374538;
    const double Fr_ref = 0.799660331374428; // |F|

    auto pairs = half_list();
    auto f = ion_dimer(r, 1.0, -1.0);
    zero_ef(f);
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::PairCoulSoft soft(rc, sc, alpha, p);
    soft.lambda = lam;
    soft.build({ctx, f});
    soft.launch({ctx, f, 0, 0});
    soft.destroy({ctx, f});

    EXPECT_NEAR(f.meta.get(kEnergy), E_ref, 1e-10);
    EXPECT_NEAR(std::fabs(f.span(kAtom_fx)[0]), Fr_ref, 1e-10);
    EXPECT_NEAR(f.span(kAtom_fx)[0] + f.span(kAtom_fx)[1], 0.0, 1e-12);
}
