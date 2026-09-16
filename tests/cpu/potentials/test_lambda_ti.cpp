/// @file tests/cpu/potentials/test_lambda_ti.cpp
/// @brief ∂U/∂λ + fixed-geometry Kirkwood TI identity for PairLJSoft.
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
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

double energy_at(double lam, cpu::Frame f, cpu::md::PairList& pairs) {
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyPair) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::PairLJSoft soft(0.5, 3.0, 8.0, 0.5, 1.0);
    soft.lambda = lam;
    soft.build({ctx, f});
    soft.launch({ctx, f, 0, 0});
    soft.destroy({ctx, f});
    return f.meta.get(kEnergy);
}

} // namespace

TEST(LambdaTI, DUdLambdaMatchesFD) {
    auto pairs = half_list();
    auto f = dimer(3.0);
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyPair) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::PairLJSoft soft(0.5, 3.0, 8.0, 0.5, 1.0);
    soft.lambda = 0.5;
    soft.build({ctx, f});
    soft.launch({ctx, f, 0, 0});
    const double dU = soft.dU_dlambda();
    soft.destroy({ctx, f});

    const double h = 1e-6;
    const double fd = (energy_at(0.5 + h, dimer(3.0), pairs)
                       - energy_at(0.5 - h, dimer(3.0), pairs))
                      / (2.0 * h);
    EXPECT_NEAR(dU, fd, 1e-6 * std::max(1.0, std::fabs(fd)));
}

TEST(LambdaTI, TrapezoidRecoversDeltaU) {
    auto pairs = half_list();
    const double U0 = energy_at(0.0, dimer(3.0), pairs);
    const double U1 = energy_at(1.0, dimer(3.0), pairs);
    // Composite trapezoid with dense λ grid (Beutler ∂U/∂λ is smooth on (0,1]).
    const int N = 400;
    double integral = 0;
    for (int i = 0; i <= N; ++i) {
        const double lam = static_cast<double>(i) / N;
        auto f = dimer(3.0);
        f.meta.get(kEnergy) = 0;
        f.meta.get(kEnergyPair) = 0;
        for (int a = 0; a < 2; ++a)
            f.span(kAtom_fx)[a] = f.span(kAtom_fy)[a] = f.span(kAtom_fz)[a] = 0;
        cpu::Context ctx;
        ctx.md.pair_list = &pairs;
        cpu::PairLJSoft soft(0.5, 3.0, 8.0, 0.5, 1.0);
        soft.lambda = lam;
        soft.build({ctx, f});
        soft.launch({ctx, f, 0, 0});
        const double w = (i == 0 || i == N) ? 0.5 : 1.0;
        integral += w * soft.dU_dlambda();
        soft.destroy({ctx, f});
    }
    integral *= 1.0 / N;
    EXPECT_NEAR(integral, U1 - U0, 1e-5);
}
