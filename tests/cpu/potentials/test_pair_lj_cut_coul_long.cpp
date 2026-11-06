/// @file tests/cpu/potentials/test_pair_lj_cut_coul_long.cpp
/// @brief Fused lj/cut/coul/long: finite energy, specials drop 1-2, analytic
///        erfc (cpu/potentials/ewald_erfc.h).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <xsimd/xsimd.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/driver/step_timing.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_real.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/ewald_erfc.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"

using namespace ATV;

namespace {
// erfc, e^{-x²} and the two pair factors in long double (glibc erfcl/expl).
struct Exact {
    long double erfc, expm, e, f;
};
Exact exact(double r2, double alpha) {
    const long double r = sqrtl(static_cast<long double>(r2));
    const long double x = alpha * r;
    const long double er = erfcl(x), em = expl(-x * x);
    return {er, em, er / r,
            (er + 1.1283791670955125738961589031L * x * em) / (r * r * r)};
}
} // namespace

TEST(EwaldErfc, ErfcAndGaussianAreUlpAccurateOnEveryInterval) {
    using B = xsimd::batch<double>;
    constexpr int W = static_cast<int>(B::size);
    // Points on and next to every fdlibm interval boundary (0.25, 0.84375,
    // 1.25, 1/0.35) plus a dense sweep of [0, 6].
    std::vector<double> xs = {0.0,
                              1e-300,
                              1e-12,
                              0.25,
                              0.84375,
                              1.25,
                              2.857142448425293,
                              2.8571424484252935,
                              5.99};
    for (double b : {0.25, 0.84375, 1.25, 2.857142448425293})
        for (double s : {-1.0, 1.0})
            xs.push_back(std::nextafter(b, b + s));
    for (int k = 0; k <= 60000; ++k)
        xs.push_back(6.0 * k / 60000.0);
    while (xs.size() % W)
        xs.push_back(1.0);
    double worst_erfc = 0.0, worst_exp = 0.0;
    alignas(64) double in[W], erfc_out[W], exp_out[W];
    for (std::size_t k = 0; k < xs.size(); k += W) {
        for (int l = 0; l < W; ++l)
            in[l] = xs[k + l];
        B erfc_x, exp_m;
        const B xb = B::load_aligned(in);
        cpu::md::ewald_erfc_exp(xb, B(1.0) / xb, erfc_x, exp_m);
        erfc_x.store_aligned(erfc_out);
        exp_m.store_aligned(exp_out);
        for (int l = 0; l < W; ++l) {
            const long double x = in[l];
            worst_erfc = std::max(
                worst_erfc,
                static_cast<double>(fabsl(erfc_out[l] / erfcl(x) - 1.0L)));
            // exp(-x²) with x² split exactly (x² = hi + lo, |lo| ≤ ulp(hi)/2):
            // exp(-hi)·(1 - lo) is ~1 ulp on any platform, whereas
            // expl(-x*x) rounds x² first wherever long double is double
            // (Apple arm64), an x²·ε error of the reference itself.
            const double hi = in[l] * in[l];
            const double lo = std::fma(in[l], in[l], -hi);
            const long double exp_ref =
                static_cast<long double>(std::exp(-hi)) * (1.0L - lo);
            worst_exp = std::max(worst_exp, static_cast<double>(fabsl(
                                                exp_out[l] / exp_ref - 1.0L)));
        }
    }
    // A few ulp for exactly representable arguments (fdlibm: < 1 ulp; the
    // xsimd exponential adds its own).
    EXPECT_LT(worst_erfc, 1.5e-15);
    EXPECT_LT(worst_exp, 1.0e-15);
}

TEST(EwaldErfc, PairFactorsMatchExactFormulaAndPadding) {
    const double alpha = 0.35;
    const int W = cpu::md::ewald_batch_width();
    const int n = 3 * W + 1; // a partial last batch
    std::vector<double> r2(n + W, 1.0), e(n + W), f(n + W);
    for (int k = 0; k < n; ++k) {
        const double r = 0.4 + 11.0 * k / (n - 1);
        r2[k] = r * r;
    }
    cpu::md::ewald_pair_factors(r2.data(), n, alpha, e.data(), f.data());
    for (int k = 0; k < n; ++k) {
        const Exact ref = exact(r2[k], alpha);
        EXPECT_NEAR(e[k], static_cast<double>(ref.e),
                    4e-15 * static_cast<double>(ref.e))
            << "r2=" << r2[k];
        EXPECT_NEAR(f[k], static_cast<double>(ref.f),
                    4e-15 * static_cast<double>(ref.f))
            << "r2=" << r2[k];
    }
}

TEST(PairLJCutCoulLong, RealSpaceCoulombIsAnalyticErfc) {
    // Two charges, ε = 0: the pair energy and force are exactly
    // k_e q_i q_j erfc(αr)/r and its derivative (no table, no polynomial).
    const double alpha = 0.35, cutoff = 9.0;
    for (double r : {0.9, 2.2, 3.6, 5.0, 8.2, 8.99}) {
        cpu::Frame frame(2);
        frame.box = cpu::Box::cubic(40.0);
        frame.span(kAtom_x)[0] = 1.0;
        frame.span(kAtom_x)[1] = 1.0 + r;
        frame.span(kAtom_charge)[0] = 0.7;
        frame.span(kAtom_charge)[1] = -0.45;
        frame.span(kAtom_tag)[0] = 1;
        frame.span(kAtom_tag)[1] = 2;
        cpu::md::PairList pairs;
        pairs.inum = 2;
        pairs.ilist = {0, 1};
        pairs.numneigh = {1, 0};
        pairs.firstneigh = {0, 1};
        pairs.neighbors = {1};
        pairs.total_pairs = 1;
        cpu::Context context;
        context.md.pair_list = &pairs;
        cpu::PairLJCutCoulLong pair({{0, 0, 0.0, 1.0}}, alpha, cutoff);
        pair.set_special_bonds(1, 1, 1, 1, 1, 1);
        pair.build({context, frame});
        pair.launch({context, frame, 0, 0, true, true});
        const Exact ref = exact(r * r, alpha);
        const double qq = ATV::units::UnitSystem::lj().qqrd2e() * 0.7 * -0.45;
        const double e_ref = qq * static_cast<double>(ref.e);
        const double f_ref = qq * static_cast<double>(ref.f) * (-r);
        EXPECT_NEAR(frame.meta.get(kEnergy), e_ref, 4e-15 * std::abs(e_ref))
            << "r=" << r;
        EXPECT_NEAR(frame.span(kAtom_fx)[0], f_ref, 4e-15 * std::abs(f_ref))
            << "r=" << r;
        EXPECT_NEAR(frame.span(kAtom_fx)[1], -f_ref, 4e-15 * std::abs(f_ref))
            << "r=" << r;
        pair.destroy({context, frame});
    }
}

TEST(PairLJCutCoulLong, ZeroVersionSpecialScaleCacheIsInitialized) {
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(20.0);
    frame.span(kAtom_x)[1] = 2.0;
    frame.span(kAtom_tag)[0] = 1;
    frame.span(kAtom_tag)[1] = 2;
    cpu::md::PairList pairs;
    pairs.inum = 2;
    pairs.ilist = {0, 1};
    pairs.numneigh = {1, 0};
    pairs.firstneigh = {0, 1};
    pairs.neighbors = {1};
    pairs.total_pairs = 1;
    ASSERT_EQ(pairs.version, 0u);
    cpu::Context context;
    context.md.pair_list = &pairs;
    cpu::PairLJCutCoulLong pair({{0, 0, 1.0, 1.0}}, 0.3, 8.0);
    pair.set_special_bonds_amber().set_energy_shift(false);
    pair.build({context, frame});
    pair.launch({context, frame, 0, 0, true, true});
    const double invr6 = 1.0 / 64.0;
    EXPECT_NEAR(frame.meta.get(kEnergy), 4.0 * (invr6 * invr6 - invr6), 1e-14);
    EXPECT_NEAR(frame.span(kAtom_fx)[0], -12.0 * (2.0 * invr6 * invr6 - invr6),
                1e-14);
    pair.destroy({context, frame});
}

TEST(PairLJCutCoulLong, FiniteEnergyWithAmberSpecials) {
    const int n = 4;
    cpu::Frame f(n);
    const double xyz[][3] = {{0, 0, 0}, {1.5, 0, 0}, {5, 0, 0}, {5, 1.5, 0}};
    const double q[] = {0.5, -0.5, 0.3, -0.3};
    const int typ[] = {0, 1, 0, 1};
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
        f.span(kAtom_charge)[i] = q[i];
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_mass)[i] = 12.0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    f.add_bond(0, 1, 0);
    f.add_bond(2, 3, 0);
    f.build_special_list();

    const double alpha = 0.3;
    const double cut = 10.0;
    const double eps = cpu::kcal_per_mol_to_ev(0.1);
    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{cut, 2.0, 1, 0, true});
    sim.set_evaluator(
        cpu::PairLJCutCoulLong{{{0, 0, eps, 3.4}, {1, 1, eps, 2.5}}, alpha, cut}
            .set_special_bonds_amber()
            .set_compute_stress(true));
    sim.set_evaluator(cpu::KSpaceEwald{alpha, /*accuracy=*/1e-5});
    sim.set_integrator(cpu::md::VelocityVerlet{1.0});
    sim.build();
    sim.run(f, 1);

    const double E = f.meta.get(kEnergy);
    EXPECT_TRUE(std::isfinite(E));
    EXPECT_LT(std::fabs(E), 1e3);
}

TEST(PairLJCutCoulLong, NameIsLammpsStyle) {
    cpu::PairLJCutCoulLong p({{{0, 0, 0.01, 3.0}}, 0.25, 8.0});
    EXPECT_EQ(p.name(), "lj/cut/coul/long");
    EXPECT_EQ(cpu::md::StepTiming::classify(p.name()),
              cpu::md::StepTiming::Pair);
}
