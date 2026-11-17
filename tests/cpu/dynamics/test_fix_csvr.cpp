/// @file tests/cpu/dynamics/test_fix_csvr.cpp
/// @brief `core/csvr.h` laws and `cpu::md::FixCSVR` group / COM-bias
///        semantics (LAMMPS `fix temp/csvr` + `compute temp/com`).
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/csvr.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;
using ATV::md::TemperatureBias;

namespace {

/// Four free atoms (lj units): types 1, 1, 2, 2, masses 1, 2, 3, 4, with
/// a drift +0.7 along x on the type-2 pair.
cpu::Frame four_atoms() {
    cpu::Frame f(4);
    const double v[4][3] = {{0.3, -0.2, 0.1},
                            {-0.1, 0.4, -0.3},
                            {0.5 + 0.7, 0.1, -0.2},
                            {-0.4 + 0.7, -0.3, 0.6}};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_mass)[i] = 1.0 + i;
        f.span(kAtom_type)[i] = i < 2 ? 1 : 2;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = v[i][0];
        f.span(kAtom_vy)[i] = v[i][1];
        f.span(kAtom_vz)[i] = v[i][2];
    }
    f.box = cpu::Box::cubic(100.0);
    f.meta.get(kEnergy) = -1.5;
    return f;
}

/// `by_type` must outlive the context (the context holds a view).
cpu::Context make_ctx(int n_dof_global, const std::vector<BigInt>& by_type = {},
                      bool known = true) {
    cpu::Context ctx;
    ctx.md.n_dof_global = n_dof_global;
    ctx.md.n_constraints_by_type = by_type.empty() ? nullptr : by_type.data();
    ctx.md.n_constraint_types = static_cast<int>(by_type.size());
    ctx.md.constraints_by_type_known = known;
    return ctx;
}

void launch(cpu::md::FixCSVR& th, cpu::Context& ctx, cpu::Frame& f,
            BigInt step) {
    th.launch({ctx, f, ATV::slot_of(md::Phase::end_of_step), step});
}

struct GroupSums {
    double m = 0, p[3] = {}, mvv = 0;
};

GroupSums sums(const cpu::Frame& f, int type) {
    GroupSums s;
    for (int i = 0; i < f.nlocal(); ++i) {
        if (f.span(kAtom_type)[i] != type)
            continue;
        const double m = f.span(kAtom_mass)[i];
        const double v[3] = {f.span(kAtom_vx)[i], f.span(kAtom_vy)[i],
                             f.span(kAtom_vz)[i]};
        s.m += m;
        for (int d = 0; d < 3; ++d)
            s.p[d] += m * v[d];
        s.mvv += m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    return s;
}

double peculiar_mvv(const GroupSums& s) {
    return s.mvv - (s.p[0] * s.p[0] + s.p[1] * s.p[1] + s.p[2] * s.p[2]) / s.m;
}

} // namespace

TEST(CsvrNoise, IsAPureFunctionOfSeedStepAndDof) {
    const auto a = ATV::md::csvr_noise(7u, 123, 40);
    const auto b = ATV::md::csvr_noise(7u, 123, 40);
    EXPECT_EQ(a.r1, b.r1);
    EXPECT_EQ(a.sum_sq, b.sum_sq);
    const auto c = ATV::md::csvr_noise(8u, 123, 40);
    const auto d = ATV::md::csvr_noise(7u, 124, 40);
    EXPECT_NE(a.r1, c.r1);
    EXPECT_NE(a.r1, d.r1);
}

TEST(CsvrNoise, MomentsMatchNormalAndChiSquared) {
    // N_f − 1 = 1 (one normal²), 2 (even), 3 (odd) and 59 (Γ path).
    constexpr int kDraws = 40000;
    for (int n_dof : {2, 3, 4, 60}) {
        SCOPED_TRACE(n_dof);
        const double k = n_dof - 1;
        double s1 = 0, s2 = 0, q1 = 0, q2 = 0;
        for (int step = 0; step < kDraws; ++step) {
            const auto z = ATV::md::csvr_noise(1234u, step, n_dof);
            s1 += z.r1;
            s2 += z.r1 * z.r1;
            q1 += z.sum_sq;
            q2 += z.sum_sq * z.sum_sq;
        }
        const double mean_r = s1 / kDraws,
                     var_r = s2 / kDraws - mean_r * mean_r;
        const double mean_q = q1 / kDraws,
                     var_q = q2 / kDraws - mean_q * mean_q;
        // 5σ bands of the sample mean / variance.
        EXPECT_NEAR(mean_r, 0.0, 5.0 / std::sqrt(kDraws));
        EXPECT_NEAR(var_r, 1.0, 5.0 * std::sqrt(2.0 / kDraws));
        EXPECT_NEAR(mean_q, k, 5.0 * std::sqrt(2.0 * k / kDraws));
        // Var of the sample variance of χ²(k): (μ4 − σ⁴)/n, μ4 = 12k(k+4).
        const double sd_var =
            std::sqrt((12.0 * k * (k + 4.0) - 4.0 * k * k) / kDraws);
        EXPECT_NEAR(var_q, 2.0 * k, 5.0 * sd_var);
    }
}

TEST(CsvrAlpha, LimitsOfTheCouplingTime) {
    const ATV::md::CsvrNoise z{0.3, 8.5};
    // τ → ∞: no coupling (c1 = 1, c2 = 0 exactly).
    EXPECT_DOUBLE_EQ(ATV::md::csvr_alpha2(2.0, 3.0, 10.0, 1.0, 1e300, z), 1.0);
    // Weak coupling: α² − 1 is dominated by the cross term 2 R₁ √(c1 c2).
    const double c2 = (1.0 - std::exp(-1e-8)) * 3.0 / 2.0 / 10.0;
    EXPECT_NEAR(ATV::md::csvr_alpha2(2.0, 3.0, 10.0, 1.0, 1e8, z) - 1.0,
                2.0 * 0.3 * std::sqrt(c2), 1e-3 * 2.0 * 0.3 * std::sqrt(c2));
    // τ → 0: K_new = K̄ (R₁² + Σ R²) / N_f.
    EXPECT_NEAR(ATV::md::csvr_alpha2(2.0, 3.0, 10.0, 1.0, 1e-9, z),
                3.0 * (0.09 + 8.5) / 10.0 / 2.0, 1e-12);
}

TEST(FixCsvr, WholeSystemRescalesEveryAtomByTheDrawnFactor) {
    cpu::Frame f = four_atoms();
    cpu::Frame before = f;
    cpu::Context ctx = make_ctx(9);
    cpu::md::FixCSVR th(0.8, 5.0, 1.0, 11u);
    th.build({ctx, f});
    EXPECT_EQ(th.n_dof(), 9);
    launch(th, ctx, f, 17);

    double mvv = 0;
    for (int t : {1, 2})
        mvv += sums(before, t).mvv;
    const double ke = 0.5 * mvv;
    const auto z = ATV::md::csvr_noise(11u, 17, 9);
    const double a2 = ATV::md::csvr_alpha2(ke, 0.5 * 9 * 0.8, 9, 1.0, 5.0, z);
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(f.span(kAtom_vx)[i],
                    std::sqrt(a2) * before.span(kAtom_vx)[i], 1e-14);
    // Bookkeeping: H̃ = −ΔK and conserved = E + K_after + H̃.
    const double dk = (a2 - 1.0) * ke;
    EXPECT_NEAR(th.thermostat_energy(), -dk, 1e-14);
    EXPECT_NEAR(f.meta.get(kThermostatEnergy), -dk, 1e-14);
    EXPECT_NEAR(f.meta.get(kConserved), -1.5 + ke + dk - dk, 1e-13);
}

TEST(FixCsvr, TypeGroupLeavesOtherAtomsAlone) {
    cpu::Frame f = four_atoms();
    cpu::Frame before = f;
    cpu::Context ctx = make_ctx(9);
    cpu::md::FixCSVR th(0.8, 5.0, 1.0, 11u, {2}, TemperatureBias::None);
    th.build({ctx, f});
    EXPECT_EQ(th.n_dof(), 3); // 3·2 − 3
    launch(th, ctx, f, 3);
    const auto z = ATV::md::csvr_noise(11u, 3, 3);
    const double ke = 0.5 * sums(before, 2).mvv;
    const double a =
        std::sqrt(ATV::md::csvr_alpha2(ke, 0.5 * 3 * 0.8, 3, 1.0, 5.0, z));
    for (int i = 0; i < 4; ++i) {
        const double expect = i < 2 ? 1.0 : a;
        EXPECT_NEAR(f.span(kAtom_vy)[i], expect * before.span(kAtom_vy)[i],
                    1e-14);
    }
}

TEST(FixCsvr, ComBiasKeepsTheGroupMomentumAndScalesThePeculiarPart) {
    cpu::Frame f = four_atoms();
    cpu::Frame before = f;
    cpu::Context ctx = make_ctx(9);
    cpu::md::FixCSVR th(0.8, 2.0, 1.0, 5u, {2}, TemperatureBias::Com);
    th.build({ctx, f});
    launch(th, ctx, f, 42);

    const GroupSums b = sums(before, 2), a = sums(f, 2);
    for (int d = 0; d < 3; ++d)
        EXPECT_NEAR(a.p[d], b.p[d], 1e-14);
    const auto z = ATV::md::csvr_noise(5u, 42, 3);
    const double ke = 0.5 * peculiar_mvv(b);
    const double a2 = ATV::md::csvr_alpha2(ke, 0.5 * 3 * 0.8, 3, 1.0, 2.0, z);
    EXPECT_NEAR(peculiar_mvv(a), a2 * peculiar_mvv(b), 1e-13);
    // Type 1 untouched.
    const GroupSums o = sums(f, 1), ob = sums(before, 1);
    EXPECT_EQ(o.mvv, ob.mvv);
}

TEST(FixCsvr, GroupDofSubtractsTheConstraintsItsAtomsOwn) {
    cpu::Frame f = four_atoms();
    // One constraint owned by a type-2 atom, two by type 1.
    const std::vector<BigInt> by_type = {0, 2, 1};
    cpu::Context ctx = make_ctx(9 - 3, by_type);
    cpu::md::FixCSVR group2(0.8, 5.0, 1.0, 1u, {2}, TemperatureBias::Com);
    group2.build({ctx, f});
    EXPECT_EQ(group2.n_dof(), 3 * 2 - 3 - 1);
    cpu::md::FixCSVR both(0.8, 5.0, 1.0, 1u, {1, 2}, TemperatureBias::None);
    both.build({ctx, f});
    EXPECT_EQ(both.n_dof(), ctx.md.n_dof_global);
    cpu::md::FixCSVR all(0.8, 5.0, 1.0, 1u);
    all.build({ctx, f});
    EXPECT_EQ(all.n_dof(), ctx.md.n_dof_global);
}

TEST(FixCsvr, RefusesAGroupItCannotCount) {
    cpu::Frame f = four_atoms();
    cpu::Context unknown = make_ctx(9, {}, false);
    cpu::md::FixCSVR th(0.8, 5.0, 1.0, 1u, {2}, TemperatureBias::None);
    EXPECT_THROW(th.build({unknown, f}), std::runtime_error);
    cpu::Context ctx = make_ctx(9);
    cpu::md::FixCSVR empty(0.8, 5.0, 1.0, 1u, {7}, TemperatureBias::None);
    EXPECT_THROW(empty.build({ctx, f}), std::runtime_error);
    EXPECT_THROW(
        (cpu::md::FixCSVR{0.8, 5.0, 1.0, 1u, {-1}, TemperatureBias::None}),
        std::invalid_argument);
    EXPECT_THROW((cpu::md::FixCSVR{0.8, 0.0, 1.0, 1u}), std::invalid_argument);
}

TEST(FixCsvr, StrongCouplingSamplesTheCanonicalKineticEnergy) {
    // τ = dt/10: each step nearly redraws K from its canonical law,
    // Γ(N_f/2, k_B T): ⟨K⟩ = N_f T/2, Var K = N_f T²/2 (lj units). The COM
    // bias makes the type-2 peculiar KE the sampled variable (N_f = 3).
    cpu::Frame f = four_atoms();
    cpu::Context ctx = make_ctx(9);
    constexpr double T = 0.8;
    cpu::md::FixCSVR th(T, 0.1, 1.0, 99u, {2}, TemperatureBias::Com);
    th.build({ctx, f});
    constexpr int kSteps = 40000;
    double k1 = 0, k2 = 0;
    for (int s = 0; s < kSteps; ++s) {
        launch(th, ctx, f, s);
        const double k = 0.5 * peculiar_mvv(sums(f, 2));
        k1 += k;
        k2 += k * k;
    }
    const double mean = k1 / kSteps, var = k2 / kSteps - mean * mean;
    const double nf = 3.0;
    EXPECT_NEAR(mean, 0.5 * nf * T, 0.03 * 0.5 * nf * T);
    EXPECT_NEAR(var, 0.5 * nf * T * T, 0.08 * 0.5 * nf * T * T);
}
