/// @file tests/cpu/potentials/test_pair_lj_types.cpp
/// @brief Unit: PairLJCut per-type coefficients (Lorentz-Berthelot mixing).
///
/// Until this landed, PairLJCut carried ONE epsilon/sigma for every atom in the
/// system. That is fine for argon and useless for anything with two elements — in
/// TIP3P the LJ site sits on oxygen only, so a single-sigma run forces the hydrogens
/// to wear the oxygen's sigma and a liquid-density water box self-destructs.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

constexpr double kEnergyTolerance = 1.0e-12;

/// Two atoms of the given types at separation r — several call sites below.
cpu::Frame dimer(int ta, int tb, double r) {
    cpu::Frame f(2);
    f.span(ATV::kAtom_x)[0] = 10.0;
    f.span(ATV::kAtom_y)[0] = 10.0;
    f.span(ATV::kAtom_z)[0] = 10.0;
    f.span(ATV::kAtom_x)[1] = 10.0 + r;
    f.span(ATV::kAtom_y)[1] = 10.0;
    f.span(ATV::kAtom_z)[1] = 10.0;
    f.span(ATV::kAtom_mass)[0] = f.span(ATV::kAtom_mass)[1] = 1.0;
    f.span(ATV::kAtom_type)[0] = ta;
    f.span(ATV::kAtom_type)[1] = tb;
    f.box = ATV::cpu::Box::cubic(40.0);
    return f;
}

double energy_of(cpu::Frame& f, cpu::PairLJCut pair, double cutoff) {
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, true});
    eval.set_evaluator(std::move(pair));
    eval.enable_timing(false);
    eval.build();
    eval.run(f);
    return f.meta.get(ATV::kEnergy);
}

/// Shifted LJ, computed by hand.
double lj_ref(double eps, double sigma, double r, double rc) {
    if (r >= rc)
        return 0.0;
    const double s6 = std::pow(sigma / r, 6.0);
    const double s6c = std::pow(sigma / rc, 6.0);
    return 4.0 * eps * (s6 * s6 - s6) - 4.0 * eps * (s6c * s6c - s6c);
}

} // namespace

/// A single-type table must reproduce the single-epsilon path exactly. This is the
/// equivalence check: the two code paths (hoisted SIMD, per-type scalar) must agree.
TEST(PairLJTypes, SingleTypeTableMatchesTheScalarPath) {
    const double eps = 0.0104, sigma = 3.4, rc = 8.5, r = 4.0;

    cpu::Frame f1 = dimer(0, 0, r);
    const double e_plain = energy_of(f1, cpu::PairLJCut{eps, sigma, rc}, rc);

    cpu::Frame f2 = dimer(0, 0, r);
    const double e_table =
        energy_of(f2, cpu::PairLJCut{{{0, 0, eps, sigma}}, rc}, rc);

    EXPECT_NEAR(e_plain, lj_ref(eps, sigma, r, rc), kEnergyTolerance);
    EXPECT_NEAR(e_table, e_plain, 1e-12)
        << "the per-type path disagrees with the single-epsilon path";
}

/// Unlike pairs get Lorentz-Berthelot mixing from the diagonal.
TEST(PairLJTypes, UnlikePairsMixByLorentzBerthelot) {
    const double e1 = 0.1, s1 = 3.0;
    const double e2 = 0.4, s2 = 4.0;
    const double rc = 9.0, r = 4.2;

    // Only the like-pairs are given; the (0,1) entry must be mixed.
    // Both like-pairs given at construction; the (0,1) entry must be mixed.
    auto make = [&]() {
        return cpu::PairLJCut{{{0, 0, e1, s1}, {1, 1, e2, s2}}, rc};
    };

    cpu::Frame f = dimer(0, 1, r);
    const double e_mixed = energy_of(f, make(), rc);

    const double eps_ij = std::sqrt(e1 * e2); // geometric mean
    const double sig_ij = 0.5 * (s1 + s2);    // arithmetic mean
    EXPECT_NEAR(e_mixed, lj_ref(eps_ij, sig_ij, r, rc), kEnergyTolerance);

    // ... and the like pairs still use their own coefficients.
    cpu::Frame f00 = dimer(0, 0, r);
    EXPECT_NEAR(energy_of(f00, make(), rc), lj_ref(e1, s1, r, rc),
                kEnergyTolerance);
    cpu::Frame f11 = dimer(1, 1, r);
    EXPECT_NEAR(energy_of(f11, make(), rc), lj_ref(e2, s2, r, rc),
                kEnergyTolerance);
}

/// An explicit off-diagonal overrides the mixed value.
TEST(PairLJTypes, ExplicitCrossTermOverridesMixing) {
    const double rc = 9.0, r = 4.2;
    auto pair = cpu::PairLJCut{{{0, 0, 0.1, 3.0},
                                {1, 1, 0.4, 4.0},
                                {0, 1, 0.9, 5.0}}, // nothing like the mix
                               rc};

    cpu::Frame f = dimer(0, 1, r);
    EXPECT_NEAR(energy_of(f, pair, rc), lj_ref(0.9, 5.0, r, rc),
                kEnergyTolerance);
}

/// THE POINT: a type with no LJ site (TIP3P hydrogen) must not interact.
///
/// An unset pair with no diagonal to mix from gets eps = 0 — NOT a silent fallback to
/// the single-sigma value, which would reintroduce the exact bug this exists to fix
/// (hydrogens wearing the oxygen's sigma, and a liquid-density water box exploding).
TEST(PairLJTypes, TypeWithoutAnLJSiteDoesNotInteract) {
    const double rc = 9.0;
    // Type 8 = O carries the LJ site; type 1 = H has none.
    // Per-type mode: there IS no system-wide sigma to leak into the hydrogens.
    auto pair = cpu::PairLJCut{{{8, 8, 0.1521 * 0.04336, 3.1507}}, rc};

    // H-H at 0.96 A — an intramolecular distance. Under the old single-sigma design
    // the hydrogens wore the oxygen's sigma = 3.15 A and this pair was a catastrophic
    // overlap; that is what tore a liquid-density water box apart.
    cpu::Frame hh = dimer(1, 1, 0.96);
    EXPECT_NEAR(energy_of(hh, pair, rc), 0.0, 1e-14)
        << "hydrogens are interacting — they have no LJ site";

    // O-H likewise: no diagonal for H, so nothing to mix.
    cpu::Frame oh = dimer(8, 1, 0.96);
    EXPECT_NEAR(energy_of(oh, pair, rc), 0.0, 1e-14);

    // O-O does interact, with its own coefficients.
    cpu::Frame oo = dimer(8, 8, 3.5);
    EXPECT_NEAR(energy_of(oo, pair, rc),
                lj_ref(0.1521 * 0.04336, 3.1507, 3.5, rc), kEnergyTolerance);
    EXPECT_LT(energy_of(oo, pair, rc), 0.0) << "O-O should be bound at 3.5 A";
}

/// A type above every coefficient line is an unset pair too: build() sizes the
/// table to the Frame's types instead of the kernel indexing past it.
TEST(PairLJTypes, TypeAboveTheTableDoesNotInteract) {
    const double rc = 9.0;
    const auto pair = cpu::PairLJCut{{{0, 0, 0.1, 3.0}}, rc};
    cpu::Frame oh = dimer(0, 3, 0.96);
    EXPECT_NEAR(energy_of(oh, pair, rc), 0.0, 1e-14);
    cpu::Frame hh = dimer(3, 3, 0.96);
    EXPECT_NEAR(energy_of(hh, pair, rc), 0.0, 1e-14);
}

/// A type that appears after build() (an MC insertion) is refused at the
/// neighbor rebuild that brings it in, not read past the table.
TEST(PairLJTypes, TypeIntroducedAfterBuildThrowsAtRebuild) {
    cpu::Frame f = dimer(0, 0, 4.0);
    cpu::md::PairList pairs;
    pairs.inum = 2;
    pairs.ilist = {0, 1};
    pairs.firstneigh = {0, 1};
    pairs.numneigh = {1, 0};
    pairs.neighbors = {1};
    pairs.total_pairs = 1;
    cpu::Context context;
    context.md.pair_list = &pairs;
    auto pair = cpu::PairLJCut{{{0, 0, 0.1, 3.0}}, 9.0};
    pair.build({context, f});
    EXPECT_NO_THROW(pair.launch({context, f, 0, 0, true, true}));
    f.span(ATV::kAtom_type)[1] = 5;
    pairs.version = 1;
    EXPECT_THROW(pair.launch({context, f, 0, 1, true, true}),
                 std::runtime_error);
    pair.destroy({context, f});
}

/// The per-type constructor refuses an empty table: a PairLJCut cannot EXIST in a
/// state where it has no coefficients. That is why the table is a constructor argument
/// and not something you attach afterwards.
TEST(PairLJTypes, PerTypeConstructorRefusesAnEmptyTable) {
    EXPECT_THROW(cpu::PairLJCut(std::vector<cpu::LJTypeCoeff>{}, 8.0),
                 std::invalid_argument);
}

/// set_type_coeff edits ONE cell of an existing table — a mutator, not a builder step.
TEST(PairLJTypes, SetTypeCoeffEditsOneEntry) {
    const double rc = 9.0, r = 4.2;
    auto pair = cpu::PairLJCut{{{0, 0, 0.1, 3.0}}, rc};
    pair.set_type_coeff(0, 0, 0.7, 3.6); // overwrite it

    cpu::Frame f = dimer(0, 0, r);
    EXPECT_NEAR(energy_of(f, pair, rc), lj_ref(0.7, 3.6, r, rc),
                kEnergyTolerance);
}

/// A single-type PairLJCut has no table, and says so rather than growing one silently.
TEST(PairLJTypes, SetTypeCoeffThrowsOnASingleTypePair) {
    cpu::PairLJCut pair{0.0104, 3.4, 8.5};
    EXPECT_THROW(pair.set_type_coeff(0, 0, 0.1, 3.0), std::logic_error);
}

#ifdef _OPENMP
TEST(PairLJTypes, LaunchUsesOnlyTheActiveTeamWithinBuiltCapacity) {
    const int saved_threads = omp_get_max_threads();
    const int saved_dynamic = omp_get_dynamic();
    const int saved_levels = omp_get_max_active_levels();
    omp_set_dynamic(0);
    omp_set_num_threads(2);
    omp_set_max_active_levels(1);

    // Two isolated dimers place nonzero work in both thread buffers. A later
    // serialized nested launch must not add the previous second buffer.
    cpu::Frame f(4);
    f.box = cpu::Box::cubic(40.0);
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = 5.0 + 12.0 * (i / 2) + 1.2 * (i % 2);
        f.span(kAtom_y)[i] = f.span(kAtom_z)[i] = 5.0;
        f.span(kAtom_type)[i] = 0;
    }
    cpu::md::PairList pairs;
    pairs.inum = pairs.n_rows = 4;
    pairs.ilist = {0, 1, 2, 3};
    pairs.numneigh = {1, 0, 1, 0};
    pairs.firstneigh = {0, 1, 1, 2};
    pairs.neighbors = {1, 3};
    pairs.total_pairs = 2;
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::PairLJCut pair{1.0, 1.0, 2.5};
    pair.build({ctx, f});
    const double s6 = std::pow(1.0 / 1.2, 6);
    const double force = -24.0 / 1.2 * s6 * (2.0 * s6 - 1.0);

    for (int mode = 0; mode < 4; ++mode) {
        // Defaults of 1 and 4 do not resize the object's two-thread capacity.
        omp_set_num_threads(mode == 1 ? 1 : 4);
        for (int i = 0; i < 4; ++i)
            f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        f.meta.get(kEnergy) = f.meta.get(kEnergyPair) = 0;
        if (mode == 3) {
            // Unlike OMP_DYNAMIC, max_active_levels deterministically forces
            // the component's nested parallel region down to one thread.
#pragma omp parallel num_threads(2)
            {
#pragma omp single
                pair.launch({ctx, f, 0, 0, true, false});
            }
        } else {
            pair.launch({ctx, f, 0, 0, true, false});
        }
        EXPECT_NEAR(f.meta.get(kEnergy), 2 * lj_ref(1, 1, 1.2, 2.5), 1e-12);
        for (int i = 0; i < 4; ++i) {
            EXPECT_NEAR(f.span(kAtom_fx)[i], i % 2 ? -force : force, 1e-12);
            EXPECT_DOUBLE_EQ(f.span(kAtom_fy)[i], 0);
            EXPECT_DOUBLE_EQ(f.span(kAtom_fz)[i], 0);
        }
    }
    pair.destroy({ctx, f});
    omp_set_max_active_levels(saved_levels);
    omp_set_num_threads(saved_threads);
    omp_set_dynamic(saved_dynamic);
}
#endif
