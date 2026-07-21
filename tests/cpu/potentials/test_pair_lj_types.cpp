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

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/driver/evaluate.h"
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
    f.simbox = ATV::cpu::Box::cubic(40.0);
    return f;
}

double energy_of(cpu::Frame& f, cpu::PairLJCut pair, double cutoff) {
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, true});
    eval.add_potential(std::move(pair));
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
