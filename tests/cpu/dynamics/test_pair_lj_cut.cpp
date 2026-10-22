/// @file tests/cpu/md/test_pair_lj_cut.cpp
/// @brief Regression test for the canonical CPU PairLJCut.
///
/// Verifies that the class resolves and that energy + per-atom forces on a two-atom test
/// match the documented LJ 12-6 formula bit-for-bit.

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

TEST(PairLJCutPort, NewClassExists) {
    ATV::cpu::PairLJCut lj{0.1, 3.0, 8.0};
    EXPECT_EQ(lj.name(), "lj");
    EXPECT_DOUBLE_EQ(lj.pair_cutoff(), 8.0);
}

TEST(PairLJCutPort, FilteredSubsetPreservesExplicitCentreRows) {
    using namespace ATV;
    cpu::Frame frame(3);
    frame.box = cpu::Box::cubic(20.0);
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_x)[i] = 1.5 * i;
        frame.span(kAtom_y)[i] = frame.span(kAtom_z)[i] = 0.0;
        frame.span(kAtom_tag)[i] = i + 1;
    }
    cpu::md::PairList pairs;
    pairs.inum = 1;
    pairs.n_rows = 3;
    pairs.ilist = {2};
    pairs.firstneigh = {0, 0, 0};
    pairs.numneigh = {0, 0, 1};
    pairs.neighbors = {0};
    pairs.total_pairs = 1;
    cpu::Context context;
    context.md.pair_list = &pairs;
    cpu::PairLJCut potential(0.2, 1.0, 4.0);
    potential.set_special_bonds(0.0, 0.0, 0.0); // activate filtering, no bonds
    potential.build({context, frame});
    potential.launch({context, frame, 0, 0, true, true});
    const double r6 = std::pow(1.0 / 3.0, 6);
    const double cut6 = std::pow(1.0 / 4.0, 6);
    const double expected_energy = 0.8 * (r6 * r6 - r6 - cut6 * cut6 + cut6);
    const double expected_fx = 4.8 * (2 * r6 * r6 - r6) / 3.0;
    EXPECT_NEAR(frame.meta.get(kEnergy), expected_energy, 1e-14);
    EXPECT_NEAR(frame.span(kAtom_fx)[2], expected_fx, 1e-14);
    EXPECT_NEAR(frame.span(kAtom_fx)[0], -expected_fx, 1e-14);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[1], 0.0);
    potential.destroy({context, frame});
}

TEST(PairLJCutPort, EnergyMatchesAnalyticFormula) {
    // Two atoms at r = sigma → V = 4ε(1 − 1) = 0.
    const double eps = 0.5;
    const double sigma = 3.0;
    const double rcut = 8.0;

    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = sigma;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.0;
    frame.span(ATV::kAtom_mass)[0] = 1.0;
    frame.span(ATV::kAtom_mass)[1] = 1.0;
    frame.box = ATV::cpu::Box::cubic(50.0);

    ATV::cpu::md::MD sim;
    sim.set_neighbor(ATV::cpu::md::NeighborList<>{rcut, /*skin=*/0.5});
    sim.set_evaluator(ATV::cpu::PairLJCut{eps, sigma, rcut});
    sim.set_integrator(ATV::cpu::md::VelocityVerlet{/*dt=*/0.001});
    sim.build();
    // 1 integration step at dt=0.001 fs with zero initial velocities ≈
    // 0 displacement; the launch records the energy at the starting
    // geometry while leaving positions essentially untouched.
    sim.run(frame, 1);

    // Energy at r≈σ with shifted potential:
    //   V = 4ε[(σ/r)¹² − (σ/r)⁶] − V_shift
    // At r=σ, (σ/r)⁶ = 1, so the first bracket is 0 → V = −V_shift.
    const double sr_cut = sigma / rcut;
    const double sr_cut6 = sr_cut * sr_cut * sr_cut * sr_cut * sr_cut * sr_cut;
    const double sr_cut12 = sr_cut6 * sr_cut6;
    const double v_shift = 4.0 * eps * (sr_cut12 - sr_cut6);
    const double expected = -v_shift;
    // 1e-6 absolute tolerance covers the tiny displacement during the
    // single integration step (atoms move ~ a·dt²/2 ~ 1e-10 Å with
    // initial force ≈ 0 at r=σ).
    EXPECT_NEAR(frame.meta.get(ATV::kEnergy), expected, 1e-6)
        << "PairLJCut at r≈σ should yield −V_shift = " << expected;
}
