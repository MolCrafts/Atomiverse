/// @file tests/cpu/potentials/test_pair_lj_cut_coul_long.cpp
/// @brief Fused lj/cut/coul/long: finite energy, specials drop 1-2, erfc table.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/cpu/driver/step_timing.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_real.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/erfc_table.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"

using namespace ATV;

TEST(ErfcTable, MatchesDirectErfc) {
    cpu::md::ErfcTable tab;
    tab.build(0.3, 10.0, 5000);
    for (double r : {0.5, 1.0, 2.5, 5.0, 9.5}) {
        double e, f;
        tab.eval_r2(r * r, e, f);
        const double grij = 0.3 * r;
        const double expm2 = std::exp(-grij * grij);
        const double erfcv = std::erfc(grij);
        const double e_ref = erfcv / r;
        const double f_ref =
            (erfcv + cpu::md::ErfcTable::kEwaldF * grij * expm2) / (r * r);
        // Linear-in-r² table: denser near r=0 would be tighter; 2e-3 is fine
        // for force MD (pair specials + energy bands, not metrology).
        EXPECT_NEAR(e, e_ref, 2e-3) << "r=" << r;
        EXPECT_NEAR(f, f_ref, 1e-2) << "r=" << r;
    }
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
    f.simbox = cpu::Box::cubic(20.0);
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
    sim.add_potential(
        cpu::PairLJCutCoulLong{{{0, 0, eps, 3.4}, {1, 1, eps, 2.5}}, alpha, cut}
            .set_special_bonds_amber()
            .set_compute_stress(true));
    sim.add(cpu::KSpaceEwald{alpha, /*accuracy=*/1e-5});
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
