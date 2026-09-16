/// @file tests/cpu/path/test_remd.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/driver/dof.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/path/remd.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {
void build_lj(cpu::md::MD& md, double /*T*/, void*) {
    md.set_evaluator(cpu::PairLJCut{0.0104, 3.4, 8.5});
    md.set_neighbor(cpu::md::NeighborList<>{8.5, 0.3});
}

/// Argon-ish simple-cubic block, dense enough that the pair energy is real.
cpu::Frame lj_block(int side, double spacing) {
    const int n = side * side * side;
    cpu::Frame f(n);
    int row = 0;
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k) {
                f.span(kAtom_x)[row] = (i + 0.5) * spacing;
                f.span(kAtom_y)[row] = (j + 0.5) * spacing;
                f.span(kAtom_z)[row] = (k + 0.5) * spacing;
                f.span(kAtom_mass)[row] = 39.948;
                f.span(kAtom_type)[row] = 0;
                f.span(kAtom_tag)[row] = row + 1;
                ++row;
            }
    f.box = cpu::Box::cubic(side * spacing);
    f.rebuild_tag_index(0, n);
    return f;
}
} // namespace

TEST(Remd, BuildsAndRunsExchangeRounds) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 4.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 39.948;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(20.0);

    cpu::path::Remd remd({100.0, 200.0, 300.0, 400.0}, 1.0, 5, 42);
    remd.set_program_builder(&build_lj, nullptr);
    remd.run(f, /*n_exchanges=*/4);
    EXPECT_GT(remd.exchange_attempts(), 0);
    // Accepts may be 0 for short run; still a valid outcome.
    EXPECT_GE(remd.exchange_accepts(), 0);
    EXPECT_LE(remd.exchange_accepts(), remd.exchange_attempts());
    EXPECT_EQ(f.nlocal(), 2);
}

TEST(Remd, RejectsNonMonotoneTemperatures) {
    EXPECT_THROW(cpu::path::Remd({300.0, 200.0}, 1.0, 1, 1),
                 std::invalid_argument);
}

TEST(Remd, RejectsNonPositiveThermostatTau) {
    cpu::Frame f = lj_block(2, 4.0);
    cpu::path::Remd remd({100.0, 200.0}, 1.0, 1, 1);
    remd.set_program_builder(&build_lj, nullptr);
    remd.thermostat_tau = -1.0;
    EXPECT_THROW(remd.run(f, 1), std::invalid_argument);
}

/// Every replica must sample the CANONICAL ensemble at its own slot
/// temperature — that is the premise the Sugita–Okamoto exchange test rests
/// on, and it is why `Remd` attaches a CSVR thermostat at `temperatures[i]`.
///
/// The discriminator: start the block **compressed** onto the Lennard-Jones
/// repulsive wall (3.0 Å against σ = 3.4 Å). Relaxing off that wall dumps a
/// large potential energy into the velocities. A thermostatted replica gives
/// it away and stays at its setpoint; a constant-energy one keeps it and runs
/// nearly three times hot. Measured: 270 K with the thermostat, 862 K without.
TEST(Remd, ReplicasAreThermostattedAtTheirSlotTemperature) {
    const double t0 = 300.0;
    cpu::Frame f = lj_block(4, 3.0); // 64 atoms, 189 DOF, compressed
    cpu::path::Remd remd({t0, 2.0 * t0}, /*dt=*/1.0,
                         /*steps_per_exchange=*/1000, /*seed=*/5);
    remd.thermostat_tau = 20.0; // 20 dt — well converged over the run
    remd.set_program_builder(&build_lj, nullptr);
    remd.run(f, /*n_exchanges=*/1);

    const double ke = f.meta.get(ATV::kKinetic); // [eV], slot 0
    const double ndof = 3.0 * f.nlocal() - 3.0;
    const double t_meas = 2.0 * ke / (ndof * units::kB_ev_per_k);

    // sqrt(2/Ndof) = 10% instantaneous spread at 189 DOF; 25% is a couple of
    // those and nowhere near the factor 2.9 a constant-energy replica shows.
    EXPECT_NEAR(t_meas, t0, 0.25 * t0)
        << "collapsed slot 0 came back at " << t_meas
        << " K for a 300 K setpoint";
}
