/// @file tests/cpu/path/test_hremd.cpp
#include <gtest/gtest.h>

#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/path/hremd.h"
#include "atomiverse/cpu/potentials/pair_lj_soft.h"

using namespace ATV;

namespace {

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

void build_lj_soft(cpu::md::MD& md, double lambda, void*) {
    cpu::PairLJSoft soft(0.0104, 3.4, 8.5);
    soft.lambda = lambda;
    md.set_evaluator(std::move(soft));
    md.set_neighbor(cpu::md::NeighborList<>{8.5, 0.3});
}

double energy_lj_soft(const cpu::Frame& frame, double lambda, void*) {
    cpu::Frame copy = frame;
    cpu::Evaluate eval;
    cpu::PairLJSoft soft(0.0104, 3.4, 8.5);
    soft.lambda = lambda;
    eval.set_evaluator(std::move(soft));
    eval.set_neighbor(cpu::md::NeighborList<>{8.5, 0.3});
    eval.build();
    eval.run(copy);
    return copy.meta.get(kEnergy);
}

void build_soft(cpu::md::MD& md, double lambda, void*) {
    cpu::PairLJSoft soft(0.0104, 3.4, 8.5);
    soft.lambda = lambda;
    md.set_evaluator(std::move(soft));
    // No neighbor for soft all-pairs path? PairLJSoft may need neighbor —
    // check: if it requires pair_list, set neighbor.
}

double energy_soft(const cpu::Frame& frame, double lambda, void*) {
    cpu::Frame f = frame; // copy for evaluate
    cpu::Evaluate eval;
    cpu::PairLJSoft soft(0.0104, 3.4, 8.5);
    soft.lambda = lambda;
    eval.set_evaluator(std::move(soft));
    eval.build();
    eval.run(f);
    return f.meta.get(kEnergy);
}

} // namespace

TEST(HRemd, ConstructsAndRunsTwoLambda) {
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
    f.box = cpu::Box::cubic(30.0);

    // PairLJSoft may need NeighborList — try without first; if MD validate fails
    // we'll fix in build.
    cpu::path::HRemd h({0.0, 1.0}, /*T=*/300.0, /*dt=*/1.0, /*steps=*/5,
                       /*seed=*/3);
    h.set_program_builder(
        [](cpu::md::MD& md, double lambda, void*) {
            cpu::PairLJSoft soft(0.0104, 3.4, 8.5);
            soft.lambda = lambda;
            md.set_evaluator(std::move(soft));
            md.set_neighbor(cpu::md::NeighborList<>{8.5, 0.3});
        },
        nullptr);
    h.set_energy_fn(
        [](const cpu::Frame& fr, double lambda, void*) {
            cpu::Frame copy = fr;
            cpu::Evaluate eval;
            cpu::PairLJSoft soft(0.0104, 3.4, 8.5);
            soft.lambda = lambda;
            eval.set_evaluator(std::move(soft));
            eval.set_neighbor(cpu::md::NeighborList<>{8.5, 0.3});
            eval.build();
            eval.run(copy);
            return copy.meta.get(kEnergy);
        },
        nullptr);
    h.run(f, 4);
    EXPECT_GE(h.exchange_attempts(), 2);
    EXPECT_EQ(f.nlocal(), 2);
    (void)build_soft;
    (void)energy_soft;
}

TEST(HRemd, RejectsNonPositiveThermostatTau) {
    auto make = [](double tau) {
        cpu::path::HRemd h({0.0, 1.0}, /*T=*/300.0, /*dt=*/1.0, /*steps=*/5,
                           /*seed=*/3);
        h.thermostat_tau = tau;
        return h;
    };
    EXPECT_NO_THROW(make(50.0));
    cpu::Frame f = lj_block(2, 3.0);
    cpu::path::HRemd bad = make(0.0);
    bad.set_program_builder(&build_lj_soft, nullptr);
    bad.set_energy_fn(&energy_lj_soft, nullptr);
    EXPECT_THROW(bad.run(f, 1), std::invalid_argument);
}

/// The exchange rule is Metropolis between two CANONICAL ensembles at the same
/// β. Constant-energy replicas do not sample one: each keeps whatever its
/// velocity draw gave it, and the acceptance ratio stops measuring anything.
/// Started from a block compressed onto the repulsive wall, an unthermostatted
/// replica converts that potential energy into heat and comes back far above
/// the setpoint; a CSVR-coupled one comes back at the setpoint.
TEST(HRemd, ReplicasAreThermostattedAtTheFixedTemperature) {
    const double t0 = 300.0;
    cpu::Frame f = lj_block(4, 3.0); // 64 atoms, 189 DOF, compressed
    cpu::path::HRemd h({0.0, 1.0}, t0, /*dt=*/1.0,
                       /*steps_per_exchange=*/1000, /*seed=*/5);
    h.thermostat_tau = 20.0; // 20 dt — well converged over the run
    h.set_program_builder(&build_lj_soft, nullptr);
    h.set_energy_fn(&energy_lj_soft, nullptr);
    h.run(f, /*n_exchanges=*/1);

    const double ke = f.meta.get(ATV::kKinetic); // [eV], slot 0
    const double ndof = 3.0 * f.nlocal() - 3.0;
    const double t_meas = 2.0 * ke / (ndof * units::kB_ev_per_k);

    // sqrt(2/Ndof) = 10% instantaneous spread at 189 DOF; 25% is a couple of
    // those and nowhere near where an unthermostatted replica lands.
    EXPECT_NEAR(t_meas, t0, 0.25 * t0)
        << "compressed slot came back at " << t_meas << " K for a " << t0
        << " K setpoint";
}
