/// @file regressions/md/qeq-fluctuating-charge/reg_qeq-fluctuating-charge.cpp
/// @brief QEq fluctuating-charge equilibration vs LAMMPS `fix qeq/point`.
///
/// Out-of-library consumer of the public API. H2CO-like 4-atom organic toy;
/// per-atom charges and coul/cut energy pinned from an offline LAMMPS run
/// (see REFERENCE.md; ctest never launches LAMMPS).

#include <chrono>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/fix_addforce.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/qeq.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

namespace {

int g_fail = 0;

void check(const char* what, double got, double want, double tol) {
    const double err = std::abs(got - want);
    const bool ok = err <= tol;
    std::printf("%-34s got=%+.12f want=%+.12f |err|=%.2e %s\n", what, got,
                want, err, ok ? "OK" : "FAIL");
    if (!ok)
        ++g_fail;
}

} // namespace

int main() {
    // Geometry identical to tools/metalwalls_oracle/inputs/lammps/
    // qeq-fluctuating-charge/in.qeq_point (LAMMPS `boundary f f f`, 40 Å box).
    const double xs[4] = {20.0, 20.0, 20.943, 19.057};
    const double ys[4] = {20.0, 20.0, 20.0, 20.0};
    const double zs[4] = {20.0, 21.205, 19.412, 19.412};
    const int types[4] = {1, 2, 3, 3};
    const double masses[4] = {12.011, 15.999, 1.008, 1.008};

    cpu::Frame f(4);
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = xs[i];
        f.span(kAtom_y)[i] = ys[i];
        f.span(kAtom_z)[i] = zs[i];
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_type)[i] = types[i];
        f.span(kAtom_mass)[i] = masses[i];
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0.0;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0.0;
    }
    f.box = cpu::Box::cubic(40.0);
    f.meta.get(kEnergy) = 0.0;

    // QEq parameters — identical to param.qeq (chi [eV], eta [eV/e²]).
    cpu::md::QEq qeq(/*cutoff [Å]=*/10.0);
    qeq.set_type(1, 5.343, 10.126);   // C
    qeq.set_type(2, 8.741, 13.364);   // O
    qeq.set_type(3, 4.528, 13.8904);  // H

    // QEq at pre_force, then coul/cut on the equilibrated charges.
    {
        cpu::md::MD sim;
        sim.set_neighbor(cpu::md::NeighborList<>{10.0, 0.5});
        sim.set_evaluator(cpu::PairCoulCut{/*cutoff [Å]=*/10.0});
        sim.set_integrator(cpu::md::VelocityVerlet{0.0});
        sim.set_observer_ref(qeq);
        sim.build();
        sim.run(f, 1);
    }

    // ── Pins: LAMMPS 30 Mar 2026 fix qeq/point dump (REFERENCE.md) ────
    check("q(C)  [e]", qeq.charges()[0], 0.046279970820, 1e-6);
    check("q(O)  [e]", qeq.charges()[1], -0.230795571743, 1e-6);
    check("q(H1) [e]", qeq.charges()[2], 0.092257800461, 1e-6);
    check("q(H2) [e]", qeq.charges()[3], 0.092257800461, 1e-6);
    check("sum q [e]", qeq.net_charge(), 0.0, 1e-12);
    // Derived QEq objective (bare-1/r kernel), see REFERENCE.md.
    check("E_qeq [eV]", qeq.qeq_energy(), -0.467311783767, 1e-6);

    // Coulomb energy of the equilibrated charges (from the same MD step).
    // LAMMPS thermo E_coul (metal units, qqr2e = 14.399645 = units::coulomb).
    check("E_coul [eV]", f.meta.get(kEnergy), -0.254699311955, 1e-6);

    // ── Fair perf twin: 1000 repeated equilibrations (see in.qeq_perf) ─
    // QEq only — dummy Bias satisfies MD validation (no pair cost).
    // Continuable MD::run(f,1) keeps QEq component state across solves.
    const int n_rep = 1000;
    cpu::md::MD perf;
    perf.set_evaluator(cpu::md::FixAddForce{0.0, 0.0, 0.0, /*type=*/1});
    perf.set_integrator(cpu::md::VelocityVerlet{0.0});
    perf.set_observer_ref(qeq);
    perf.build();
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < n_rep; ++rep)
        perf.run(f, 1);
    const auto t1 = std::chrono::steady_clock::now();
    const double t_atv =
        std::chrono::duration<double>(t1 - t0).count();
    const double t_lmp = 0.000610294; // LAMMPS loop time, 1000 solves (pin)
    std::printf("perf: T_atv=%.6f s  T_lmp=%.6f s  ratio=%.2f (1000 solves)\n",
                t_atv, t_lmp, t_atv / t_lmp);

    if (g_fail == 0)
        std::printf("qeq-fluctuating-charge: all pins OK\n");
    return g_fail == 0 ? 0 : 1;
}
