/// @file regressions/md/settle_tip3p/reg_settle_tip3p.cpp
/// @brief END-TO-END REGRESSION: SETTLE holds two TIP3P waters rigid for 1 ps
///        at dt = 2 fs, AND leaves a free rigid molecule's motion alone.
///
/// See REFERENCE.md. Out-of-library consumer, one flat main().

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_settle.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

/// Minimum-image distance between two rows [Å]. Intramolecular distances need
/// it as much as intermolecular ones: the integrator wraps coordinates, so a
/// molecule sitting on a box face has its hydrogens an image away.
double dist(const cpu::Frame& f, int i, int j) {
    double dx = f.span(kAtom_x)[i] - f.span(kAtom_x)[j];
    double dy = f.span(kAtom_y)[i] - f.span(kAtom_y)[j];
    double dz = f.span(kAtom_z)[i] - f.span(kAtom_z)[j];
    f.box.min_image(dx, dy, dz);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/// Row of a tag. Rows are not stable — `atom_modify sort` spatially reorders
/// owned atoms at reneighbour — so an out-of-library consumer that wants a
/// named atom must ask the Frame, exactly as the engine's own components do.
int row_of(const cpu::Frame& f, TagInt tag) {
    const auto& index = f.meta.get(cpu::kTagIndex);
    const auto it = index.find(tag);
    if (it == index.end()) {
        std::printf("FAIL: tag %lld vanished from the frame\n",
                    static_cast<long long>(tag));
        std::exit(1);
    }
    return it->second;
}

} // namespace

int main() {
    // TIP3P geometry (Jorgensen et al., J. Chem. Phys. 79, 926 (1983)).
    const double d_oh = 0.9572;
    const double d_hh = 1.5139;
    const double half = 0.5 * d_hh;
    const double h = std::sqrt(d_oh * d_oh - half * half);
    const double dt = 2.0;      // [fs] — only a RIGID water survives this
    const double cutoff = 8.0;  // [Å]
    const double box_len = 24.0;
    const int n_steps = 500;    // 1 ps

    // Two waters, 3.5 Å apart: close enough that the oxygen-oxygen LJ is a
    // real force (and a real torque on each molecule), far enough that the
    // pair is not in the repulsive wall.
    cpu::Frame f(6);
    const double origin[2][3] = {{10.0, 10.0, 10.0}, {13.5, 10.3, 9.8}};
    for (int m = 0; m < 2; ++m) {
        const int o = 3 * m, a = o + 1, b = o + 2;
        // Second molecule is rolled 90° about x so the two are not parallel.
        const double s = (m == 0) ? 1.0 : 0.0;
        const double c = (m == 0) ? 0.0 : 1.0;
        f.span(kAtom_x)[o] = origin[m][0];
        f.span(kAtom_y)[o] = origin[m][1] + s * h;
        f.span(kAtom_z)[o] = origin[m][2] + c * h;
        f.span(kAtom_x)[a] = origin[m][0] - half;
        f.span(kAtom_y)[a] = origin[m][1];
        f.span(kAtom_z)[a] = origin[m][2];
        f.span(kAtom_x)[b] = origin[m][0] + half;
        f.span(kAtom_y)[b] = origin[m][1];
        f.span(kAtom_z)[b] = origin[m][2];
        f.span(kAtom_mass)[o] = 15.9994;
        f.span(kAtom_mass)[a] = 1.008;
        f.span(kAtom_mass)[b] = 1.008;
        f.span(kAtom_type)[o] = 8;
        f.span(kAtom_type)[a] = 1;
        f.span(kAtom_type)[b] = 1;
        f.span(kAtom_tag)[o] = 3 * m + 1;
        f.span(kAtom_tag)[a] = 3 * m + 2;
        f.span(kAtom_tag)[b] = 3 * m + 3;
    }
    // Kick molecule 0 so the pair also rotates, not just translates.
    f.span(kAtom_vx)[0] = 0.004;
    f.span(kAtom_vy)[1] = -0.003;
    f.span(kAtom_vz)[2] = 0.002;
    f.box = cpu::Box::cubic(box_len);
    f.rebuild_tag_index(0, f.nlocal());

    // TIP3P excludes the intramolecular non-bonded terms; the topology is what
    // says which pairs those are.
    for (int m = 0; m < 2; ++m) {
        const TagInt to = 3 * m + 1, ta = 3 * m + 2, tb = 3 * m + 3;
        f.add_bond(to, ta, 0);
        f.add_bond(to, tb, 0);
        f.add_angle(ta, to, tb, 0);
    }
    f.build_special_list();

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    // TIP3P puts the Lennard-Jones site on OXYGEN only (ε = 0.1521 kcal/mol,
    // σ = 3.1507 Å); hydrogen has none, so every pair touching an H is zero.
    sim.set_evaluator(
        cpu::PairLJCut{{{8, 8, 0.1521 * 0.04336, 3.1507}}, cutoff}
            .set_special_bonds(0.0, 0.0, 0.0));
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.5});
    cpu::md::FixSettle settle(dt, d_oh, d_hh);
    settle.add_water(1, 2, 3);
    settle.add_water(4, 5, 6);
    sim.set_observer(std::move(settle));
    sim.enable_timing(false);
    sim.build();
    sim.run(f, n_steps);

    double worst = 0.0;
    for (int m = 0; m < 2; ++m) {
        const int o = row_of(f, 3 * m + 1);
        const int a = row_of(f, 3 * m + 2);
        const int b = row_of(f, 3 * m + 3);
        const int ii[3] = {o, o, a}, jj[3] = {a, b, b};
        const double want[3] = {d_oh, d_oh, d_hh};
        for (int c = 0; c < 3; ++c) {
            const double e = std::fabs(dist(f, ii[c], jj[c]) - want[c])
                             / want[c];
            std::printf("  mol %d constraint %d: rel err %.3e\n", m, c, e);
            worst = std::fmax(worst, e);
        }
    }
    std::printf("SETTLE: %d steps x %.1f fs, worst rel geometry err = %.3e "
                "(tol 1e-10)\n",
                n_steps, dt, worst);
    if (!(worst < 1e-10)) {
        std::printf("FAIL: the constrained geometry drifted\n");
        return 1;
    }

    // SETTLE is the closed-form SHAKE solution, not a reshaping of the
    // unconstrained triangle: on a molecule that is already rigid and moving
    // freely it must do NOTHING. Integrate one isolated water with no forces
    // and compare against the exact free-rigid-body answer — a pure
    // translation of the centre of mass.
    cpu::Frame g(3);
    g.span(kAtom_x)[0] = 5.0;
    g.span(kAtom_y)[0] = 5.0 + h;
    g.span(kAtom_z)[0] = 5.0;
    g.span(kAtom_x)[1] = 5.0 - half;
    g.span(kAtom_y)[1] = 5.0;
    g.span(kAtom_z)[1] = 5.0;
    g.span(kAtom_x)[2] = 5.0 + half;
    g.span(kAtom_y)[2] = 5.0;
    g.span(kAtom_z)[2] = 5.0;
    const double v_com[3] = {0.002, -0.001, 0.0015}; // [Å/fs]
    for (int i = 0; i < 3; ++i) {
        g.span(kAtom_mass)[i] = (i == 0) ? 15.9994 : 1.008;
        g.span(kAtom_type)[i] = (i == 0) ? 8 : 1;
        g.span(kAtom_tag)[i] = i + 1;
        g.span(kAtom_vx)[i] = v_com[0];
        g.span(kAtom_vy)[i] = v_com[1];
        g.span(kAtom_vz)[i] = v_com[2];
    }
    g.box = cpu::Box::cubic(box_len);
    g.rebuild_tag_index(0, g.nlocal());
    double x0[3][3];
    for (int i = 0; i < 3; ++i) {
        x0[i][0] = g.span(kAtom_x)[i];
        x0[i][1] = g.span(kAtom_y)[i];
        x0[i][2] = g.span(kAtom_z)[i];
    }

    cpu::md::MD drift;
    drift.set_integrator(cpu::md::VelocityVerlet{dt});
    drift.set_evaluator(
        cpu::PairLJCut{{{8, 8, 0.1521 * 0.04336, 3.1507}}, cutoff});
    drift.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.5});
    cpu::md::FixSettle free_settle(dt, d_oh, d_hh);
    free_settle.add_water(1, 2, 3);
    drift.set_observer(std::move(free_settle));
    drift.enable_timing(false);
    drift.build();
    drift.run(g, n_steps);

    double worst_drift = 0.0;
    for (int i = 0; i < 3; ++i) {
        const int r = row_of(g, i + 1);
        const double want[3] = {x0[i][0] + v_com[0] * dt * n_steps,
                                x0[i][1] + v_com[1] * dt * n_steps,
                                x0[i][2] + v_com[2] * dt * n_steps};
        const double got[3] = {g.span(kAtom_x)[r], g.span(kAtom_y)[r],
                               g.span(kAtom_z)[r]};
        for (int d = 0; d < 3; ++d) {
            double e = got[d] - want[d];
            // The integrator wraps into the box; the exact answer does not.
            e -= box_len * std::round(e / box_len);
            worst_drift = std::fmax(worst_drift, std::fabs(e));
        }
    }
    std::printf("free rigid drift: worst |r - r_exact| = %.3e A (tol 1e-9)\n",
                worst_drift);
    if (!(worst_drift < 1e-9)) {
        std::printf("FAIL: SETTLE perturbed a molecule that was already "
                    "rigid — it is reshaping the triangle, not solving the "
                    "constraint\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
