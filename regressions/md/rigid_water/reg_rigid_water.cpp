/// @file regressions/md/rigid_water/reg_rigid_water.cpp
/// @brief END-TO-END REGRESSION: RATTLE rigid water — the geometry invariant and
///        the equipartition theorem, which together pin the degree-of-freedom count.
///
/// See REFERENCE.md. Out-of-library consumer, one flat main().

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // TIP3P geometry (Jorgensen 1983): r(OH) = 0.9572 A, theta = 104.52 deg.
    const double r_oh = 0.9572;
    const double theta_deg = 104.52;
    const double d_hh = 2.0 * r_oh * std::sin(0.5 * theta_deg * M_PI / 180.0);

    const int n_side = 7; // 343 molecules; box 21.7 A > 2 * cutoff
    // LIQUID density now. The LJ site sits on oxygen ONLY (TIP3P), which needs a
    // per-type coefficient table; before PairLJCut had one, the hydrogens were forced
    // to wear the oxygen's sigma = 3.15 A and a box at this spacing tore itself apart.
    const double spacing = 3.10;           // ~ liquid water number density
    const double cutoff = 8.0, skin = 1.0; // [A]
    const double dt = 2.0;         // [fs] — only a RIGID water survives this
    const double t_target = 300.0; // [K]
    const int n_equil = 2000, n_prod = 3000, n_sample = 30;

    const int n_mol = n_side * n_side * n_side;
    const int n = 3 * n_mol;
    const double box_len = n_side * spacing;
    if (box_len <= 2.0 * cutoff) {
        std::printf("FAIL: box %.2f A too small for cutoff %.2f A\n", box_len,
                    cutoff);
        return 1;
    }

    // Build the box of rigid-geometry waters.
    cpu::Frame frame(n);
    const double half = 0.5 * theta_deg * M_PI / 180.0;
    int idx = 0, mol = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                const double ox = (ix + 0.5) * spacing;
                const double oy = (iy + 0.5) * spacing;
                const double oz = (iz + 0.5) * spacing;
                const double phi =
                    0.7 * double(mol); // stagger the orientations
                const int o = idx, h1 = idx + 1, h2 = idx + 2;

                frame.span(ATV::kAtom_x)[o] = ox;
                frame.span(ATV::kAtom_y)[o] = oy;
                frame.span(ATV::kAtom_z)[o] = oz;
                frame.span(ATV::kAtom_mass)[o] = 15.9994;
                frame.span(ATV::kAtom_type)[o] = 8;

                frame.span(ATV::kAtom_x)[h1] =
                    ox + r_oh * std::sin(half) * std::cos(phi);
                frame.span(ATV::kAtom_y)[h1] =
                    oy + r_oh * std::sin(half) * std::sin(phi);
                frame.span(ATV::kAtom_z)[h1] = oz + r_oh * std::cos(half);
                frame.span(ATV::kAtom_mass)[h1] = 1.008;
                frame.span(ATV::kAtom_type)[h1] = 1;

                frame.span(ATV::kAtom_x)[h2] =
                    ox - r_oh * std::sin(half) * std::cos(phi);
                frame.span(ATV::kAtom_y)[h2] =
                    oy - r_oh * std::sin(half) * std::sin(phi);
                frame.span(ATV::kAtom_z)[h2] = oz + r_oh * std::cos(half);
                frame.span(ATV::kAtom_mass)[h2] = 1.008;
                frame.span(ATV::kAtom_type)[h2] = 1;

                idx += 3;
                ++mol;
            }
    frame.box = ATV::cpu::Box::cubic(box_len);
    frame.rebuild_tag_index(0, n);
    for (int m = 0; m < n_mol; ++m) {
        const int o = 3 * m, h1 = o + 1, h2 = o + 2;
        frame.add_bond(frame.span(ATV::kAtom_tag)[o],
                       frame.span(ATV::kAtom_tag)[h1], 0);
        frame.add_bond(frame.span(ATV::kAtom_tag)[o],
                       frame.span(ATV::kAtom_tag)[h2], 0);
        frame.add_angle(frame.span(ATV::kAtom_tag)[h1],
                        frame.span(ATV::kAtom_tag)[o],
                        frame.span(ATV::kAtom_tag)[h2], 0);
    }
    frame.build_special_list(); // intramolecular pairs are NOT LJ neighbours

    // Remember which TAGS make up each molecule. Rows are not an identity:
    // `atom_modify sort` spatially reorders the owned atoms at the first
    // reneighbour after `nfreq` steps (LAMMPS default 1000), so row `3m` stops
    // being molecule `m`'s oxygen part-way through the run. The engine's own
    // constraint solver keys on tags for exactly this reason; a consumer that
    // wants a named atom back must ask the Frame's tag index.
    std::vector<ATV::TagInt> mol_tags(3 * n_mol);
    for (int m = 0; m < n_mol; ++m)
        for (int k = 0; k < 3; ++k)
            mol_tags[3 * m + k] = frame.span(ATV::kAtom_tag)[3 * m + k];

    cpu::md::MD sim;
    // TIP3P: the Lennard-Jones site is on OXYGEN only. Hydrogen has none, so the
    // (H,H) and (O,H) pairs mix to eps = 0 and do not interact at all.
    sim.set_evaluator(cpu::PairLJCut{{{8, 8, 0.1521 * 0.04336, 3.1507}}, cutoff}
                          // O-O
                          .set_special_bonds(0.0, 0.0, 0.0));
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_thermostat(cpu::md::FixCSVR{t_target, 100.0, dt, 7777u});
    sim.set_observer(cpu::md::InitVelocity{t_target, 4242u});

    cpu::md::FixRattle rattle{dt};
    rattle.constrain_bond(0, r_oh);  // the two O-H bonds
    rattle.constrain_angle(0, d_hh); // the H-H distance IS the angle
    sim.set_observer_ref(rattle);

    sim.enable_timing(false);
    sim.build();

    sim.run(frame, n_equil);

    // Sample the temperature, computed with the DOF count the engine published.
    // Rigid water: 6 DOF per molecule (3 translation + 3 rotation), not 9.
    const int ndof_rigid = 6 * n_mol - 3;
    const int ndof_naive =
        3 * n - 3; // what every consumer hardcoded before this work

    double t_sum = 0.0, worst_geom = 0.0;
    for (int s = 0; s < n_sample; ++s) {
        sim.run(frame, n_prod / n_sample);

        const double ke = frame.meta.get(ATV::kKinetic); // [eV]
        t_sum += 2.0 * ke / (double(ndof_rigid) * units::kB_ev_per_k);

        const auto& tag_row = frame.meta.get(ATV::cpu::kTagIndex);
        for (int m = 0; m < n_mol; ++m) {
            const int o = tag_row.at(mol_tags[3 * m + 0]);
            const int h1 = tag_row.at(mol_tags[3 * m + 1]);
            const int h2 = tag_row.at(mol_tags[3 * m + 2]);
            const int ii[3] = {o, o, h1};
            const int jj[3] = {h1, h2, h2};
            const double want[3] = {r_oh, r_oh, d_hh};
            for (int c = 0; c < 3; ++c) {
                double dx = frame.span(ATV::kAtom_x)[ii[c]]
                            - frame.span(ATV::kAtom_x)[jj[c]];
                double dy = frame.span(ATV::kAtom_y)[ii[c]]
                            - frame.span(ATV::kAtom_y)[jj[c]];
                double dz = frame.span(ATV::kAtom_z)[ii[c]]
                            - frame.span(ATV::kAtom_z)[jj[c]];
                frame.box.min_image(dx, dy, dz);
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                worst_geom =
                    std::fmax(worst_geom, std::fabs(r - want[c]) / want[c]);
            }
        }
    }
    const double t_mean = t_sum / double(n_sample);
    // What the SAME trajectory would report if Ndof still ignored the constraints.
    const double t_naive = t_mean * double(ndof_rigid) / double(ndof_naive);

    const double geom_tol = 1e-8;
    const double t_tol = 0.03; // 3% — a thermostatted mean over 6 ps
    const double t_err = std::fabs(t_mean - t_target) / t_target;

    std::printf(
        "RATTLE rigid water: %d molecules, dt = %.1f fs, T_set = %.0f K\n",
        n_mol, dt, t_target);
    std::printf("  Ndof (rigid, 6/mol)  = %d      [naive 3N-3 would be %d]\n",
                ndof_rigid, ndof_naive);
    std::printf("  <T>                  = %.2f K   want %.0f K  (err %.2f%%, "
                "tol %.0f%%)\n",
                t_mean, t_target, 100.0 * t_err, 100.0 * t_tol);
    std::printf(
        "  <T> with naive Ndof  = %.2f K   <- the bug this pins: %.0f%% cold\n",
        t_naive, 100.0 * (1.0 - t_naive / t_mean));
    std::printf(
        "  worst |dr|/r         = %.2e   want <= %.0e   [rigid geometry]\n",
        worst_geom, geom_tol);

    bool ok = true;
    if (!(worst_geom <= geom_tol)) {
        std::printf("FAIL: the constrained geometry drifted\n");
        ok = false;
    }
    if (!(t_err <= t_tol)) {
        std::printf(
            "FAIL: equipartition violated — <T> is %.2f%% off the setpoint. "
            "The DOF count is wrong.\n",
            100.0 * t_err);
        ok = false;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
