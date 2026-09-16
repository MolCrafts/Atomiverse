/// @file regressions/md/freeze-electrode-atoms/reg_freeze-electrode-atoms.cpp
/// @brief E2E: wall atoms frozen under LJ NVE match LAMMPS fix setforce 0 pin.
///
/// See REFERENCE.md for oracle (LAMMPS), geometry, and tolerances.
/// OUT-OF-LIBRARY consumer: public headers + link atomiverse only.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/dynamics/fix_freeze.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // ── Geometry matches tools/metalwalls_oracle/.../in.lammps ─────────
    const double eps = 0.010341; // [eV]
    const double sigma = 3.4;    // [A]
    const double cutoff = 8.0;   // [A]
    const double skin = 0.3;     // [A]
    const double dt = 1.0;       // [fs]  == LAMMPS metal timestep 0.001 ps
    const int n_steps = 1000;
    const int wall_type = 2;
    const int fluid_type = 1;

    const double wall_xyz[4][3] = {
        {2.0, 2.0, 0.0},
        {2.0, 6.0, 0.0},
        {6.0, 2.0, 0.0},
        {6.0, 6.0, 0.0},
    };
    const double fluid_z = 3.5;

    cpu::Frame frame(8);
    // Pin expected wall positions by stable tag (tags survive atom sort).
    std::unordered_map<TagInt, std::array<double, 3>> wall_x0;
    std::unordered_map<TagInt, std::array<double, 3>> fluid_x0;

    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i] = wall_xyz[i][0];
        frame.span(kAtom_y)[i] = wall_xyz[i][1];
        frame.span(kAtom_z)[i] = wall_xyz[i][2];
        frame.span(kAtom_mass)[i] = 196.9665;
        frame.span(kAtom_type)[i] = wall_type;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
        wall_x0[static_cast<TagInt>(i + 1)] = {wall_xyz[i][0], wall_xyz[i][1],
                                               wall_xyz[i][2]};
    }
    for (int i = 0; i < 4; ++i) {
        const int j = 4 + i;
        frame.span(kAtom_x)[j] = wall_xyz[i][0];
        frame.span(kAtom_y)[j] = wall_xyz[i][1];
        frame.span(kAtom_z)[j] = fluid_z;
        frame.span(kAtom_mass)[j] = 39.948;
        frame.span(kAtom_type)[j] = fluid_type;
        frame.span(kAtom_tag)[j] = j + 1;
        fluid_x0[static_cast<TagInt>(j + 1)] = {wall_xyz[i][0], wall_xyz[i][1],
                                                fluid_z};
    }
    frame.box = cpu::Box::cubic(10.0);

    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    // Seed fluid (and briefly wall); FixFreeze.build re-zeros wall v.
    sim.set_observer(cpu::md::InitVelocity{/*T=*/200.0, /*seed=*/99u});
    sim.set_observer(cpu::md::FixFreeze{wall_type});
    sim.enable_timing(false);
    sim.build();

    const auto t0 = std::chrono::steady_clock::now();
    sim.run(frame, n_steps);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    // ── Pin checks by tag (atom sort may reorder local rows) ──────────
    const double pos_tol = 1.0e-10; // [A]
    const double vel_tol = 1.0e-12; // [A/fs]
    double max_dx = 0.0;
    double max_v = 0.0;
    int n_wall_seen = 0;
    double max_fluid_disp = 0.0;
    int n_fluid_seen = 0;

    const int n = frame.nlocal();
    for (int i = 0; i < n; ++i) {
        const int ty = static_cast<int>(frame.span(kAtom_type)[i]);
        const TagInt tag = frame.span(kAtom_tag)[i];
        if (ty == wall_type) {
            ++n_wall_seen;
            const auto it = wall_x0.find(tag);
            if (it == wall_x0.end()) {
                std::printf("FAIL: unexpected wall tag %lld\n",
                            static_cast<long long>(tag));
                return 1;
            }
            const double dx = std::fabs(frame.span(kAtom_x)[i] - it->second[0]);
            const double dy = std::fabs(frame.span(kAtom_y)[i] - it->second[1]);
            const double dz = std::fabs(frame.span(kAtom_z)[i] - it->second[2]);
            max_dx = std::fmax(max_dx, std::fmax(dx, std::fmax(dy, dz)));
            max_v = std::fmax(max_v, std::fabs(frame.span(kAtom_vx)[i]));
            max_v = std::fmax(max_v, std::fabs(frame.span(kAtom_vy)[i]));
            max_v = std::fmax(max_v, std::fabs(frame.span(kAtom_vz)[i]));
        } else if (ty == fluid_type) {
            ++n_fluid_seen;
            const auto it = fluid_x0.find(tag);
            if (it != fluid_x0.end()) {
                const double dx = frame.span(kAtom_x)[i] - it->second[0];
                const double dy = frame.span(kAtom_y)[i] - it->second[1];
                const double dz = frame.span(kAtom_z)[i] - it->second[2];
                max_fluid_disp = std::fmax(
                    max_fluid_disp, std::sqrt(dx * dx + dy * dy + dz * dz));
            }
        }
    }

    std::printf("freeze-electrode-atoms  N=8  steps=%d  dt=%.1f fs\n", n_steps,
                dt);
    std::printf("  wall atoms seen       = %d   want 4\n", n_wall_seen);
    std::printf("  fluid atoms seen      = %d   want 4\n", n_fluid_seen);
    std::printf("  max wall |dx| [A]     = %.3e   want <= %.1e\n", max_dx,
                pos_tol);
    std::printf("  max wall |v|  [A/fs]  = %.3e   want <= %.1e\n", max_v,
                vel_tol);
    std::printf("  max fluid disp [A]    = %.3e   want > 1e-3 (selective)\n",
                max_fluid_disp);
    std::printf("  wall_time_%d_steps    = %.6f s\n", n_steps, wall_s);

    int rc = 0;
    if (n_wall_seen != 4 || n_fluid_seen != 4) {
        std::printf("FAIL: atom type counts wrong after MD\n");
        rc = 1;
    }
    if (max_dx > pos_tol) {
        std::printf("FAIL: wall atom moved (LAMMPS setforce-0 pin broken)\n");
        rc = 1;
    }
    if (max_v > vel_tol) {
        std::printf("FAIL: wall velocity non-zero\n");
        rc = 1;
    }
    if (max_fluid_disp <= 1.0e-3) {
        std::printf("FAIL: fluid did not move — freeze may be global\n");
        rc = 1;
    }
    if (rc == 0)
        std::printf("PASS\n");
    return rc;
}
