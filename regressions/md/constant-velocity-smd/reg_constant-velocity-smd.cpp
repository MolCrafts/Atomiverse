
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include <cmath>
#include <cstdio>
using namespace ATV;

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/fix_move.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
int main() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 5;
    f.span(kAtom_x)[1] = 0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 15;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(30);
    const double vz = 0.01;
    const double dt = 0.001;
    // `add_ref` borrows: the lender must outlive the driver, whose destructor
    // ends the live continuable run by calling `destroy()` on every component.
    cpu::md::FixMove move(0, 0, vz, 1, dt);
    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{0.01, 3.4, 6.0});
    sim.set_neighbor(cpu::md::NeighborList<>{6.0, 0.3});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.on(ATV::md::Phase::initial_integrate).add_ref(move, +200);
    sim.on(ATV::md::Phase::post_force).add_ref(move, +150);
    sim.build();
    sim.run(f, 100);
    // Rows are not stable under `atom_modify sort`; the pulled atom is tag 1.
    int row = -1;
    for (int i = 0; i < f.nlocal(); ++i)
        if (f.span(kAtom_tag)[i] == 1)
            row = i;
    if (row < 0) {
        std::printf("FAIL: tag 1 vanished from the frame\n");
        return 1;
    }
    double z = f.span(kAtom_z)[row];
    // MD run(100) launches steps 0..99 → t = 99 * dt at last impose.
    double z_pin = 5.0 + vz * dt * 99;
    std::printf("constant-velocity-smd z=%.9f pin=%.9f\n", z, z_pin);
    return std::abs(z - z_pin) < 1e-6 ? 0 : 1;
}
