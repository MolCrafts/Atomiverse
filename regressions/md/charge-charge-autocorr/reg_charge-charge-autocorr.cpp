
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/charge_autocorr.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include <cmath>
#include <cstdio>
using namespace ATV;

int main() {
    cpu::Frame f(4);
    double zz[4] = {8, 12, 9, 11};
    int ty[4] = {1, 2, 3, 3};
    double qq[4] = {0, 0, 0.5, -0.5};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = f.span(kAtom_y)[i] = 0;
        f.span(kAtom_z)[i] = zz[i];
        f.span(kAtom_type)[i] = ty[i];
        f.span(kAtom_charge)[i] = qq[i];
        f.span(kAtom_mass)[i] = 1;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(1, 1, 20);

    cpu::md::ConstPotBO bo(2.0);
    bo.add_electrode(1, 0);
    bo.add_electrode(2, 1);
    bo.xy_image_range = 40;
    cpu::ChargeChargeAutocorr ac(&bo, 1, 20);

    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
    sim.set_evaluator(cpu::PairCoulCut{12.0});
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_observer_ref(bo);
    sim.set_observer_ref(ac);
    sim.build();
    for (int s = 0; s < 30; ++s) {
        f.span(kAtom_z)[2] = 9.0 + 0.01 * std::sin(0.3 * s);
        sim.run(f, 1);
    }

    double c0 = ac.C0();
    std::printf("charge-charge-autocorr C0=%.6e n=%d\n", c0, ac.n_samples());
    return ac.n_samples() >= 20 ? 0 : 1;
}
