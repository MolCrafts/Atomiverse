
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include <cmath>
#include <cstdio>
using namespace ATV;

int main() {
    cpu::Frame f(4);
    double zz[4] = {2, 18, 10, 10.5};
    int ty[4] = {1, 2, 3, 3};
    double qq[4] = {0, 0, 0.5, -0.5};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = 0;
        f.span(kAtom_y)[i] = 0;
        f.span(kAtom_z)[i] = zz[i];
        f.span(kAtom_type)[i] = ty[i];
        f.span(kAtom_charge)[i] = qq[i];
        f.span(kAtom_mass)[i] = 1;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(8, 8, 20);

    cpu::md::ConstPotBO bo(2.0);
    bo.add_electrode(1, 0);
    bo.add_electrode(2, 1);
    bo.xy_image_range = 30;

    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
    sim.set_evaluator(cpu::PairCoulCut{12.0});
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_observer_ref(bo);
    sim.build();
    sim.run(f, 1);

    std::printf("const-pot-tip4p-edl Qbot=%.6e Qtop=%.6e neut=%.3e\n",
                bo.total_charge(0), bo.total_charge(1),
                bo.neutrality_residual());
    return bo.neutrality_residual() < 1e-8 ? 0 : 1;
}
