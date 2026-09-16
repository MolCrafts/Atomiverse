
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/elastance_observer.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include <cmath>
#include <cstdio>
using namespace ATV;

int main() {
    cpu::Frame f(2);
    f.span(kAtom_z)[0] = 5;
    f.span(kAtom_z)[1] = 15;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = f.span(kAtom_y)[i] = 0;
        f.span(kAtom_type)[i] = i + 1;
        f.span(kAtom_charge)[i] = 0;
        f.span(kAtom_mass)[i] = 1;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(5, 5, 20);

    cpu::md::ConstPotBO bo(2.0);
    bo.add_electrode(1, 0);
    bo.add_electrode(2, 1);
    bo.xy_image_range = 80;
    cpu::ElastanceObserver obs(&bo);

    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
    sim.set_evaluator(cpu::PairCoulCut{12.0});
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_observer_ref(bo);
    sim.set_observer_ref(obs);
    sim.build();
    sim.run(f, 1);

    std::printf("elastance-condition-observer cond=%.6e C=%.6e\n",
                obs.last_condition(), obs.last_capacitance());
    return (obs.last_condition() >= 1.0 && std::isfinite(obs.last_capacitance())
            && obs.last_capacitance() > 0.0)
               ? 0
               : 1;
}
