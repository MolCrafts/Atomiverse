/// @file reg_lammps-comm-brick-03-verlet.cpp
/// @brief Isolated-atom MD: 10 Verlet steps, kEnergy == 0 eV, nlocal == 1.
///
/// Public headers + link atomiverse. No third-party. Literature: Plimpton
/// JCP 117 (1995); Thompson CPC 271 (2022); LAMMPS 78a7ddf src/verlet.cpp.

#include <cstdio>
#include <cstdlib>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    cpu::Frame frame(1);
    frame.span(kAtom_x)[0] = 1.0;
    frame.span(kAtom_y)[0] = 1.0;
    frame.span(kAtom_z)[0] = 1.0;
    frame.span(kAtom_mass)[0] = 1.0;
    frame.span(kAtom_type)[0] = 0;
    frame.box = cpu::Box::cubic(10.0);

    constexpr double eps = 0.0103; // [eV]
    constexpr double sigma = 3.4;  // [Å]
    constexpr double cutoff = 2.5; // [Å] < L/2
    constexpr double skin = 0.3;   // [Å]
    constexpr double dt = 1.0;     // [fs]

    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.build();
    sim.run(frame, 10);

    const int nlocal = frame.nlocal();
    const double energy = frame.meta.get(kEnergy);
    std::printf("nlocal %d\n", nlocal);
    std::printf("energy %.8e eV\n", energy);

    if (nlocal != 1) {
        std::fprintf(stderr, "FAIL: nlocal=%d want 1\n", nlocal);
        return EXIT_FAILURE;
    }
    if (energy != 0.0) {
        std::fprintf(stderr, "FAIL: kEnergy=%.8e eV want 0\n", energy);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
