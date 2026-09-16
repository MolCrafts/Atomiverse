/// @file reg_no_backend.cpp
/// @brief backend-mirror-05-dissolve layout regression — a preset runs end to
///        end WITHOUT the caller ever naming a monolithic engine class. The
///        loop runtime is composed from communication / device / asynchronous
///        sub-runtimes behind the preset; the consumer only touches the public
///        preset API (cpu::md::MD) and Frame.
///
/// Behavioural guarantee, not a physics golden: a few LJ NVE steps run and the
/// total energy stays finite — proof the dissolved runtime dispatches a Plan.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main() {
    using namespace ATV;

    // 8 argon atoms on a simple-cubic sub-lattice in a large box. This is a
    // finiteness smoke, not a physics golden — but the setup must be physical:
    // mass MUST be set (velocity init draws v ~ sqrt(kT/m), so m=0 gives NaN),
    // and cutoff <= box/2 keeps the minimum-image convention unambiguous.
    const int n = 8;
    const double spacing = 4.0;  // [Å], just past the LJ well (2^(1/6)·σ)
    const double box_len = 20.0; // [Å], > 2·cutoff
    cpu::Frame frame(n);
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto mass = frame.span(kAtom_mass);
    auto type = frame.span(kAtom_type);
    int i = 0;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz, ++i) {
                x[i] = ix * spacing;
                y[i] = iy * spacing;
                z[i] = iz * spacing;
                mass[i] = 39.948; // argon [amu]
                type[i] = 0;      // LJ is type-agnostic here
            }
    frame.box = cpu::Box::cubic(box_len);

    // A preset composes the dissolved runtime internally — the caller never
    // names any engine/backend type.
    const double eps = 0.0104, sigma = 3.4, cutoff = 8.0, skin = 0.3, dt = 2.0;
    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_observer(cpu::md::InitVelocity{/*T[K]=*/60.0, /*seed=*/12345u});
    sim.build();
    sim.run(frame, 20);

    const double e = frame.meta.get(kEnergy) + frame.meta.get(kKinetic); // [eV]
    if (!std::isfinite(e)) {
        std::fprintf(stderr, "no_backend: total energy is not finite (%g)\n",
                     e);
        return EXIT_FAILURE;
    }
    std::puts("no_backend: MD preset ran 20 NVE steps on the composed runtime");
    return EXIT_SUCCESS;
}
