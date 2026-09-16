/// @file regressions/md/fene_wca_chain/reg_fene_wca_chain.cpp
/// Kremer–Grest short chain: bonds stay < R0 after short MD.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/bond_fene.h"
#include "atomiverse/cpu/potentials/pair_wca.h"

using namespace ATV;

int main() {
    constexpr int N = 8;
    const double K = 30.0;  // eV/Å² (order-of-magnitude textbook)
    const double R0 = 1.5;  // Å
    const double eps = 1.0; // eV
    const double sigma = 1.0;
    const double bond0 = 0.96; // < 2^{1/6}σ and < R0

    cpu::Frame f(N);
    for (int i = 0; i < N; ++i) {
        f.span(kAtom_x)[i] = i * bond0;
        f.span(kAtom_y)[i] = 0;
        f.span(kAtom_z)[i] = 0;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::cubic(100.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    for (int i = 0; i + 1 < N; ++i)
        f.add_bond(i, i + 1, 0);

    cpu::md::MD sim;
    sim.set_evaluator(cpu::BondFene{{K}, {R0}});
    sim.set_evaluator(cpu::PairWCA{eps, sigma});
    sim.set_integrator(cpu::md::VelocityVerlet{0.001});
    sim.build();
    sim.run(f, 200);

    double rmax = 0;
    for (int i = 0; i + 1 < N; ++i) {
        double dx = f.span(kAtom_x)[i + 1] - f.span(kAtom_x)[i];
        double dy = f.span(kAtom_y)[i + 1] - f.span(kAtom_y)[i];
        double dz = f.span(kAtom_z)[i + 1] - f.span(kAtom_z)[i];
        f.box.min_image(dx, dy, dz);
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (r > rmax)
            rmax = r;
        if (!(r < R0) || !std::isfinite(r)) {
            std::printf("FAIL bond %d r=%.6f R0=%.3f\n", i, r, R0);
            return 1;
        }
    }
    const double E = f.meta.get(kEnergy);
    std::printf("FENE+WCA chain: rmax=%.6f R0=%.3f E=%.6e\n", rmax, R0, E);
    if (!std::isfinite(E)) {
        std::printf("FAIL energy NaN\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
