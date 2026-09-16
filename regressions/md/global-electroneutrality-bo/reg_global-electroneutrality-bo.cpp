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

// Real MetalWalls c8ad175d pin (mpirun -n 1 mw, 2026-07-29):
// tools/metalwalls_oracle/inputs/metalwalls/global-electroneutrality-bo/
// total_charges.out step 0 — sum_e Q_e = -q_ion exactly under charge_neutrality global.
static constexpr double kMwQ0 = -5.189485789245821e-1;  // E1, Psi = 0 V [e]
static constexpr double kMwQ1 = -4.810514210754179e-1;  // E2, Psi = 1 V [e]
static constexpr double kMwSum = -1.0;                  // = -q_ion

int main() {
    cpu::Frame f(3);
    // type1 @ z=5 (E1), type2 @ z=15 (E2), type3 ion @ z=10 with q=+1
    double zz[3] = {5, 15, 10};
    int ty[3] = {1, 2, 3};
    double qq[3] = {0, 0, 1};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_x)[i] = f.span(kAtom_y)[i] = 0;
        f.span(kAtom_z)[i] = zz[i];
        f.span(kAtom_type)[i] = ty[i];
        f.span(kAtom_charge)[i] = qq[i];
        f.span(kAtom_mass)[i] = 1;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(10, 10, 20);

    cpu::md::ConstPotBO bo(2.0);
    bo.add_electrode(1, 0.0);
    bo.add_electrode(2, 1.0);
    bo.electroneutrality = true;
    bo.global_electroneutrality = true;
    // Large image sum so the truncated elastance approaches MW's 2D-Ewald twin
    // closely enough to pin per-electrode Q (see REFERENCE.md).
    bo.xy_image_range = 1280;

    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
    sim.set_evaluator(cpu::PairCoulCut{12.0});
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_observer_ref(bo);
    sim.build();
    sim.run(f, 1);

    const double q0 = bo.total_charge(0);
    const double q1 = bo.total_charge(1);
    const double sumQ = q0 + q1;
    const double tgt = bo.neutrality_target();

    std::printf("global-electroneutrality-bo Q0=%.15e Q1=%.15e sumQ=%.15e target=%.15e\n",
                q0, q1, sumQ, tgt);

    int fail = 0;
    // Primary pin: global electroneutrality — MW prints sum = -1 e exactly.
    if (std::abs(tgt - kMwSum) > 1e-12) {
        std::fprintf(stderr, "neutrality_target not -q_ion (got %.15e)\n", tgt);
        ++fail;
    }
    if (std::abs(sumQ - kMwSum) > 1e-8) {
        std::fprintf(stderr, "sumQ != MW pin -1 e (got %.15e)\n", sumQ);
        ++fail;
    }
    // Secondary pin: per-electrode partition against MW total_charges.out.
    // ATV truncated image sum vs MW 2D-Ewald: ~6e-6 residual at image_range=1280.
    constexpr double kPerElectrodeTol = 2e-5;
    if (std::abs(q0 - kMwQ0) > kPerElectrodeTol) {
        std::fprintf(stderr, "Q0 vs MW: got %.15e want %.15e (tol %g)\n", q0, kMwQ0,
                     kPerElectrodeTol);
        ++fail;
    }
    if (std::abs(q1 - kMwQ1) > kPerElectrodeTol) {
        std::fprintf(stderr, "Q1 vs MW: got %.15e want %.15e (tol %g)\n", q1, kMwQ1,
                     kPerElectrodeTol);
        ++fail;
    }
    return fail ? 1 : 0;
}
