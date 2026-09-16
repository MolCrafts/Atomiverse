#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_dpd.h"
#include <cmath>
#include <cstdio>
using namespace ATV;
int main() {
    constexpr int N = 16;
    cpu::Frame f(N);
    for (int i = 0; i < N; ++i) {
        f.span(kAtom_x)[i] = 0.5 * (i % 4);
        f.span(kAtom_y)[i] = 0.5 * ((i / 4) % 2);
        f.span(kAtom_z)[i] = 0.5 * (i / 8);
        f.span(kAtom_type)[i] = i % 2; // A/B
        f.span(kAtom_mass)[i] = 1;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::cubic(4.0);
    // a_AA=a_BB=25, a_AB=40
    std::vector<std::vector<double>> a{{25, 40}, {40, 25}};
    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairDpd{a, 4.5, 300.0, 1.0, 0.01, 3});
    sim.set_integrator(cpu::md::VelocityVerlet{0.01});
    sim.enable_timing(false);
    sim.build();
    sim.run(f, 30);
    const double E = f.meta.get(kEnergy);
    std::printf("dpd_ab E=%.6e\n", E);
    if (!std::isfinite(E)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
