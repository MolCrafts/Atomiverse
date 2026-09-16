/// @file regressions/md/brownian_einstein/reg_brownian_einstein.cpp
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/brownian.h"
#include "atomiverse/cpu/dynamics/fix_addforce.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

int main() {
    const double D = 0.02; // Å²/fs
    const double dt = 0.5;
    const int n_steps = 2000;
    const int n_part = 50;

    cpu::Frame f(n_part);
    for (int i = 0; i < n_part; ++i) {
        f.span(kAtom_x)[i] = 0;
        f.span(kAtom_y)[i] = 0;
        f.span(kAtom_z)[i] = 0;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::cubic(1e6); // free sandbox — no wrap

    // Free-particle Brownian: F=0 Bias satisfies MD force-source validate.
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::Brownian{dt, D, 300.0, 7});
    sim.set_evaluator(cpu::md::FixAddForce{0.0, 0.0, 0.0, /*type=*/0});
    sim.enable_timing(false);
    sim.build();
    sim.run(f, n_steps);

    double msd = 0;
    for (int i = 0; i < n_part; ++i) {
        const double x = f.span(kAtom_x)[i];
        const double y = f.span(kAtom_y)[i];
        const double z = f.span(kAtom_z)[i];
        msd += x * x + y * y + z * z;
    }
    msd /= n_part;
    const double t = n_steps * dt;
    const double expect = 6.0 * D * t;
    const double ratio = msd / expect;
    std::printf("Brownian Einstein: MSD=%.6f  6Dt=%.6f  ratio=%.4f\n", msd,
                expect, ratio);
    if (!(ratio > 0.7 && ratio < 1.3)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
