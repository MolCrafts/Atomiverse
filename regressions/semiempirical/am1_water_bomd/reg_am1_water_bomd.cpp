/// @file regressions/semiempirical/am1_water_bomd/reg_am1_water_bomd.cpp
/// @brief Short AM1 BOMD smoke: MD + VelocityVerlet + AM1 forces.
///
/// Method: AM1 (Dewar et al., J. Am. Chem. Soc. 107, 3902 (1985)).
/// Pins that semiempirical forces bind cleanly to `cpu::md::MD` for ≥20 steps
/// with finite PE/KE. Not a thermostatted production run.
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#include "atomiverse/facade/stage.h"

namespace {

ATV::cpu::Frame make_water() {
    // Near-experimental geometry (Å) so residual forces are moderate.
    ATV::cpu::Frame f(3);
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_x)[1] = 0.9572;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_x)[2] = -0.239987;
    f.span(ATV::kAtom_y)[2] = 0.926627;
    f.span(ATV::kAtom_z)[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        f.span(ATV::kAtom_tag)[i] = i + 1;
        f.span(ATV::kAtom_mass)[i] =
            (f.span(ATV::kAtom_type)[i] == 8) ? 15.999 : 1.008;
        f.span(ATV::kAtom_vx)[i] = 0.0;
        f.span(ATV::kAtom_vy)[i] = 0.0;
        f.span(ATV::kAtom_vz)[i] = 0.0;
    }
    f.simbox = ATV::cpu::Box::cubic(20.0);
    return f;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = make_water();

    constexpr double dt = 0.25; // fs — conservative for unconstrained O–H
    constexpr int n_steps = 20;
    constexpr double T0 = 10.0; // K — gentle KE for SCF-stable BOMD smoke

    cpu::semiempirical::AM1Config cfg;
    cfg.max_iter = 300;
    cfg.density_tol = 1.0e-7;
    cfg.energy_tol = 1.0e-9;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.add(cpu::md::InitVelocity{T0, /*seed=*/7u});
    sim.on(md::Phase::force)
        .add(cpu::semiempirical::AM1Evaluator{cfg}.with_gradient(true));
    sim.build();
    sim.run(frame, n_steps);

    const double pe = frame.meta.get(kEnergy);
    const double ke = frame.meta.get(kKinetic);
    std::printf("AM1 BOMD: steps=%d  PE=%.6f eV  KE=%.6f eV  nlocal=%d\n",
                n_steps, pe, ke, frame.nlocal());

    if (!std::isfinite(pe) || !std::isfinite(ke)) {
        std::fprintf(stderr, "FAIL: non-finite PE or KE\n");
        return 1;
    }
    if (!(pe < 0.0)) {
        std::fprintf(stderr, "FAIL: PE not bound (PE=%g eV)\n", pe);
        return 1;
    }
    if (!(ke >= 0.0)) {
        std::fprintf(stderr, "FAIL: KE negative (KE=%g eV)\n", ke);
        return 1;
    }
    // Mild sanity: no explosion after ~5 fs of dynamics.
    if (!(std::abs(pe) < 1.0e4) || !(ke < 10.0)) {
        std::fprintf(stderr, "FAIL: energy magnitude out of range\n");
        return 1;
    }

    std::printf("PASS: AM1 BOMD smoke (%d steps, finite PE/KE)\n", n_steps);
    return 0;
}
