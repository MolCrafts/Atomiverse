/// @file tests/cpu/planewave/test_pw_bomd.cpp
/// @brief Unit gate: MD binds PWEvaluator forces for a short BOMD step.
///
/// One method surface only — public MD + VelocityVerlet + PWEvaluator.
/// Full NVE energy-drift pin lives in regressions/pw/pw-bomd-h (e2e).

#include <gtest/gtest.h>

#include <cmath>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

cpu::Frame make_h_cubic() {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    // Hand-set v (InitVelocity COM-zeros N=1).
    f.span(kAtom_vx)[0] = 0.002;
    f.span(kAtom_vy)[0] = 0.0;
    f.span(kAtom_vz)[0] = 0.0;
    f.box = cpu::Box::cubic(6.0);
    return f;
}

PW cheap_cfg() {
    PW cfg;
    cfg.gth_name = "GTH-PBE-q1";
    cfg.ecut_ry = 5.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 8;
    cfg.mix_beta = 0.5;
    return cfg;
}

} // namespace

TEST(PWBomd, MdBindsEvaluatorForcesTwoSteps) {
    auto frame = make_h_cubic();

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.5});
    sim.on(md::Phase::force).add(PWEvaluator{cheap_cfg()}.with_gradient(true));
    sim.enable_timing(false);
    sim.build();
    ASSERT_EQ(sim.run(frame, 2), 2);

    const double pe = frame.meta.get(kEnergy);
    const double ke = frame.meta.get(kKinetic);
    EXPECT_TRUE(std::isfinite(pe)) << "PE=" << pe;
    EXPECT_TRUE(std::isfinite(ke)) << "KE=" << ke;
    EXPECT_GT(ke, 1e-12);

    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.span(kAtom_x)[0])));
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.span(kAtom_y)[0])));
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.span(kAtom_z)[0])));
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.span(kAtom_fx)[0])));
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.span(kAtom_fy)[0])));
    EXPECT_TRUE(std::isfinite(static_cast<double>(frame.span(kAtom_fz)[0])));

    const double fmag = std::sqrt(static_cast<double>(
        frame.span(kAtom_fx)[0] * frame.span(kAtom_fx)[0]
        + frame.span(kAtom_fy)[0] * frame.span(kAtom_fy)[0]
        + frame.span(kAtom_fz)[0] * frame.span(kAtom_fz)[0]));
    // High-symmetry site residual (toy PP + short SCF).
    EXPECT_LT(fmag, 0.5) << " |F|=" << fmag;
}
