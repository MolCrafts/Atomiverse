/// @file tests/cpu/semiempirical/test_rm1.cpp
/// @brief Unit gates for RM1 parameter tables and evaluator.

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_parameters.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_evaluator.h"
#include "atomiverse/cpu/semiempirical/rm1_parameters.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include "atomiverse/facade/stage.h"

using namespace ATV;
using namespace ATV::cpu::semiempirical;

namespace {

cpu::Frame make_water() {
    // Gas-phase water geometry near experiment (Å).
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_mass)[0] = 15.999;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_mass)[2] = 1.008;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.9572;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_x)[2] = -0.239987;
    f.span(kAtom_y)[2] = 0.926627;
    f.span(kAtom_z)[2] = 0.0;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

TEST(RM1, ParameterLookup) {
    EXPECT_TRUE(rm1_has_parameter(1));
    EXPECT_TRUE(rm1_has_parameter(6));
    EXPECT_TRUE(rm1_has_parameter(8));
    EXPECT_FALSE(rm1_has_parameter(2)); // He not in v1 table
    const auto* h = rm1_parameter(1);
    ASSERT_NE(h, nullptr);
    EXPECT_NEAR(h->uss, -11.960677, 1e-6);
    const auto* c = rm1_parameter(6);
    ASSERT_NE(c, nullptr);
    EXPECT_NE(c->uss, am1_parameter(6)->uss);
}

TEST(RM1, UnsupportedElementThrows) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 2; // He
    f.span(kAtom_mass)[0] = 4.0;
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0.0;
    f.simbox = cpu::Box::cubic(10.0);
    RM1Solver solver;
    EXPECT_THROW(solver.build(f), std::runtime_error);
}

TEST(RM1, WaterEnergyFiniteAndDiffersFromAM1) {
    auto frame = make_water();
    AM1Solver am1;
    am1.build(frame);
    const auto& ra = am1.run(frame);
    EXPECT_TRUE(ra.converged);
    EXPECT_TRUE(std::isfinite(ra.total_energy_ev));

    RM1Solver rm1;
    rm1.build(frame);
    const auto& rr = rm1.run(frame);
    EXPECT_TRUE(rr.converged);
    EXPECT_TRUE(std::isfinite(rr.total_energy_ev));
    EXPECT_GT(std::abs(rr.total_energy_ev - ra.total_energy_ev), 0.05)
        << "RM1 and AM1 water energies should differ";
}

TEST(RM1, EvaluatorWritesEnergyAndForces) {
    auto frame = make_water();
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_fx)[i] = 99.0;
        frame.span(kAtom_fy)[i] = 99.0;
        frame.span(kAtom_fz)[i] = 99.0;
    }
    RM1Evaluator eval{};
    eval.with_gradient(true);
    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);
    EXPECT_TRUE(std::isfinite(frame.meta.get(kEnergy)));
    EXPECT_NE(frame.span(kAtom_fx)[0], AccReal(99.0));
}
