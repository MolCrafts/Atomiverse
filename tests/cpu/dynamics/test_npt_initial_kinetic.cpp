/// @file tests/cpu/dynamics/test_npt_initial_kinetic.cpp
/// @brief First NoseHooverNPT::launch reads the current velocity state.
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"

#include <cmath>
#include <limits>

using namespace ATV;

TEST(NptInitialKinetic,
     FirstThermostatHalfStepUsesActualKineticAtAnyStepIndex) {
    constexpr double temperature = 100.0, tau = 10.0, dt = 0.1;
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(20.0);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_x)[i] = 4.0 + i;
        frame.span(kAtom_y)[i] = 4.0;
        frame.span(kAtom_z)[i] = 4.0;
        frame.span(kAtom_mass)[i] = 2.0;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
        frame.span(kAtom_fx)[i] = 0.0;
        frame.span(kAtom_fy)[i] = 0.0;
        frame.span(kAtom_fz)[i] = 0.0;
    }
    cpu::Context context;
    context.md.n_dof_global = 3;
    // One thermostat link and no active pressure axes isolate the first
    // NHC half-step: eta_dot = G*dt/4, v' = v*exp(-G*dt^2/8).
    cpu::md::NoseHooverNPT integrator{
        temperature, tau, 0.0, 100.0, dt, 0.0, 1, 1, {false, false, false}};
    integrator.build({context, frame});

    // A velocity initializer may run after NPT::build; a newly mounted NPT
    // component can begin at a nonzero cumulative trajectory step.
    frame.span(kAtom_vx)[0] = 0.05;
    frame.span(kAtom_vx)[1] = -0.05;
    const double velocity = frame.span(kAtom_vx)[0];
    const double kinetic = units::mass_v2_to_energy * 2.0 * velocity * velocity;
    const double kT = units::kB_ev_per_k * temperature;
    const double G = (2.0 * kinetic - 3.0 * kT) / (3.0 * kT * tau * tau);
    const double expected = velocity * std::exp(-G * dt * dt / 8.0);

    integrator.launch(
        {context, frame, slot_of(md::Phase::initial_integrate), 17});

    const double tolerance = 8 * std::numeric_limits<StorageReal>::epsilon();
    EXPECT_NEAR(frame.span(kAtom_vx)[0], expected, tolerance);
    EXPECT_NEAR(frame.span(kAtom_vx)[1], -expected, tolerance);
}
