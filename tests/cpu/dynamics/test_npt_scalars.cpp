/// Single MTK initial propagation against its analytical zero-force limit.
/// Martyna et al., JCP 101, 4177 (1994), doi:10.1063/1.467468.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/driver/dof.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include <cmath>
#include <gtest/gtest.h>

TEST(NptScalars, MtkUsesAtomCountIndependentlyOfConstraintDof) {
    for (int constraints : {0, 1, 2}) {
        ATV::cpu::Frame frame(2);
        frame.box = ATV::cpu::Box::cubic(10);
        constexpr double kinetic = 2, dt = 0.001;
        const double speed = std::sqrt(kinetic / ATV::units::mass_v2_to_energy);
        frame.span(ATV::kAtom_mass)[0] = frame.span(ATV::kAtom_mass)[1] = 1;
        frame.span(ATV::kAtom_vy)[0] = speed;
        frame.span(ATV::kAtom_vy)[1] = -speed;
        frame.meta.get(ATV::kKinetic) = kinetic;
        ATV::cpu::Context context;
        ATV::cpu::publish_global_dof(context, frame, constraints);
        const double dof = context.md.n_dof_global;
        const double kT = 2 * kinetic / dof;
        ATV::cpu::md::NoseHooverNPT npt(kT / ATV::units::kB_ev_per_k, 1, 0, 1,
                                        dt, 0, 1, 1);
        npt.build({context, frame});
        npt.initialize_forces(
            {context, frame, ATV::slot_of(ATV::md::Phase::end_of_step), 0});
        const double before = frame.box.volume();
        npt.launch({context, frame,
                    ATV::slot_of(ATV::md::Phase::initial_integrate), 0});
        // At P=0 with initial chain momenta zero and T measured exactly,
        // eps_dot = KE*(1+1/N)*dt/W and ln(V'/V) = 3*eps_dot*dt.
        const double W = (dof + 3) * kT;
        const double expected = 3 * kinetic * (1 + 1.0 / 2) * dt * dt / W;
        EXPECT_NEAR(std::log(frame.box.volume() / before), expected, 1e-13)
            << "constraints=" << constraints;
        npt.destroy({context, frame});
    }
}
