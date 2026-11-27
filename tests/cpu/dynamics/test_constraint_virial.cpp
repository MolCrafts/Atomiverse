#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/cpu/dynamics/fix_lincs.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/dynamics/fix_shake.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using namespace ATV;
namespace {
cpu::Frame rotor(double velocity_scale = 1.0) {
    cpu::Frame f(2);
    f.box = cpu::Box::cubic(10);
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_mass)[i] = i ? 3 : 1;
    const double r[3] = {1.0 / 3, 2.0 / 3, 2.0 / 3};
    const double v[3] = {0.2 / std::sqrt(5.0), -0.1 / std::sqrt(5.0), 0};
    const ColKey<StorageReal> pos[3] = {kAtom_x, kAtom_y, kAtom_z};
    const ColKey<StorageReal> vel[3] = {kAtom_vx, kAtom_vy, kAtom_vz};
    for (int d = 0; d < 3; ++d) {
        f.span(pos[d])[0] = 5 + 0.75 * r[d];
        f.span(pos[d])[1] = 5 - 0.25 * r[d];
        f.span(vel[d])[0] = 0.75 * v[d] * velocity_scale;
        f.span(vel[d])[1] = -0.25 * v[d] * velocity_scale;
    }
    f.clear_tag_index();
    f.rebuild_tag_index(0, 2);
    f.add_bond(0, 1, 0);
    return f;
}
template <class Constraint>
void initial_pressure(const char* style) {
    cpu::Context ctx;
    ctx.units = units::from_style(style);
    auto f = rotor(1 / std::sqrt(ctx.units.kinetic_energy.scale()));
    const double dt = 0.01 * std::sqrt(ctx.units.kinetic_energy.scale());
    Constraint fix(dt, 1e-13);
    fix.constrain_bond(0, 1);
    fix.build({ctx, f});
    fix.initialize_forces({ctx, f, slot_of(md::Phase::post_force), 0});
    // Analytic rigid rotor reaction tensor. Its trace cancels the internal
    // kinetic pressure; the bond direction fixes all six components.
    double lambda = -0.75 * 0.01;
    if constexpr (std::is_same_v<Constraint, cpu::md::FixShake>) {
        // SHAKE's bootstrap force predicts one full drift with dt^2/2.
        const double v2 = 0.01 / ctx.units.kinetic_energy.scale();
        lambda = -2 * 0.75 * v2 / (1 + std::sqrt(1 - dt * dt * v2))
                 / ctx.units.velocity_impulse.scale();
    }
    const double r[3] = {1.0 / 3, 2.0 / 3, 2.0 / 3};
    const double expected[6] = {r[0] * r[0], r[1] * r[1], r[2] * r[2],
                                r[0] * r[1], r[0] * r[2], r[1] * r[2]};
    for (int d = 0; d < 6; ++d)
        EXPECT_NEAR(f.meta.get(kStress)[d], lambda * expected[d], 1e-11)
            << style << " component " << d;
    const ColKey<AccReal> force[3] = {kAtom_fx, kAtom_fy, kAtom_fz};
    for (int d = 0; d < 3; ++d) {
        const double reaction =
            std::is_same_v<Constraint, cpu::md::FixShake> ? lambda * r[d] : 0.0;
        EXPECT_NEAR(f.span(force[d])[0], reaction, 1e-11);
        EXPECT_NEAR(f.span(force[d])[1], -reaction, 1e-11);
    }
}
template <class Constraint>
void rotor_step() {
    auto f = rotor();
    cpu::md::MD sim;
    sim.set_units("lj");
    sim.set_evaluator(cpu::BondHarmonic{{0.0}, {1.0}});
    sim.set_integrator(cpu::md::VelocityVerlet{0.002});
    Constraint fix(0.002, 1e-13);
    fix.constrain_bond(0, 1);
    sim.set_observer(std::move(fix));
    sim.build();
    sim.run(f, 1);
    double dx = f.span(kAtom_x)[0] - f.span(kAtom_x)[1];
    double dy = f.span(kAtom_y)[0] - f.span(kAtom_y)[1];
    double dz = f.span(kAtom_z)[0] - f.span(kAtom_z)[1];
    EXPECT_NEAR(dx * dx + dy * dy + dz * dz, 1, 2e-10);
    const auto& w = f.meta.get(kStress);
    EXPECT_NEAR(w[0] + w[1] + w[2] + 2 * f.meta.get(kKinetic), 0, 3e-7);
}
} // namespace
TEST(ConstraintVirial, RattleRunZeroHasReactionInEveryUnitStyle) {
    for (const char* u : {"lj", "real", "metal"})
        initial_pressure<cpu::md::FixRattle>(u);
}
TEST(ConstraintVirial, LincsRunZeroHasReactionInEveryUnitStyle) {
    for (const char* u : {"lj", "real", "metal"})
        initial_pressure<cpu::md::FixLincs>(u);
}
TEST(ConstraintVirial, LincsRigidRotorHasMolecularPressure) {
    rotor_step<cpu::md::FixLincs>();
}
TEST(ConstraintVirial, ShakeRunZeroHasReactionInEveryUnitStyle) {
    for (const char* u : {"lj", "real", "metal"})
        initial_pressure<cpu::md::FixShake>(u);
}
TEST(ConstraintVirial, RattleRigidRotorHasMolecularPressure) {
    rotor_step<cpu::md::FixRattle>();
}
TEST(ConstraintVirial, ShakeRigidRotorHasMolecularPressure) {
    rotor_step<cpu::md::FixShake>();
}
TEST(ConstraintVirial, AlgorithmsHaveDistinctVelocityBindings) {
    cpu::md::FixShake shake(1);
    cpu::md::FixRattle rattle(1);
    EXPECT_EQ(shake.name(), "shake");
    EXPECT_EQ(rattle.name(), "rattle");
    EXPECT_EQ(shake.default_bindings().size(), 1u);
    EXPECT_EQ(rattle.default_bindings().size(), 4u);
    EXPECT_THROW((cpu::md::FixShake{std::numeric_limits<double>::infinity()}),
                 std::invalid_argument);
    EXPECT_THROW(
        shake.constrain_bond(0, std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
}

TEST(ConstraintVirial, RattleInitialReactionCancelsPhysicalBondForce) {
    auto f = rotor(0.0);
    const ColKey<AccReal> force[3] = {kAtom_fx, kAtom_fy, kAtom_fz};
    const double r[3] = {1.0 / 3, 2.0 / 3, 2.0 / 3};
    for (int d = 0; d < 3; ++d) {
        f.span(force[d])[0] = 1.3 * r[d];
        f.span(force[d])[1] = -1.3 * r[d];
    }
    cpu::Context ctx;
    ctx.units = units::from_style("real");
    cpu::md::FixRattle fix(1.0);
    fix.constrain_bond(0, 1);
    fix.build({ctx, f});
    fix.initialize_forces({ctx, f, slot_of(md::Phase::post_force), 0});
    const auto& w = f.meta.get(kStress);
    EXPECT_NEAR(w[0] + w[1] + w[2], -1.3, 1e-11);
    for (int d = 0; d < 3; ++d) {
        EXPECT_DOUBLE_EQ(f.span(force[d])[0], 1.3 * r[d]);
        EXPECT_DOUBLE_EQ(f.span(force[d])[1], -1.3 * r[d]);
    }
}

// The SHAKE force stage sends one forward pass: each owner's predicted next
// position (LAMMPS xshake), image-shifted like a coordinate.
TEST(ConstraintVirial, ShakePacksThePredictedPositionWithTheImageShift) {
    cpu::Context ctx;
    ctx.units = units::from_style("lj");
    auto f = rotor();
    cpu::md::FixShake fix(0.01, 1e-13);
    fix.constrain_bond(0, 1);
    fix.build({ctx, f});
    fix.initialize_forces({ctx, f, slot_of(md::Phase::post_force), 0});
    fix.set_comm_mode(cpu::md::FixShake::CommMode::PREDICTED);
    const int list[2] = {0, 1};
    const int none[3] = {0, 0, 0}, image[3] = {1, 0, -2};
    double plain[6], shifted[6];
    ASSERT_EQ(fix.pack_forward_comm(2, list, plain, 0, none), 6);
    ASSERT_EQ(fix.pack_forward_comm(2, list, shifted, 1, image), 6);
    // The solved prediction holds the bond at its constrained length.
    double d2 = 0;
    for (int d = 0; d < 3; ++d)
        d2 += (plain[d] - plain[3 + d]) * (plain[d] - plain[3 + d]);
    EXPECT_NEAR(d2, 1.0, 1e-11);
    const double shift[3] = {10.0, 0.0, -20.0};
    for (int k = 0; k < 6; ++k)
        EXPECT_DOUBLE_EQ(shifted[k], plain[k] + shift[k % 3]);
    // Unpacking writes the prediction rows back, not the Frame columns.
    const double x0 = f.span(kAtom_x)[0];
    fix.unpack_forward_comm(2, 0, shifted);
    EXPECT_DOUBLE_EQ(f.span(kAtom_x)[0], x0);
    double again[6];
    ASSERT_EQ(fix.pack_forward_comm(2, list, again, 0, none), 6);
    for (int k = 0; k < 6; ++k)
        EXPECT_DOUBLE_EQ(again[k], shifted[k]);
}
