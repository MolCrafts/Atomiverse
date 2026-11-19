/// @file tests/core/test_unit_system.cpp
/// @brief LAMMPS `Update::set_units` rows (the run-level `units` table).
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "atomiverse/core/unit_system.h"
#include "atomiverse/core/units.h"

using namespace ATV;

TEST(UnitSystem, ReducedUnitsAreCoherent) {
    constexpr auto lj = units::from_style(units::Style::lj);
    static_assert(lj.kinetic_energy(2.) == 2.);
    static_assert(lj.velocity_impulse(3.) == 3.);
    EXPECT_DOUBLE_EQ(lj.thermal_energy(1.), 1.);
    EXPECT_DOUBLE_EQ(lj.coulomb_constant(), 1.);
    EXPECT_DOUBLE_EQ(lj.pressure(1.), 1.);
}
TEST(UnitSystem, PhysicalValuesUseTheSingleRegistry) {
    const auto real = units::from_style("real");
    const auto metal = units::from_style("metal");
    const auto si = units::from_style("si");
    EXPECT_NEAR(real.kinetic_energy(1.), 2390.057361376673, 1e-9);
    EXPECT_NEAR(metal.kinetic_energy(1.), .00010364269656262175, 1e-17);
    EXPECT_NEAR(real.boltzmann_constant(), .0019872042586408316, 1e-15);
    EXPECT_NEAR(si.coulomb_constant(), 8.9875517923e9, 1e-5);
    EXPECT_NEAR(units::from_style("micro").electric_field(1.), 1000., 1e-10);
    EXPECT_NEAR(units::from_style("nano").electric_field(1.), 160.2176634,
                1e-10);
    for (int s = 0; s < units::kStyleCount; ++s) {
        const auto run = units::from_style(static_cast<units::Style>(s));
        EXPECT_NEAR(run.velocity_impulse(run.kinetic_energy(3.)), 3., 1e-12);
    }
    EXPECT_THROW(units::from_style("unknown"), std::invalid_argument);
}
TEST(UnitSystem, StyleDefaultsRemainExplicit) {
    EXPECT_DOUBLE_EQ(units::from_style("real").dt, 1.);
    EXPECT_DOUBLE_EQ(units::from_style("metal").dt, .001);
    EXPECT_DOUBLE_EQ(units::from_style("nano").dt, .00045);
    EXPECT_DOUBLE_EQ(units::from_style("real").skin, 2.);
}

#include "atomiverse/core/unit_system.h"

TEST(UnitQuantity, AlgebraAndDimensionChecks) {
    constexpr auto u = units::UnitContext::from_style("real");
    constexpr auto charge = u.quantity(2., u.charge());
    constexpr auto field = u.quantity(.03, u.electric_field());
    constexpr auto f = charge * field;
    static_assert(f.unit.dimension == u.force().dimension);
    EXPECT_NEAR(f.numerical_value_in(u.force()), 1.3836328698371416, 1e-14);
    EXPECT_THROW(f.numerical_value_in(u.length()), std::invalid_argument);
    constexpr auto metal = units::UnitContext::from_style("metal");
    EXPECT_NEAR(f.numerical_value_in(metal.force()), .06, 1e-14);
    const units::Unit reduced_force{u.force().dimension, 1., 1};
    EXPECT_THROW(f.numerical_value_in(reduced_force), std::invalid_argument);
    EXPECT_NEAR((f * u.quantity(3., u.length())).numerical_value_in(u.energy()),
                4.150898609511425, 1e-13);
}

TEST(UnitQuantity, ResolvedFieldConversionHasNoWrapperStorageOrRuntimeLookup) {
    using ElectricFieldConversion = units::Conversion;
    static_assert(sizeof(ElectricFieldConversion) == sizeof(double));
    static_assert(std::is_trivially_copyable_v<ElectricFieldConversion>);
    static_assert(std::is_standard_layout_v<ElectricFieldConversion>);
    constexpr auto metal =
        units::from_style(units::Style::metal).electric_field;
    constexpr auto real = units::from_style(units::Style::real).electric_field;
    constexpr auto electron =
        units::from_style(units::Style::electron).electric_field;
    static_assert(metal(.03) == .03);
    EXPECT_NEAR(real(.03), .6918164349185708, 1e-14);
    EXPECT_NEAR(real(.06), 1.3836328698371416, 1e-14);
    EXPECT_NEAR(electron(1.), 1.9446903812113616e-10, 1e-23);
}

#include "atomiverse/core/field_units.h"
TEST(UnitQuantity, EveryCanonicalAtomColumnHasAnExplicitInterpretation) {
    const auto u = units::UnitContext::from_style("real");
    for (unsigned i = 0; i < static_cast<unsigned>(AtomFieldId::COUNT); ++i)
        EXPECT_NO_THROW(units::field_unit(static_cast<AtomFieldId>(i), u));
    EXPECT_EQ(units::field_unit(AtomFieldId::fx, u).dimension,
              u.force().dimension);
    EXPECT_EQ(units::field_unit(AtomFieldId::image, u).dimension,
              units::Dimension{});
    EXPECT_THROW(units::field_unit(AtomFieldId::COUNT, u),
                 std::invalid_argument);
}

TEST(UnitQuantity, StaticUnitsConversionChecksTypesBeforeRuntime) {
    constexpr auto real = units::UnitContext::from_style(units::Style::real);
    constexpr auto metal = units::UnitContext::from_style(units::Style::metal);
    constexpr auto source = real.energy();
    constexpr auto target = metal.energy();
    static_assert(noexcept(units::convert<source, target>(1.)));
    static_assert(units::convert<target, target>(.03) == .03);
    EXPECT_NEAR((units::convert<source, target>(23.060547830619)), 1., 1e-12);
}
