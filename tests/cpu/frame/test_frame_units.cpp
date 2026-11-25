#include "atomiverse/core/physical_units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/units/frame.h"
#include <gtest/gtest.h>
using namespace ATV;

namespace {
cpu::Frame physical_frame() {
    cpu::Frame f(2);
    ATV::units::annotate(f, "real");
    f.box = cpu::Box::orthorhombic(10., 20., 30.);
    f.span(kAtom_x)[0] = 2.;
    f.span(kAtom_vx)[0] = .5;
    f.span(kAtom_fx)[0] = 2.;
    f.span(kAtom_mass)[0] = 12.;
    f.span(kAtom_charge)[0] = -1.;
    f.span(kAtom_image)(0, 0) = 2;
    f.meta.get(kEnergy) = 23.060547830619;
    f.meta.get(kStress)[0] = 23.060547830619;
    f.meta.get(kStressKinetic)[1] = 23.060547830619;
    return f;
}
} // namespace
TEST(FrameUnits, RealMetalScalesVelocityEnergyForceAndExtensiveStress) {
    auto original = physical_frame();
    auto metal = ATV::units::convert(original, "metal");
    EXPECT_EQ(ATV::units::frame_profile(metal), "metal");
    EXPECT_DOUBLE_EQ(metal.span(kAtom_x)[0], 2.);
    EXPECT_NEAR(metal.span(kAtom_vx)[0], 500., 1e-12);
    EXPECT_NEAR(metal.meta.get(kEnergy), 1., 1e-12);
    EXPECT_NEAR(metal.meta.get(kStress)[0], 1., 1e-12);
    EXPECT_NEAR(metal.meta.get(kStressKinetic)[1], 1., 1e-12);
    EXPECT_NEAR(metal.span(kAtom_fx)[0], 2. / 23.060547830619, 1e-14);
    EXPECT_EQ(metal.span(kAtom_image)(0, 0), 2);
    EXPECT_DOUBLE_EQ(original.span(kAtom_vx)[0], .5);
}
TEST(FrameUnits, EveryPhysicalStyleRoundTripsThroughSi) {
    const auto original = physical_frame();
    for (const auto* style : {"real", "metal", "si", "cgs", "electron", "micro",
                              "nano", "electronic"}) {
        auto converted = ATV::units::convert(original, style);
        auto restored =
            ATV::units::convert(ATV::units::convert(converted, "si"), "real");
        EXPECT_NEAR(restored.span(kAtom_x)[0], 2., 1e-13) << style;
        EXPECT_NEAR(restored.span(kAtom_vx)[0], .5, 1e-13) << style;
        EXPECT_NEAR(restored.span(kAtom_fx)[0], 2., 1e-13) << style;
        EXPECT_NEAR(restored.span(kAtom_charge)[0], -1., 1e-13) << style;
        EXPECT_NEAR(restored.meta.get(kEnergy), 23.060547830619, 1e-12)
            << style;
        EXPECT_NEAR(restored.box.cell()(2, 2), 30., 1e-12) << style;
    }
}
TEST(FrameUnits, ElectronicProfileIsDifferentFromLammpsElectron) {
    auto f = ATV::units::convert(physical_frame(), "electronic");
    EXPECT_NEAR(f.span(kAtom_vx)[0], .5, 1e-14); // fs, not metal ps
    EXPECT_NEAR(f.meta.get(kEnergy), 1., 1e-12); // eV, not electron Hartree
    auto electron = ATV::units::convert(f, "electron");
    EXPECT_NEAR(electron.span(kAtom_x)[0], 2. * units::kAngstromInBohr, 1e-12);
    EXPECT_NEAR(electron.meta.get(kEnergy), 1. / units::kHartreeInEv, 1e-12);
}
TEST(FrameUnits, UnknownFloatingFieldsFailWithoutChangingSource) {
    auto f = physical_frame();
    f.meta.insert(MetaKey<double>{"unknown_observable", true}, 4.);
    EXPECT_THROW(ATV::units::convert(f, "metal"), std::invalid_argument);
    EXPECT_EQ(ATV::units::frame_profile(f), "real");
    EXPECT_DOUBLE_EQ(f.span(kAtom_vx)[0], .5);
    auto extra = physical_frame();
    extra.block("atoms").insert("custom_float", cpu::ColumnSpec{});
    EXPECT_THROW(ATV::units::convert(extra, "si"), std::invalid_argument);
}
TEST(FrameUnits, MissingUnitsAndMissingLjReferenceAreRejected) {
    cpu::Frame f(1);
    EXPECT_THROW(ATV::units::convert(f, "metal"), std::invalid_argument);
    ATV::units::annotate(f, "lj");
    EXPECT_THROW(ATV::units::convert(f, "metal"), std::invalid_argument);
    EXPECT_NO_THROW(ATV::units::convert(f, "lj"));
    EXPECT_THROW(ATV::units::convert(physical_frame(), "lj"),
                 std::invalid_argument);
}
TEST(FrameUnits, LjReferenceDefinesPhysicalTimeAndChargeAndPersists) {
    auto physical = physical_frame();
    units::ReducedReference argon{"argon", 6.6335209e-26, 3.4e-10,
                                  1.654017e-21};
    auto reduced = ATV::units::convert(physical, "lj", argon);
    ASSERT_TRUE(ATV::units::reduced_reference(reduced));
    EXPECT_EQ(ATV::units::reduced_reference(reduced)->identity, "argon");
    auto restored = ATV::units::convert(reduced, "real");
    EXPECT_NEAR(restored.span(kAtom_vx)[0], .5, 1e-13);
    EXPECT_NEAR(restored.span(kAtom_charge)[0], -1., 1e-13);
    auto other = argon;
    other.identity = "other";
    other.length_m *= 2;
    auto changed = ATV::units::convert(reduced, "lj", other);
    EXPECT_NEAR(changed.span(kAtom_x)[0], reduced.span(kAtom_x)[0] / 2, 1e-14);
    EXPECT_EQ(ATV::units::reduced_reference(changed)->identity, "other");
    auto invalid = argon;
    invalid.mass_kg = -1;
    EXPECT_THROW(ATV::units::convert(physical, "lj", invalid),
                 std::invalid_argument);
}
TEST(PhysicalUnits, ElectricalDefinitionsAndMolarEnergyHavePhysicalMeaning) {
    const auto real = units::PhysicalUnits::from_style("real");
    const auto si = units::PhysicalUnits::from_style("si");
    const auto nano = units::PhysicalUnits::from_style("nano");
    EXPECT_NEAR(units::convert(1., real.energy(), si.energy()),
                4184. / units::avogadro, 1e-35);
    EXPECT_NEAR(units::convert(1., real.pressure(), si.pressure()), 101325.,
                1e-9);
    EXPECT_NEAR(units::convert(1., nano.electric_field(), si.electric_field()),
                1.e9, 1e-6);
}

template <class T>
concept FrameHasUnitAlgorithms =
    requires(T& frame) { frame.set_units("real"); };
template <class T>
concept FrameHasUnitConversion =
    requires(const T& frame) { frame.converted_to("real"); };
static_assert(!FrameHasUnitAlgorithms<cpu::Frame>);
static_assert(!FrameHasUnitConversion<cpu::Frame>);

TEST(FrameUnits, SolverPublicationPreservesParentBuffersAndGeometry) {
    auto frame = physical_frame();
    auto* const position = frame.span(kAtom_x).data();
    auto* const force = frame.span(kAtom_fx).data();
    auto* const charge = frame.span(kAtom_charge).data();
    units::evaluate_in(frame, "electronic", [](cpu::Frame& native) {
        native.span(kAtom_fx)[0] = 3.;
        native.span(kAtom_charge)[0] = -.25;
        native.meta.get(kEnergy) = 2.;
    });
    EXPECT_EQ(position, frame.span(kAtom_x).data());
    EXPECT_EQ(force, frame.span(kAtom_fx).data());
    EXPECT_EQ(charge, frame.span(kAtom_charge).data());
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 2.);
    EXPECT_NEAR(frame.span(kAtom_fx)[0], 3. * 23.060547830619, 1.e-11);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_charge)[0], -.25);
    EXPECT_NEAR(frame.meta.get(kEnergy), 2. * 23.060547830619, 1.e-11);
}
TEST(FrameUnits, SolverFailureDoesNotPublishPartialResults) {
    auto frame = physical_frame();
    EXPECT_THROW(units::evaluate_in(frame, "electronic",
                                    [](cpu::Frame& native) {
                                        native.span(kAtom_fx)[0] = 123.;
                                        native.meta.get(kEnergy) = 456.;
                                        throw std::runtime_error(
                                            "failed solve");
                                    }),
                 std::runtime_error);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 2.);
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 23.060547830619);
}
#ifdef ATV_HAS_HF
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/scf/methods.h"
TEST(FrameUnits, ElectronicForcesAndEnergyAgreeAcrossRunStyles) {
    cpu::Frame native(2);
    units::annotate(native, "electronic");
    native.span(kAtom_x)[1] = .8;
    for (int i = 0; i < 2; ++i) {
        native.span(kAtom_atomic_number)[i] = 1;
        native.span(kAtom_mass)[i] = 1.008;
    }
    native.box = cpu::Box::cubic(20.);
    cpu::scf::RHF reference{cpu::scf::BasisSet::STO3G};
    reference.forces(native);
    for (const auto* style :
         {"metal", "real", "si", "electron", "micro", "nano", "cgs"}) {
        SCOPED_TRACE(style);
        auto work = units::convert(native, style);
        auto* const force = work.span(kAtom_fx).data();
        cpu::Evaluate evaluation;
        evaluation.set_units(style);
        cpu::scf::RHF method{cpu::scf::BasisSet::STO3G};
        method.set_units(style);
        evaluation.on(evaluate::Phase::evaluate).add(std::move(method));
        evaluation.build();
        evaluation.run(work);
        EXPECT_EQ(force, work.span(kAtom_fx).data());
        auto result = units::convert(work, "electronic");
        EXPECT_NEAR(result.meta.get(kEnergy), native.meta.get(kEnergy), 1.e-9);
        for (int i = 0; i < 2; ++i)
            EXPECT_NEAR(result.span(kAtom_fx)[i], native.span(kAtom_fx)[i],
                        1.e-8);
    }
}
#endif

#ifdef ATV_HAS_SEMIEMPIRICAL
#include "atomiverse/cpu/semiempirical/am1_solver.h"
TEST(FrameUnits, NddoBuildAndRunNormalizeGeometryBeforeValidation) {
    cpu::Frame native(2);
    units::annotate(native, "electronic");
    native.span(kAtom_x)[1] = .74;
    native.span(kAtom_atomic_number)[0] = native.span(kAtom_atomic_number)[1] =
        1;
    native.box = cpu::Box::cubic(20.);
    cpu::semiempirical::AM1 reference;
    reference.forces(native);
    for (const auto* style : {"real", "si", "electron"}) {
        SCOPED_TRACE(style);
        auto work = units::convert(native, style);
        cpu::semiempirical::AM1 method;
        EXPECT_NO_THROW(method.build(work));
        method.forces(work);
        auto result = units::convert(work, "electronic");
        EXPECT_NEAR(result.meta.get(kEnergy), native.meta.get(kEnergy), 1.e-8);
        EXPECT_NEAR(result.span(kAtom_fx)[0], native.span(kAtom_fx)[0], 1.e-7);
    }
}
#endif
static_assert(std::is_trivially_copyable_v<units::Conversion>);

TEST(PhysicalUnits, PreparedElectronicContextKeepsItsOwnProfileNameAndClock) {
    const auto native = units::UnitSystem::electronic();
    const auto metal = units::from_style(units::Style::metal);
    EXPECT_STREQ(native.name(), "electronic");
    EXPECT_STREQ(metal.name(), "metal");
    EXPECT_DOUBLE_EQ(native.time_from_femtoseconds(1.), 1.);
    EXPECT_DOUBLE_EQ(metal.time_from_femtoseconds(1.), .001);
}

TEST(FrameUnits, ThermostatEnergyAndKineticMonteCarloMetadataConvert) {
    auto f = physical_frame();
    auto put = [&f]<class T>(const MetaKey<T>& key, T value) {
        if (f.meta.contains(key.key))
            f.meta.set(key, value);
        else
            f.meta.insert(key, value);
    };
    put(kThermostatEnergy, 23.060547830619);
    // The KMC clock stays in ps, whatever the run's time unit.
    const MetaKey<double> clock{"kmc_time_ps", true};
    const MetaKey<double> waiting{"kmc_waiting_time_ps", true};
    const MetaKey<MetaF64x3> displacement{"kmc_displacement", true};
    put(clock, 2.5);
    put(waiting, .25);
    put(displacement, MetaF64x3{1., 2., 3.});
    const auto out = ATV::units::convert(f, "nano");
    const auto metal = ATV::units::convert(f, "metal");
    EXPECT_NEAR(metal.meta.get(kThermostatEnergy), 1., 1e-12);
    EXPECT_DOUBLE_EQ(out.meta.get(clock), 2.5);
    EXPECT_DOUBLE_EQ(out.meta.get(waiting), .25);
    EXPECT_NEAR(out.meta.get(displacement)[2], .3, 1e-14);
}
