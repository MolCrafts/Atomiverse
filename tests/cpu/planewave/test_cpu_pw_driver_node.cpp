/// @file tests/cpu/planewave/test_cpu_pw_driver_node.cpp
/// @brief Unit: PW is a Driver node with multi-path pw::Phase stage bindings.

#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/stages/density_stage.h"
#include "atomiverse/cpu/planewave/stages/diagonalize_stage.h"
#include "atomiverse/cpu/planewave/stages/mix_stage.h"
#include "atomiverse/cpu/planewave/stages/pw_converged.h"
#include "atomiverse/cpu/planewave/stages/veff_stage.h"

using ATV::cpu::pw::DensityStage;
using ATV::cpu::pw::DiagonalizeStage;
using ATV::cpu::pw::GthPseudopotential;
using ATV::cpu::pw::MixStage;
using ATV::cpu::pw::PW;
using ATV::cpu::pw::PwConverged;
using ATV::cpu::pw::UpfPseudopotential;
using ATV::cpu::pw::VeffStage;
using ATV::electronic::Smearing;

namespace {
constexpr double kEcutRy = 30.0; // [Ry]
constexpr int kKMesh = 4;        // Monkhorst-Pack points per dimension
} // namespace

// PW is the plane-wave self-consistency Driver node (MeanFieldMethod / GpwDft).
TEST(PwDriverNode, InheritsCpuDriverPwPhase) {
    static_assert(std::is_base_of_v<ATV::cpu::Driver<ATV::pw::Phase>, PW>);
    PW m{
        {std::make_shared<GthPseudopotential>(molrs::Element::H, "GTH-PBE-q1")},
        kEcutRy,
        kKMesh};
    EXPECT_EQ(ATV::slot_count<ATV::pw::Phase>(), 5);
    EXPECT_EQ(ATV::pw::kind_of(ATV::pw::Phase{}), ATV::PhaseKind::SCF);
}

// Stage default bindings target the matching pw::Phase slots (all product paths).
TEST(PwDriverNode, StageDefaultBindings) {
    auto noop = []() {};
    bool conv = false;
    VeffStage veff(noop);
    DiagonalizeStage diag(noop);
    DensityStage dens(noop);
    MixStage mix(noop);
    PwConverged crit(conv);

    auto bind0 = [](const auto& c) {
        auto b = c.default_bindings();
        EXPECT_EQ(b.size(), 1u);
        EXPECT_EQ(b[0].kind, ATV::PhaseKind::SCF);
        return b[0].phase;
    };
    EXPECT_EQ(bind0(veff), ATV::slot_of(ATV::pw::Phase::veff));
    EXPECT_EQ(bind0(diag), ATV::slot_of(ATV::pw::Phase::diagonalize));
    EXPECT_EQ(bind0(dens), ATV::slot_of(ATV::pw::Phase::density));
    EXPECT_EQ(bind0(mix), ATV::slot_of(ATV::pw::Phase::mix));
    EXPECT_EQ(bind0(crit), ATV::slot_of(ATV::pw::Phase::check));
}

// Path-local StepFns are the multi-path shape (fixed-occ / smear × spin).
TEST(PwDriverNode, PathLocalStepFnsAreCallable) {
    int hits = 0;
    VeffStage veff([&]() { ++hits; });
    DiagonalizeStage diag([&]() { ++hits; });
    DensityStage dens([&]() { ++hits; });
    MixStage mix([&]() { ++hits; });

    // Construction + binding surface is the unit gate (no full SCF).
    EXPECT_EQ(veff.name(), "pw_veff_stage");
    EXPECT_EQ(diag.name(), "pw_diagonalize_stage");
    EXPECT_EQ(dens.name(), "pw_density_stage");
    EXPECT_EQ(mix.name(), "pw_mix_stage");
    EXPECT_EQ(hits, 0);
}

// Public multi-path members remain constructible without *Config bags.
TEST(PwDriverNode, MultiPathPublicMembers) {
    PW m{
        {std::make_shared<GthPseudopotential>(molrs::Element::H, "GTH-PBE-q1")},
        kEcutRy,
        kKMesh};
    m.spin_polarized = true;
    m.multiplicity = 3;
    m.smearing = ATV::electronic::Smearing::fermi_dirac;
    m.degauss = 0.02;
    m.extra_bands = 4;
    EXPECT_TRUE(m.spin_polarized);
    EXPECT_EQ(m.multiplicity, 3);
    EXPECT_EQ(static_cast<int>(m.smearing),
              static_cast<int>(ATV::electronic::Smearing::fermi_dirac));
}

// Identity is fixed at construction: an empty pseudopotential is refused
// there, not at the first run.
TEST(PwDriverNode, EmptyPseudopotentialThrowsAtConstruction) {
    EXPECT_THROW(
        (PW{{std::make_shared<GthPseudopotential>(molrs::Element::H, "")},
            kEcutRy,
            kKMesh}),
        std::invalid_argument);
    EXPECT_THROW(
        (PW{{std::make_shared<UpfPseudopotential>("")}, kEcutRy, kKMesh}),
        std::invalid_argument);
}

TEST(PwDriverNode, NestedMountBindsToOneRuntime) {
    PW method{
        {std::make_shared<GthPseudopotential>(molrs::Element::H, "GTH-PBE-q1")},
        kEcutRy,
        kKMesh};
    ATV::cpu::Runtime runtime;
    ATV::cpu::Runtime other;
    ATV::cpu::DriverNode& node = method;
    // A nested PW attaches to its parent Runtime and refuses a second one.
    EXPECT_NO_THROW(node.build_on(runtime));
    EXPECT_THROW(node.build_on(other), std::logic_error);
    node.destroy_on(runtime);
    EXPECT_NO_THROW(node.build_on(other));
    node.destroy_on(other);
    EXPECT_TRUE(method.provides_energy());
    EXPECT_TRUE(method.provides_force());
}
