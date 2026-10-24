/// @file tests/cpu/semiempirical/test_nddo_driver_node.cpp
/// @brief Unit: AM1 / RM1 are Driver nodes on nddo::Phase.

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include "atomiverse/cpu/semiempirical/stages/nddo_stages.h"

using ATV::cpu::semiempirical::AM1;
using ATV::cpu::semiempirical::NddoConverged;
using ATV::cpu::semiempirical::NddoStep;
using ATV::cpu::semiempirical::RM1;

namespace {

struct LoopHarness {
    ATV::cpu::Frame frame{1};
    ATV::cpu::Context ctx;
    ATV::ConvergenceMetrics metrics;

    ATV::cpu::LoopState state() {
        return ATV::cpu::LoopState{0, frame, ctx, metrics};
    }
};

} // namespace

TEST(NddoDriverNode, InheritsCpuDriverNddoPhase) {
    static_assert(std::is_base_of_v<ATV::cpu::Driver<ATV::nddo::Phase>, AM1>);
    static_assert(std::is_base_of_v<ATV::cpu::Driver<ATV::nddo::Phase>, RM1>);
    static_assert(std::is_base_of_v<ATV::cpu::QmEngine, RM1>);
    EXPECT_EQ(ATV::slot_count<ATV::nddo::Phase>(), 4);
    EXPECT_EQ(ATV::nddo::kind_of(ATV::nddo::Phase{}), ATV::PhaseKind::SCF);
    const std::vector<std::string> names =
        ATV::nddo::phase_names(ATV::nddo::Phase{});
    ASSERT_EQ(names.size(), 4u);
    EXPECT_EQ(names[0], "diagonalize");
    EXPECT_EQ(names[1], "density");
    EXPECT_EQ(names[2], "fock");
    EXPECT_EQ(names[3], "check");
}

TEST(NddoDriverNode, StageDefaultBindings) {
    auto noop = []() {};
    double density_residual = 1.0;
    double energy_residual = 1.0;
    NddoStep diagonalize(ATV::nddo::Phase::diagonalize, "nddo_diagonalize",
                         noop);
    NddoStep density(ATV::nddo::Phase::density, "nddo_density", noop);
    NddoStep fock(ATV::nddo::Phase::fock, "nddo_fock", noop);
    NddoStep check(ATV::nddo::Phase::check, "nddo_check", noop);
    NddoConverged crit(density_residual, energy_residual, 1e-8, 1e-10);

    auto bind0 = [](const auto& c) {
        auto b = c.default_bindings();
        EXPECT_EQ(b.size(), 1u);
        EXPECT_EQ(b[0].kind, ATV::PhaseKind::SCF);
        return b[0].phase;
    };
    EXPECT_EQ(bind0(diagonalize), ATV::slot_of(ATV::nddo::Phase::diagonalize));
    EXPECT_EQ(bind0(density), ATV::slot_of(ATV::nddo::Phase::density));
    EXPECT_EQ(bind0(fock), ATV::slot_of(ATV::nddo::Phase::fock));
    EXPECT_EQ(bind0(check), ATV::slot_of(ATV::nddo::Phase::check));
    EXPECT_EQ(bind0(crit), ATV::slot_of(ATV::nddo::Phase::check));
    EXPECT_EQ(diagonalize.name(), "nddo_diagonalize");
    EXPECT_EQ(crit.name(), "nddo_converged");
}

TEST(NddoDriverNode, CriterionRequiresBothResidualsAtTolerance) {
    LoopHarness harness;
    double density_residual = 1e-8;
    double energy_residual = 1.0;
    NddoConverged on_density(density_residual, energy_residual, 1e-8, 1e-10);
    EXPECT_EQ(on_density.verdict(harness.state()), ATV::Verdict::Continue);

    density_residual = 2e-8;
    NddoConverged above(density_residual, energy_residual, 1e-8, 1e-10);
    EXPECT_EQ(above.verdict(harness.state()), ATV::Verdict::Continue);

    energy_residual = 1e-10;
    NddoConverged on_energy(density_residual, energy_residual, 1e-8, 1e-10);
    EXPECT_EQ(on_energy.verdict(harness.state()), ATV::Verdict::Continue);
    density_residual = 1e-8;
    EXPECT_EQ(on_energy.verdict(harness.state()), ATV::Verdict::Converged);
}

TEST(NddoDriverNode, PublicMembersHaveNoConfigBag) {
    RM1 method{/*charge=*/1, /*multiplicity=*/1};
    method.max_iter = 30;
    EXPECT_EQ(method.charge, 1);
    EXPECT_EQ(method.max_iter, 30);
    EXPECT_EQ(method.qm_engine_name(), "rm1");
    EXPECT_EQ(method.as_embedding(), nullptr);
    EXPECT_EQ(method.default_budget(), -1);
}

TEST(NddoDriverNode, CriterionUpdatesControlsWithoutRebindingResiduals) {
    LoopHarness harness;
    double density_residual = 2.e-8, energy_residual = 2.e-10;
    NddoConverged criterion(density_residual, energy_residual, 1.e-8, 1.e-10);
    EXPECT_EQ(criterion.verdict(harness.state()), ATV::Verdict::Continue);
    criterion.set_tolerances(2.e-8, 2.e-10);
    EXPECT_EQ(criterion.verdict(harness.state()), ATV::Verdict::Converged);
    energy_residual = 3.e-10;
    EXPECT_EQ(criterion.verdict(harness.state()), ATV::Verdict::Continue);
}

TEST(NddoDriverNode, MoveKeepsPersistentStageAndCriterionAddresses) {
    const auto check = []<typename Method>() {
        Method original;
        ASSERT_EQ(original.plan().slots.size(), 4u);
        ASSERT_EQ(original.plan().lifecycle_only.size(), 1u);
        auto* stage = original.plan().slots[0].ops[0].component;
        auto* criterion = original.plan().lifecycle_only[0];
        Method moved(std::move(original));
        EXPECT_EQ(moved.plan().slots[0].ops[0].component, stage);
        EXPECT_EQ(moved.plan().lifecycle_only[0], criterion);
        moved.reset_history();
        EXPECT_EQ(moved.plan().slots[0].ops[0].component, stage);
        EXPECT_EQ(moved.plan().lifecycle_only[0], criterion);
    };
    check.operator()<AM1>();
    check.operator()<RM1>();
}

TEST(NddoDriverNode, MountedMoveIsRejectedAndLifecycleIsReleasedOnce) {
    const auto check = []<typename Method>() {
        struct Probe : ATV::cpu::Component {
            int builds = 0, destroys = 0;
            std::string name() const override { return "nddo_lifecycle_probe"; }
            void build(const ATV::cpu::BuildContext&) override { ++builds; }
            void launch(const ATV::cpu::LaunchContext&) override {}
            void destroy(const ATV::cpu::DestroyContext&) override {
                ++destroys;
            }
        } probe;
        ATV::cpu::Frame frame(1);
        ATV::cpu::Runtime runtime;
        Method method;
        method.on(ATV::nddo::Phase::check).add_ref(probe);
        runtime.init();
        runtime.load(method.plan());
        runtime.set_frame(frame);
        method.build_on(runtime);
        EXPECT_EQ(probe.builds, 1);
        EXPECT_EQ(probe.destroys, 0);
        EXPECT_THROW({ Method moved(std::move(method)); }, std::logic_error);
        EXPECT_NE(method.execution_plan(), nullptr);
        EXPECT_EQ(probe.destroys, 0);
        method.destroy_on(runtime);
        EXPECT_EQ(probe.destroys, 1);
        EXPECT_EQ(method.execution_plan(), nullptr);
        method.build_on(runtime);
        EXPECT_EQ(probe.builds, 2);
        method.destroy_on(runtime);
        EXPECT_EQ(probe.destroys, 2);
    };
    check.operator()<AM1>();
    check.operator()<RM1>();
}
