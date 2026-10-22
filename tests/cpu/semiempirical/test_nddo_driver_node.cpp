/// @file tests/cpu/semiempirical/test_nddo_driver_node.cpp
/// @brief Unit: AM1 / RM1 are Driver nodes on nddo::Phase.

#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include "atomiverse/cpu/semiempirical/stages/nddo_stages.h"

using ATV::cpu::semiempirical::AM1;
using ATV::cpu::semiempirical::NddoConverged;
using ATV::cpu::semiempirical::NddoStep;
using ATV::cpu::semiempirical::RM1;

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
    bool conv = false;
    NddoStep diagonalize(ATV::nddo::Phase::diagonalize, "nddo_diagonalize",
                         noop);
    NddoStep density(ATV::nddo::Phase::density, "nddo_density", noop);
    NddoStep fock(ATV::nddo::Phase::fock, "nddo_fock", noop);
    NddoStep check(ATV::nddo::Phase::check, "nddo_check", noop);
    NddoConverged crit(conv);

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

TEST(NddoDriverNode, PublicMembersHaveNoConfigBag) {
    RM1 method{/*charge=*/1, /*multiplicity=*/1};
    method.max_iter = 30;
    EXPECT_EQ(method.charge, 1);
    EXPECT_EQ(method.max_iter, 30);
    EXPECT_EQ(method.qm_engine_name(), "rm1");
    EXPECT_EQ(method.as_embedding(), nullptr);
}
