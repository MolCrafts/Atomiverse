/// @file tests/cpu/scf/test_dhf_driver.cpp
/// @brief Driver shape of cpu::scf::DHF. No SCF solve.

#include <stdexcept>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dhf.h"

using namespace ATV;

// Driver shape only. The SCF solve stays in regressions/scf/ghf.
TEST(DhfDriver, PresetIsScfDriver) {
    static_assert(
        std::is_base_of_v<cpu::Driver<ATV::scf::Phase>, cpu::scf::DHF>,
        "cpu::scf::DHF must be cpu::Driver<scf::Phase>");
    static_assert(std::is_base_of_v<cpu::DriverNode, cpu::scf::DHF>,
                  "cpu::scf::DHF must be a nestable DriverNode");
    static_assert(std::is_base_of_v<cpu::HF, cpu::scf::DHF>,
                  "cpu::scf::DHF must carry the HF tag");
    static_assert(!std::is_base_of_v<cpu::Component, cpu::scf::DHF>,
                  "cpu::scf::DHF must not be a Component");

    cpu::scf::DHF dhf{cpu::scf::BasisSet::STO3G, 0, 1e-10, 80};
    EXPECT_TRUE(dhf.provides_energy());
    EXPECT_FALSE(dhf.provides_force());
    EXPECT_TRUE(dhf.requires_full_frame());
    EXPECT_EQ(dhf.default_budget(), dhf.max_iter);
    EXPECT_EQ(dhf.charge, 0);
    EXPECT_DOUBLE_EQ(dhf.tol, 1e-10);
    EXPECT_EQ(dhf.max_iter, 80);

    const auto& plan = dhf.plan();
    EXPECT_EQ(plan.kind, PhaseKind::SCF);
    EXPECT_TRUE(plan.requires_full_frame);
    ASSERT_EQ(plan.slots.size(),
              static_cast<std::size_t>(ATV::slot_count<ATV::scf::Phase>()));
    const char* names[] = {"dhf_fock", "dhf_mix", "dhf_diagonalize",
                           "dhf_density", "dhf_check"};
    const ATV::scf::Phase phases[] = {
        ATV::scf::Phase::fock, ATV::scf::Phase::mix,
        ATV::scf::Phase::diagonalize, ATV::scf::Phase::density,
        ATV::scf::Phase::check};
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(plan.slots[static_cast<std::size_t>(i)].phase,
                  ATV::slot_of(phases[i]));
        ASSERT_EQ(plan.slots[static_cast<std::size_t>(i)].ops.size(), 1u);
        const auto* op =
            plan.slots[static_cast<std::size_t>(i)].ops[0].component;
        ASSERT_NE(op, nullptr);
        EXPECT_EQ(op->name(), names[i]);
    }
    ASSERT_EQ(plan.lifecycle_only.size(), 1u);
    const auto* crit =
        dynamic_cast<const cpu::Criterion*>(plan.lifecycle_only[0]);
    ASSERT_NE(crit, nullptr);
    EXPECT_EQ(crit->name(), "dhf_energy_delta");
    EXPECT_TRUE(crit->contributes_to_convergence());

    cpu::Frame frame;
    try {
        dhf.forces(frame);
        FAIL() << "forces must throw";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("dhf_nuclear_forces"),
                  std::string::npos);
    }
}
