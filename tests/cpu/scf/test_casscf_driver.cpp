/// @file tests/cpu/scf/test_casscf_driver.cpp
/// @brief Driver shape of cpu::scf::CASSCF. No orbital or SCF solve.

#include <stdexcept>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/scf/post_hf.h"
#include "atomiverse/cpu/scf/rhf.h"

using namespace ATV;

// Driver shape only. The correlation solve stays in
// regressions/scf/multireference.
TEST(CasscfDriver, PresetIsMcDriver) {
    using Cas = cpu::scf::CASSCF<cpu::scf::RHF>;
    static_assert(std::is_base_of_v<cpu::Driver<ATV::mc::Phase>, Cas>,
                  "cpu::scf::CASSCF must be cpu::Driver<mc::Phase>");
    static_assert(std::is_base_of_v<cpu::DriverNode, Cas>,
                  "cpu::scf::CASSCF must be a nestable DriverNode");
    static_assert(!std::is_base_of_v<cpu::HF, Cas>,
                  "cpu::scf::CASSCF must not carry the HF tag");
    static_assert(!std::is_base_of_v<cpu::Component, Cas>,
                  "cpu::scf::CASSCF must not be a Component");

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    rhf.tol = 1e-12;
    cpu::scf::CASSCF<cpu::scf::RHF> loose{rhf, /*ncas=*/2, /*nelecas=*/2};
    EXPECT_DOUBLE_EQ(loose.orb_tol, 1e-8);
    EXPECT_EQ(loose.max_macro, 50);

    cpu::scf::CASSCF<cpu::scf::RHF> mc{rhf, /*ncas=*/2, /*nelecas=*/4,
                                       /*orb_tol=*/1e-6, /*max_macro=*/12};
    EXPECT_TRUE(mc.provides_energy());
    EXPECT_FALSE(mc.provides_force());
    EXPECT_TRUE(mc.requires_full_frame());
    EXPECT_FALSE(mc.converged());
    EXPECT_EQ(mc.default_budget(), mc.max_macro);
    EXPECT_EQ(mc.ncas, 2);
    EXPECT_EQ(mc.nelecas, 4);
    EXPECT_DOUBLE_EQ(mc.orb_tol, 1e-6);
    EXPECT_EQ(mc.max_macro, 12);

    const auto& plan = mc.plan();
    EXPECT_EQ(plan.kind, PhaseKind::SCF);
    EXPECT_TRUE(plan.requires_full_frame);
    ASSERT_EQ(plan.slots.size(),
              static_cast<std::size_t>(ATV::slot_count<ATV::mc::Phase>()));
    const char* names[] = {"casscf_rotate", "casscf_casci", "casscf_gradient",
                           "casscf_check"};
    const ATV::mc::Phase phases[] = {
        ATV::mc::Phase::rotate, ATV::mc::Phase::casci, ATV::mc::Phase::gradient,
        ATV::mc::Phase::check};
    for (int i = 0; i < 4; ++i) {
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
    EXPECT_EQ(crit->name(), "casscf_orbital");
    EXPECT_TRUE(crit->contributes_to_convergence());

    EXPECT_THROW(cpu::scf::CASSCF<cpu::scf::RHF>(rhf, /*ncas=*/0,
                                                 /*nelecas=*/2),
                 std::invalid_argument);
}
