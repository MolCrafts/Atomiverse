/// @file tests/cpu/scf/test_cc2_driver.cpp
/// @brief Driver shape of cpu::scf::CC2. No amplitude or SCF solve.

#include <type_traits>

#include <gtest/gtest.h>

#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/scf/post_hf.h"
#include "atomiverse/cpu/scf/rhf.h"

using namespace ATV;

// Driver shape only. The correlation solve stays in
// regressions/scf/coupled-cluster.
TEST(Cc2Driver, PresetIsCcDriver) {
    using Cc2 = cpu::scf::CC2<cpu::scf::RHF>;
    static_assert(std::is_base_of_v<cpu::Driver<ATV::cc::Phase>, Cc2>,
                  "cpu::scf::CC2 must be cpu::Driver<cc::Phase>");
    static_assert(std::is_base_of_v<cpu::DriverNode, Cc2>,
                  "cpu::scf::CC2 must be a nestable DriverNode");
    static_assert(!std::is_base_of_v<cpu::HF, Cc2>,
                  "cpu::scf::CC2 must not carry the HF tag");
    static_assert(!std::is_base_of_v<cpu::Component, Cc2>,
                  "cpu::scf::CC2 must not be a Component");

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    cpu::scf::CC2<cpu::scf::RHF> cc2{rhf, /*diis=*/false, /*tol=*/1e-8,
                                     /*max_iter=*/40};
    EXPECT_TRUE(cc2.provides_energy());
    EXPECT_FALSE(cc2.provides_force());
    EXPECT_TRUE(cc2.requires_full_frame());
    EXPECT_FALSE(cc2.converged());
    EXPECT_EQ(cc2.default_budget(), cc2.max_iter);
    EXPECT_FALSE(cc2.diis);
    EXPECT_DOUBLE_EQ(cc2.tol, 1e-8);
    EXPECT_EQ(cc2.max_iter, 40);

    const auto& plan = cc2.plan();
    EXPECT_EQ(plan.kind, PhaseKind::SCF);
    EXPECT_TRUE(plan.requires_full_frame);
    ASSERT_EQ(plan.slots.size(),
              static_cast<std::size_t>(ATV::slot_count<ATV::cc::Phase>()));
    const char* names[] = {"cc2_update", "cc2_mix", "cc2_energy", "cc2_check"};
    const ATV::cc::Phase phases[] = {
        ATV::cc::Phase::update, ATV::cc::Phase::mix, ATV::cc::Phase::energy,
        ATV::cc::Phase::check};
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
    EXPECT_EQ(crit->name(), "cc2_amplitude");
    EXPECT_TRUE(crit->contributes_to_convergence());
}
