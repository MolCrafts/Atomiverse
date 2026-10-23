/// @file tests/cpu/scf/test_ccsd_driver.cpp
/// @brief Driver shape of cpu::scf::CCSD. No amplitude or SCF solve.

#include <stdexcept>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/post_hf.h"
#include "atomiverse/cpu/scf/rhf.h"

using namespace ATV;

// Driver shape only. The correlation solve stays in
// regressions/scf/coupled-cluster.
TEST(CcsdDriver, PresetIsCcDriver) {
    using Ccsd = cpu::scf::CCSD<cpu::scf::RHF>;
    static_assert(std::is_base_of_v<cpu::Driver<ATV::cc::Phase>, Ccsd>,
                  "cpu::scf::CCSD must be cpu::Driver<cc::Phase>");
    static_assert(std::is_base_of_v<cpu::DriverNode, Ccsd>,
                  "cpu::scf::CCSD must be a nestable DriverNode");
    static_assert(!std::is_base_of_v<cpu::HF, Ccsd>,
                  "cpu::scf::CCSD must not carry the HF tag");
    static_assert(!std::is_base_of_v<cpu::Component, Ccsd>,
                  "cpu::scf::CCSD must not be a Component");

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    cpu::scf::CCSD<cpu::scf::RHF> ccsd{rhf, /*diis=*/false, /*frozen=*/1,
                                       /*tol=*/1e-8, /*max_iter=*/40};
    EXPECT_TRUE(ccsd.provides_energy());
    EXPECT_TRUE(ccsd.provides_force());
    EXPECT_TRUE(ccsd.requires_full_frame());
    EXPECT_FALSE(ccsd.converged());
    EXPECT_EQ(ccsd.default_budget(), ccsd.max_iter);
    EXPECT_FALSE(ccsd.diis);
    EXPECT_EQ(ccsd.frozen, 1);
    EXPECT_DOUBLE_EQ(ccsd.tol, 1e-8);
    EXPECT_EQ(ccsd.max_iter, 40);

    const auto& plan = ccsd.plan();
    EXPECT_EQ(plan.kind, PhaseKind::SCF);
    EXPECT_TRUE(plan.requires_full_frame);
    ASSERT_EQ(plan.slots.size(),
              static_cast<std::size_t>(ATV::slot_count<ATV::cc::Phase>()));
    const char* names[] = {"ccsd_update", "ccsd_mix", "ccsd_energy",
                           "ccsd_check"};
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
    EXPECT_EQ(crit->name(), "ccsd_amplitude");
    EXPECT_TRUE(crit->contributes_to_convergence());

    cpu::Frame frame;
    try {
        ccsd.forces(frame);
        FAIL() << "frozen-core forces must throw before a solve";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("frozen-core"), std::string::npos);
    }
}
