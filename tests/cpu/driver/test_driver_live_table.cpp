/// @file tests/cpu/driver/test_driver_live_table.cpp
/// @brief Single-method gates for the live mount table after build().

#include <gtest/gtest.h>

#include "atomiverse/core/driver_template.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/evaluate/evaluate.h"

namespace {

class LiveProbe : public ATV::cpu::Potential {
  public:
    std::string name() const override { return "live_probe"; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

} // namespace

TEST(TestDriver, BuildRetainsMountsAndSecondCompileMatchesPlan) {
    ATV::cpu::Evaluate eval;
    LiveProbe a;
    LiveProbe b;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(a);
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(b);
    eval.build();
    EXPECT_EQ(eval.mounted().size(), 5u);
    EXPECT_TRUE(eval.is_mounted(ATV::PluginIdentity{&a}));
    EXPECT_TRUE(eval.is_mounted(ATV::PluginIdentity{&b}));
    EXPECT_FALSE((ATV::PluginIdentity{&a} == ATV::PluginIdentity{&b}));

    const auto first = eval.plan();
    eval.build();
    const auto& second = eval.plan();
    ASSERT_EQ(first.slots.size(), second.slots.size());
    for (std::size_t s = 0; s < first.slots.size(); ++s) {
        ASSERT_EQ(first.slots[s].ops.size(), second.slots[s].ops.size());
        for (std::size_t o = 0; o < first.slots[s].ops.size(); ++o)
            EXPECT_EQ(first.slots[s].ops[o].component,
                      second.slots[s].ops[o].component);
    }
}
