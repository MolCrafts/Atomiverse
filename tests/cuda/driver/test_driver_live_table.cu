/// @file tests/cuda/driver/test_driver_live_table.cu
/// @brief Live mount table after cuda::Evaluate::build() (no device).

#include <gtest/gtest.h>

#include "atomiverse/core/driver_template.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/potential.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"

namespace {

class LiveProbe : public ATV::cuda::Potential {
  public:
    std::string name() const override { return "live_probe"; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

} // namespace

TEST(TestCudaDriver, BuildRetainsMountsAndSecondCompileMatchesPlan) {
    ATV::cuda::Evaluate eval;
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
