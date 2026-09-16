/// @file tests/cpu/communication/test_tag_index_lifetime.cpp
/// @brief Who gets to keep the Frame tag index.
///
/// `DomainRuntime` frees the tag→row map when nothing resolves global tags at
/// run time (~50 B/atom, twice). The predicate used to be
/// `!frame.has_bonded()`, which asks the wrong question: a component can name
/// atoms by tag and carry no topology at all. `FixSettle` is handed water
/// triples as tags and has no bonds, so the engine dropped the one map it
/// needs — and the fix carried a private O(N) rebuild to paper over it.
///
/// The need is now declared (`Component::requires_tag_index`) and propagated
/// through the loop node, so these two tests pin both halves: the release
/// policy itself, and the driver-side scan that feeds it.
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/domain_runtime.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

cpu::Frame two_atoms() {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = 1.0 * i;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = 0.0;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(10.0);
    f.rebuild_tag_index(0, 2);
    return f;
}

/// A force term that resolves its own atoms by tag and owns no topology —
/// the `FixSettle` shape, reduced to the one declaration under test.
class TagNamingPotential : public cpu::Potential {
  public:
    explicit TagNamingPotential(bool declares) : declares_(declares) {}
    std::string name() const override { return "tag_naming"; }
    bool requires_tag_index() const override { return declares_; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::evaluate::Phase::evaluate, 0)};
    }
    void launch(const cpu::LaunchContext& ctx) override {
        ctx.frame.meta.get(kEnergy) += 0.0;
    }

  private:
    bool declares_;
};

} // namespace

TEST(DomainRuntime, ReleasesTheTagIndexWhenNothingResolvesTags) {
    cpu::Frame f = two_atoms();
    ASSERT_FALSE(f.has_bonded());
    ASSERT_EQ(f.meta.get(cpu::kTagIndex).size(), 2u);

    cpu::Context ctx;
    cpu::communication::DomainRuntime dr;
    dr.allocate(ctx);
    dr.bind_frame(f, ctx, /*tag_index_required=*/false);

    EXPECT_TRUE(f.meta.get(cpu::kTagIndex).empty());
}

TEST(DomainRuntime, KeepsTheTagIndexWhenAComponentDeclaresTheNeed) {
    cpu::Frame f = two_atoms();
    ASSERT_FALSE(f.has_bonded()); // the old predicate would drop it here

    cpu::Context ctx;
    cpu::communication::DomainRuntime dr;
    dr.allocate(ctx);
    dr.bind_frame(f, ctx, /*tag_index_required=*/true);

    const auto& index = f.meta.get(cpu::kTagIndex);
    ASSERT_EQ(index.size(), 2u);
    EXPECT_EQ(index.at(f.span(kAtom_tag)[0]), 0);
    EXPECT_EQ(index.at(f.span(kAtom_tag)[1]), 1);
}

TEST(Driver, ReportsTheTagIndexNeedOfItsComponents) {
    cpu::Evaluate without;
    without.set_evaluator(TagNamingPotential{false});
    EXPECT_FALSE(without.requires_tag_index());

    cpu::Evaluate with;
    with.set_evaluator(TagNamingPotential{true});
    EXPECT_TRUE(with.requires_tag_index());
}

/// End of the wire: a declaring component in a real driver keeps the index
/// alive across the run, on a frame with no bonded topology at all.
TEST(Driver, ATagNamingComponentKeepsTheIndexAliveThroughARun) {
    cpu::Frame f = two_atoms();
    ASSERT_FALSE(f.has_bonded());

    cpu::Evaluate eval;
    eval.set_evaluator(TagNamingPotential{true});
    eval.build();
    eval.run(f);

    const auto& index = f.meta.get(cpu::kTagIndex);
    ASSERT_EQ(index.size(), 2u);
    EXPECT_EQ(index.at(f.span(kAtom_tag)[1]), 1);
}
