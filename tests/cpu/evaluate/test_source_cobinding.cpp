/// @file tests/cpu/evaluate/test_source_cobinding.cpp
/// @brief Unit: two accumulating energy sources in ONE `cpu::Evaluate` pass
///        both land in `kEnergy` (driver-owned zero + `+=` contract).
///
/// e2e-in-tests wave-2: no RHF/SCF under this unit. QM×MM co-binding pin is
/// `regressions/architecture/lj_rhf_coupled`. Spec background:
/// `.claude/specs/esmm-contract-05-accumulate-switch` (U7).

#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// @brief Energy the classical stub contributes, per launch [eV].
constexpr double kStubEnergy = 1.5;

/// @brief Assertion tolerance on `kEnergy` [eV].
constexpr double kEnergyTol = 1e-12;

/// @brief Minimal classical energy source: `kEnergy += ...` only.
class AccumulatingStub final : public cpu::Potential {
  public:
    AccumulatingStub(double e, std::string tag) : e_(e), tag_(std::move(tag)) {}

    std::string name() const override { return tag_; }

    void launch(const cpu::LaunchContext& ctx) override {
        ++launches;
        ctx.frame.meta.get(kEnergy) += e_;
    }

    int launches = 0;

  private:
    double e_;
    std::string tag_;
};

cpu::Frame make_h2() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = -0.3561;
    f.span(kAtom_x)[1] = 0.3561;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

TEST(SourceCoBinding, TwoAccumulatingStubsSum) {
    cpu::Frame frame = make_h2();

    AccumulatingStub a{kStubEnergy, "stub_a"};
    AccumulatingStub b{2.0 * kStubEnergy, "stub_b"};

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(a, /*order=*/0);
    eval.on(evaluate::Phase::evaluate).add_ref(b, /*order=*/10);
    eval.set_threads(1);
    eval.build();
    eval.run(frame);

    ASSERT_EQ(a.launches, 1);
    ASSERT_EQ(b.launches, 1);
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergy)), 3.0 * kStubEnergy,
                kEnergyTol)
        << "two classical `+=` sources in one Evaluate pass did not sum — the "
           "harness (ZeroEnergy @ -999 + one pass) is broken";
}

TEST(SourceCoBinding, StubSumIndependentOfBindingOrder) {
    double results[2] = {0.0, 0.0};

    for (int variant = 0; variant < 2; ++variant) {
        cpu::Frame frame = make_h2();
        AccumulatingStub a{kStubEnergy, "a"};
        AccumulatingStub b{2.0 * kStubEnergy, "b"};

        cpu::Evaluate eval;
        const int a_order = (variant == 0) ? 0 : 10;
        const int b_order = (variant == 0) ? 10 : 0;
        eval.on(evaluate::Phase::evaluate).add_ref(a, a_order);
        eval.on(evaluate::Phase::evaluate).add_ref(b, b_order);
        eval.set_threads(1);
        eval.build();
        eval.run(frame);
        results[variant] = static_cast<double>(frame.meta.get(kEnergy));
    }

    EXPECT_NEAR(results[0], results[1], kEnergyTol);
    EXPECT_NEAR(results[0], 3.0 * kStubEnergy, kEnergyTol);
}
