/// @file tests/cpu/driver/test_driver_nesting.cpp
/// @brief Native driver-node nesting — an inner loop cannot stop its parent.
///
/// Migrated from the deleted nesting-adapter test (scf-driver-refactor-04a-nest).
/// Nesting is `on(phase).add_ref(inner, order, budget)` — a native Plan node.
///
/// The bug being pinned here shipped TWICE:
///   - probe 2: LBFGS wrote outer_status.converged unconditionally
///   - probe 3: first NEB image to converge would stop the entire band
/// Both had the same cause: ONE IterationStatus on the Context. Now each loop
/// owns its own and Context merely points at the active one.

#include <gtest/gtest.h>

#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/cpu_driver.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/facade/stage.h"

#include "cpu/driver_test_helpers.h"

using namespace ATV;

namespace {

/// @brief Minimal loop over md::Phase — the PARENT.
class OuterLoop : public cpu::CPUDriver<ATV::md::Phase> {
  public:
    int run(cpu::Frame& frame, int budget) {
        setup_pass(frame);
        int iters = 0;
        for (int s = 0; s < budget; ++s) {
            mutable_outer_status().reset();
            step_once(s);
            ++iters;
            if (poll_criteria_(s) != Verdict::Continue)
                break;
        }
        teardown_pass();
        return iters;
    }

  protected:
    void validate_components() const override {}
};

/// @brief Minimal loop over minimizer::Phase — the CHILD (different enum).
class InnerLoop : public cpu::CPUDriver<ATV::minimizer::Phase> {
  public:
    int default_budget() const override { return 10; }

  protected:
    void validate_components() const override {}
};

/// @brief Counts build/launch/destroy; reports zero force after `at` calls.
class ConvergeAt : public cpu::Component {
  public:
    explicit ConvergeAt(int at) : at_(at) {}
    std::string name() const override { return "converge_at"; }
    void build(const cpu::BuildContext&) override { ++builds; }
    void destroy(const cpu::DestroyContext&) override { ++destroys; }
    void launch(const cpu::LaunchContext& ctx) override {
        const bool done = (calls >= at_);
        ++calls;
        if (auto* m = ctx.runtime.loop.metrics) {
            m->max_force = done ? 0.0 : 1e3;
            m->rms_force = done ? 0.0 : 1e3;
            m->energy_change = done ? 0.0 : 1e3;
            m->has_displacement = false;
        }
    }
    int builds = 0;
    int destroys = 0;
    int calls = 0;

  private:
    int at_;
};

class PublishForce : public cpu::Component {
  public:
    explicit PublishForce(double f) : f_(f) {}
    std::string name() const override { return "publish_force"; }
    void launch(const cpu::LaunchContext& ctx) override {
        if (auto* m = ctx.runtime.loop.metrics) {
            m->max_force = f_;
            m->rms_force = f_;
            m->energy_change = f_;
            m->has_displacement = false;
        }
    }

  private:
    double f_;
};

class RuntimeProbe : public cpu::Component {
  public:
    std::string name() const override { return "runtime_probe"; }
    void build(const cpu::BuildContext& ctx) override {
        ++builds;
        build_runtime = ctx.runtime.exec.runtime;
    }
    void launch(const cpu::LaunchContext& ctx) override {
        ++calls;
        launch_runtime = ctx.runtime.exec.runtime;
        launch_dd = ctx.runtime.md.dd;
        frame_addr = &ctx.frame;
    }
    int builds = 0;
    int calls = 0;
    cpu::Backend* build_runtime = nullptr;
    cpu::Backend* launch_runtime = nullptr;
    const void* launch_dd = nullptr;
    const void* frame_addr = nullptr;
};

cpu::Frame two_atoms() {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(ATV::kAtom_x)[i] = 0.5 * i;
        f.span(ATV::kAtom_y)[i] = 0.0;
        f.span(ATV::kAtom_z)[i] = 0.0;
        f.span(ATV::kAtom_mass)[i] = 1.0;
        f.span(ATV::kAtom_type)[i] = 1;
    }
    f.simbox = ATV::cpu::Box::cubic(10.0);
    return f;
}

} // namespace

// ── ac-006: inner convergence does NOT stop the outer loop ──────────────────

TEST(DriverNesting, NestedConvergenceDoesNotStopTheParent) {
    cpu::Frame frame = two_atoms();

    InnerLoop inner;
    ConvergeAt converge{/*at=*/2};
    inner.on(ATV::minimizer::Phase::convergence_check).add_ref(converge);
    inner.add_criterion(cpu::ForceConverged{});

    OuterLoop outer;
    PublishForce big{1e3};
    outer.on(ATV::md::Phase::force).add_ref(big);
    outer.on(ATV::md::Phase::end_of_step)
        .add_ref(inner, /*order=*/0, /*budget=*/10);
    outer.add_criterion(cpu::ForceConverged{});
    outer.build();

    const int outer_iters = outer.run(frame, /*budget=*/5);

    EXPECT_EQ(outer_iters, 5)
        << "the inner loop stopped the outer loop — the bug that "
           "shipped in probe 2 and again in probe 3";
    EXPECT_FALSE(outer.converged())
        << "the parent adopted its child's verdict; status is not per-loop";

    // Borrowed inner is inspectable after the parent run (ac-007).
    EXPECT_TRUE(inner.converged());
    EXPECT_EQ(inner.stop_reason(), "force_converged");
}

// ── ac-005: one Runtime, one build, budget per outer iteration ──────────────

TEST(DriverNesting, RunsOnTheParentsRuntime) {
    cpu::Frame frame = two_atoms();

    InnerLoop inner;
    RuntimeProbe inner_probe;
    inner.on(ATV::minimizer::Phase::force).add_ref(inner_probe);

    OuterLoop outer;
    RuntimeProbe outer_probe;
    outer.on(ATV::md::Phase::force).add_ref(outer_probe);
    outer.on(ATV::md::Phase::end_of_step)
        .add_ref(inner, /*order=*/0, /*budget=*/3);
    outer.build();

    outer.run(frame, /*budget=*/4);

    EXPECT_NE(inner_probe.launch_runtime, nullptr);
    EXPECT_EQ(inner_probe.launch_runtime, outer_probe.launch_runtime);
    EXPECT_EQ(inner_probe.launch_dd, outer_probe.launch_dd);
    EXPECT_EQ(inner_probe.frame_addr, outer_probe.frame_addr);
    EXPECT_EQ(inner_probe.build_runtime, outer_probe.build_runtime);
}

TEST(DriverNesting, InnerBuildRunsOnceAtOuterBuild) {
    cpu::Frame frame = two_atoms();

    InnerLoop inner;
    ConvergeAt inner_comp{/*at=*/999}; // never converges; full budget
    inner.on(ATV::minimizer::Phase::convergence_check).add_ref(inner_comp);

    OuterLoop outer;
    outer.on(ATV::md::Phase::end_of_step)
        .add_ref(inner, /*order=*/0, /*budget=*/3);

    outer.build();
    EXPECT_EQ(inner_comp.builds, 0)
        << "build() compiles Plans; it must not run them";

    outer.run(frame, /*budget=*/4);

    EXPECT_EQ(inner_comp.builds, 1);
    EXPECT_EQ(inner_comp.destroys, 1);
    // 4 outer × 3 inner = 12 launches
    EXPECT_EQ(inner_comp.calls, 12);
}

// ── Child keeps its own phase enum ──────────────────────────────────────────

TEST(DriverNesting, ChildKeepsItsOwnPhaseEnum) {
    static_assert(std::is_base_of_v<cpu::CPUDriver<ATV::md::Phase>, OuterLoop>,
                  "parent loops over md::Phase");
    static_assert(
        std::is_base_of_v<cpu::CPUDriver<ATV::minimizer::Phase>, InnerLoop>,
        "child loops over minimizer::Phase");
    static_assert(std::is_base_of_v<cpu::Driver, OuterLoop>,
                  "a CPUDriver IS a nestable Driver node");
    static_assert(std::is_base_of_v<cpu::Driver, InnerLoop>,
                  "child is a nestable Driver node");
    SUCCEED();
}

TEST(DriverNesting, DefaultBudgetWhenUnspecified) {
    cpu::Frame frame = two_atoms();

    InnerLoop inner; // default_budget() == 10
    ConvergeAt inner_comp{/*at=*/999};
    inner.on(ATV::minimizer::Phase::convergence_check).add_ref(inner_comp);

    OuterLoop outer;
    // budget = -1 → child.default_budget() == 10
    outer.on(ATV::md::Phase::end_of_step).add_ref(inner);
    outer.build();
    outer.run(frame, /*budget=*/2);

    // 2 outer × 10 inner default budget
    EXPECT_EQ(inner_comp.calls, 20);
}
