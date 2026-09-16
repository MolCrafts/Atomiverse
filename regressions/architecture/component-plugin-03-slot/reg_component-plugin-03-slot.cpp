/// @file reg_component-plugin-03-slot.cpp
/// @brief Public-API gate: on(phase).add_ref(inner) occupies a Component slot.
///
/// Hard-coded architecture literals (oracle: harness).

#include <cstdio>
#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/driver/nested_loop_adapter.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

class OuterLoop : public cpu::Driver<ATV::md::Phase> {
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

class InnerLoop : public cpu::Driver<ATV::minimizer::Phase> {
  public:
    int default_budget() const override { return 10; }

  protected:
    void validate_components() const override {}
};

class ConvergeAt : public cpu::Component {
  public:
    explicit ConvergeAt(int at) : at_(at) {}
    std::string name() const override { return "converge_at"; }
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
    int calls = 0;

  private:
    int at_;
};

class PublishForce : public cpu::Component {
  public:
    std::string name() const override { return "publish_force"; }
    void launch(const cpu::LaunchContext& ctx) override {
        if (auto* m = ctx.runtime.loop.metrics) {
            m->max_force = 1e3;
            m->rms_force = 1e3;
            m->energy_change = 1e3;
            m->has_displacement = false;
        }
    }
};

} // namespace

int main() {
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_x)[i] = 0.5 * i;
        frame.span(ATV::kAtom_y)[i] = 0.0;
        frame.span(ATV::kAtom_z)[i] = 0.0;
        frame.span(ATV::kAtom_mass)[i] = 1.0;
        frame.span(ATV::kAtom_type)[i] = 1;
    }
    frame.box = cpu::Box::cubic(10.0);

    InnerLoop inner;
    ConvergeAt converge{/*at=*/2};
    inner.on(ATV::minimizer::Phase::convergence_check).add_ref(converge);
    inner.add_criterion(cpu::ForceConverged{});

    OuterLoop outer;
    PublishForce big;
    outer.on(ATV::md::Phase::force).add_ref(big);
    outer.on(ATV::md::Phase::end_of_step)
        .add_ref(inner, /*order=*/0, /*budget=*/10);
    outer.add_criterion(cpu::ForceConverged{});
    outer.build();

    const auto& ops =
        outer.plan().slots[ATV::slot_of(ATV::md::Phase::end_of_step)].ops;
    if (ops.empty()
        || dynamic_cast<cpu::NestedLoopAdapter*>(ops.front().component)
               == nullptr) {
        std::printf("FAIL: nested slot is not a NestedLoopAdapter\n");
        return 1;
    }

    const int outer_iters = outer.run(frame, 5);
    if (outer_iters != 5 || outer.converged() || !inner.converged()) {
        std::printf(
            "FAIL: outer_iters=%d outer.converged=%d inner.converged=%d\n",
            outer_iters, outer.converged() ? 1 : 0, inner.converged() ? 1 : 0);
        return 1;
    }
    std::printf("ok: slot=adapter outer_iters=5 inner_converged=1\n");
    return 0;
}
