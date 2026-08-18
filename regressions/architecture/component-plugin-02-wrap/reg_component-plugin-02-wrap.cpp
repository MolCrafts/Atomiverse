/// @file reg_component-plugin-02-wrap.cpp
/// @brief Public-API gate: NestedLoopAdapter hides a DriverNode as Component.
///
/// Hard-coded architecture literals (oracle: harness).

#include <cstdio>
#include <string>
#include <type_traits>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
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
        }
        teardown_pass();
        return iters;
    }

  protected:
    void validate_components() const override {}
};

class InnerLoop : public cpu::Driver<ATV::minimizer::Phase> {
  protected:
    void validate_components() const override {}
};

class Counter : public cpu::Component {
  public:
    std::string name() const override { return "counter"; }
    void launch(const cpu::LaunchContext&) override { ++launches; }
    int launches = 0;
};

} // namespace

int main() {
    if (std::is_base_of_v<cpu::DriverNode, cpu::NestedLoopAdapter>) {
        std::printf("FAIL: adapter must not be a DriverNode\n");
        return 1;
    }

    InnerLoop inner;
    Counter counter;
    inner.on(ATV::minimizer::Phase::force).add_ref(counter);
    cpu::NestedLoopAdapter adapter(inner, 3);
    OuterLoop outer;
    outer.on(ATV::md::Phase::force).add_ref(adapter);
    outer.build();

    const auto& ops =
        outer.plan().slots[ATV::slot_of(ATV::md::Phase::force)].ops;
    if (ops.empty() || ops.front().component == nullptr) {
        std::printf("FAIL: outer adapter op must be a Component\n");
        return 1;
    }

    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_x)[i] = 0.5 * i;
        frame.span(ATV::kAtom_y)[i] = 0.0;
        frame.span(ATV::kAtom_z)[i] = 0.0;
        frame.span(ATV::kAtom_mass)[i] = 1.0;
    }
    outer.run(frame, 1);
    if (counter.launches != 3) {
        std::printf("FAIL: launches=%d want=3\n", counter.launches);
        return 1;
    }
    std::printf("ok: wrap launches=3 adapter_is_driver=0\n");
    return 0;
}
