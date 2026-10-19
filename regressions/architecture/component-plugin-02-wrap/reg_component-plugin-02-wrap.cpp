/// @file reg_component-plugin-02-wrap.cpp
/// @brief Public-API gate: on(phase).add_ref nests a driver as one Component.
///
/// The inner loop is a physical (Minimize-phase) driver, so it mounts at the
/// parent's `end_of_step`: a physical child may not replace geometry or force
/// accumulators inside the parent's force/integration pipeline.
///
/// Hard-coded architecture literals (oracle: harness).

#include <cstdio>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/driver/driver_node.h"
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
    InnerLoop inner;
    Counter counter;
    inner.on(ATV::minimizer::Phase::force).add_ref(counter);
    OuterLoop outer;
    outer.on(ATV::md::Phase::end_of_step)
        .add_ref(inner, /*order=*/0, /*budget=*/3);
    outer.build();

    const auto& ops =
        outer.plan().slots[ATV::slot_of(ATV::md::Phase::end_of_step)].ops;
    if (ops.empty() || ops.front().component == nullptr
        || dynamic_cast<cpu::DriverNode*>(ops.front().component) != nullptr) {
        std::printf(
            "FAIL: nested slot must be a Component and not a DriverNode\n");
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
    std::printf("ok: wrap launches=3 slot_is_driver=0\n");
    return 0;
}
