/// @file reg_component-plugin-06-unmount.cpp
/// @brief Mid-run unfix of last evaluator throws how-to; remount continues.

#include <cstdio>
#include <stdexcept>
#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

namespace {

class FakeEval : public ATV::cpu::Potential {
  public:
    std::string name() const override { return "fake_eval"; }
    void launch(const ATV::cpu::LaunchContext&) override { ++launches; }
    int launches = 0;
};

class FakeInteg : public ATV::cpu::Integrator {
  public:
    std::string name() const override { return "fake_integ"; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

} // namespace

int main() {
    ATV::cpu::md::MD md;
    FakeInteg integ;
    FakeEval ev;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(ev);
    md.build();

    ATV::cpu::Frame frame(1);
    frame.span(ATV::kAtom_x)[0] = 0;
    frame.span(ATV::kAtom_y)[0] = 0;
    frame.span(ATV::kAtom_z)[0] = 0;
    frame.span(ATV::kAtom_mass)[0] = 1;
    frame.span(ATV::kAtom_type)[0] = 1;
    frame.span(ATV::kAtom_tag)[0] = 1;
    frame.box = ATV::cpu::Box::cubic(10.0);

    md.run(frame, 2);
    if (ev.launches != 2) {
        std::printf("FAIL: launches=%d want=2\n", ev.launches);
        return 1;
    }
    try {
        md.unfix("fake_eval");
        std::printf("FAIL: unfix last evaluator should throw\n");
        return 1;
    } catch (const std::runtime_error& e) {
        const std::string w = e.what();
        if (w.find("missing required") == std::string::npos
            || (w.find("force source") == std::string::npos
                && w.find("set_evaluator") == std::string::npos
                && w.find("on(") == std::string::npos)) {
            std::printf("FAIL: how-to missing: %s\n", e.what());
            return 1;
        }
    }
    md.on(ATV::md::Phase::force).add_ref(ev);
    md.run(frame, 2);
    if (ev.launches != 4) {
        std::printf("FAIL: launches=%d want=4\n", ev.launches);
        return 1;
    }
    std::printf("ok: unfix-remount launches=4\n");
    return 0;
}
