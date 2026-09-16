/// @file reg_component-plugin-05-live.cpp
/// @brief Live table survives compile(); second compile matches pointers.

#include <cstdio>
#include <string>

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

int main() {
    ATV::cpu::Evaluate eval;
    LiveProbe a;
    LiveProbe b;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(a);
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(b);
    eval.compile();
    if (eval.mounted().size() != 5) {
        std::printf("FAIL: mounted_count=%zu want=5\n", eval.mounted().size());
        return 1;
    }
    const auto first = eval.plan();
    eval.compile();
    const auto& second = eval.plan();
    for (std::size_t s = 0; s < first.slots.size(); ++s) {
        if (first.slots[s].ops.size() != second.slots[s].ops.size()) {
            std::printf("FAIL: slot size drift\n");
            return 1;
        }
        for (std::size_t o = 0; o < first.slots[s].ops.size(); ++o) {
            if (first.slots[s].ops[o].component
                != second.slots[s].ops[o].component) {
                std::printf("FAIL: pointer mismatch\n");
                return 1;
            }
        }
    }
    std::printf("ok: mounted_count=5 second_compile_equal=1\n");
    return 0;
}
