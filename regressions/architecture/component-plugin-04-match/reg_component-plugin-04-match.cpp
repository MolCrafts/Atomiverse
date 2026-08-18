/// @file reg_component-plugin-04-match.cpp
/// @brief Probe-only energy Component (not on the old kind list) builds Evaluate.

#include <cstdio>
#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/evaluate/evaluate.h"

namespace {

class ProbeOnlyEnergy : public ATV::cpu::Component {
  public:
    std::string name() const override { return "probe_only_energy"; }
    bool provides_energy() const override { return true; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

} // namespace

int main() {
    ATV::cpu::Evaluate eval;
    ProbeOnlyEnergy energy;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(energy);
    eval.build();
    std::printf("ok: probe-only energy source accepted by Evaluate::build\n");
    return 0;
}
