/// @file tests/cpu/driver_test_helpers.h
/// @brief Shared component stubs for CPU driver validation tests.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/optimizer.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/frame/frame.h"

namespace atv_test {

class NoOpForce : public ATV::cpu::Potential {
  public:
    std::string name() const override { return "noop_force"; }
    std::vector<ATV::DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::force, 0)};
    }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

class StubIntegrator : public ATV::cpu::Integrator {
  public:
    int step_count = 0;
    std::string name() const override { return "stub_integrator"; }
    void launch(const ATV::cpu::LaunchContext&) override { ++step_count; }
};

class StubForceEvaluator : public ATV::cpu::Potential {
  public:
    std::string name() const override { return "stub_force"; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

class StubEnergyEvaluator : public ATV::cpu::Potential {
  public:
    std::string name() const override { return "stub_energy"; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

class StubOptimizer : public ATV::cpu::Optimizer {
  public:
    int step_count = 0;
    std::string name() const override { return "stub_optimizer"; }
    void launch(const ATV::cpu::LaunchContext&) override { ++step_count; }
};

class StubBareInfra : public ATV::cpu::Component {
  public:
    std::string name() const override { return "stub_bare_infra"; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

/// Probe-only energy source: not on the old kind list.
class StubProbeEnergy : public ATV::cpu::Component {
  public:
    std::string name() const override { return "stub_probe_energy"; }
    bool provides_energy() const override { return true; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

/// Probe-only force source: not on the old kind list.
class StubProbeForce : public ATV::cpu::Component {
  public:
    std::string name() const override { return "stub_probe_force"; }
    bool provides_force() const override { return true; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

inline ATV::cpu::Frame make_trivial_frame(int n = 1) {
    ATV::cpu::Frame f(n);
    for (int i = 0; i < n; ++i) {
        f.span(ATV::kAtom_x)[i] = 0;
        f.span(ATV::kAtom_y)[i] = 0;
        f.span(ATV::kAtom_z)[i] = 0;
        f.span(ATV::kAtom_mass)[i] = 1.0;
        f.span(ATV::kAtom_type)[i] = 0;
        f.span(ATV::kAtom_tag)[i] = i + 1;
    }
    f.box = ATV::cpu::Box::cubic(10.0);
    return f;
}

} // namespace atv_test
