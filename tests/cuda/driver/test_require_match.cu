/// @file tests/cuda/driver/test_require_match.cu
/// @brief Single-method gates for cuda::Driver::require_* / probe-only energy.
///
/// `build()` only validates the component set and compiles the Plan — no
/// device work.

#include <gtest/gtest.h>

#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/hf.cuh"
#include "atomiverse/cuda/component/integrator.cuh"
#include "atomiverse/cuda/component/optimizer.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/minimize/minimize.cuh"

namespace {

class StubIntegrator : public ATV::cuda::Integrator {
  public:
    std::string name() const override { return "stub_integrator"; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

class StubOptimizer : public ATV::cuda::Optimizer {
  public:
    std::string name() const override { return "stub_optimizer"; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

class StubBareInfra : public ATV::cuda::Component {
  public:
    std::string name() const override { return "stub_bare_infra"; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

class StubProbeEnergy : public ATV::cuda::Component {
  public:
    std::string name() const override { return "stub_probe_energy"; }
    bool provides_energy() const override { return true; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

class StubProbeForce : public ATV::cuda::Component {
  public:
    std::string name() const override { return "stub_probe_force"; }
    bool provides_force() const override { return true; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

class StubHFMethod : public ATV::cuda::Component, public ATV::cuda::HF {
  public:
    std::string name() const override { return "stub_hf_method"; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

} // namespace

TEST(TestCudaRequireMatch, RequireEnergyOrForceAcceptsProbeOnlyComponent) {
    ATV::cuda::Evaluate eval;
    StubProbeEnergy energy;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(energy);
    EXPECT_NO_THROW(eval.build());
}

TEST(TestCudaRequireMatch, RequireEnergyOrForceAcceptsHFTag) {
    ATV::cuda::Evaluate eval;
    StubHFMethod hf;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(hf);
    EXPECT_NO_THROW(eval.build());
}

TEST(TestCudaRequireMatch, RequireEnergyOrForceRejectsBareComponent) {
    ATV::cuda::Evaluate eval;
    StubBareInfra bare;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(bare);
    try {
        eval.build();
        FAIL() << "expected missing energy or force source";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("energy or force source"),
                  std::string::npos)
            << e.what();
    }
}

TEST(TestCudaRequireMatch, RequireForceAcceptsProbeOnlyForce) {
    ATV::cuda::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    EXPECT_NO_THROW(md.build());
}

TEST(TestCudaRequireMatch, RequireForceRejectsEnergyOnlyProbe) {
    ATV::cuda::md::MD md;
    StubIntegrator integ;
    StubProbeEnergy energy;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(energy);
    try {
        md.build();
        FAIL() << "expected missing force source";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("force source"), std::string::npos)
            << e.what();
    }
}

TEST(TestCudaRequireMatch, RequireIntegratorThrows) {
    ATV::cuda::md::MD md;
    StubProbeForce force;
    md.on(ATV::md::Phase::force).add_ref(force);
    try {
        md.build();
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Integrator"), std::string::npos)
            << e.what();
    }
}

TEST(TestCudaRequireMatch, RequireOptimizerThrows) {
    ATV::cuda::minimizer::Minimize opt;
    StubProbeEnergy energy;
    opt.on(ATV::minimizer::Phase::force).add_ref(energy);
    try {
        opt.build();
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Optimizer"), std::string::npos)
            << e.what();
    }
}
