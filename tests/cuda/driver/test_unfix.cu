/// @file tests/cuda/driver/test_unfix.cu
/// @brief Single-method gates for cuda::Driver::unfix (table + closure).
///
/// These tests stop at `build()` / `unfix` / remount — no `run()`, no device.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "atomiverse/core/driver_template.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/integrator.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"

namespace {

class StubIntegrator : public ATV::cuda::Integrator {
  public:
    std::string name() const override { return "stub_integrator"; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

class StubProbeForce : public ATV::cuda::Component {
  public:
    std::string name() const override { return "stub_probe_force"; }
    bool provides_force() const override { return true; }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

class CountedObs : public ATV::cuda::Component {
  public:
    std::string name() const override { return "counted_obs"; }
    void launch(const ATV::cuda::LaunchContext&) override { ++launches; }
    int launches = 0;
};

} // namespace

TEST(TestCudaDriverUnfix, UnfixObserverDropsFromLiveTable) {
    ATV::cuda::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    CountedObs obs;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    md.on(ATV::md::Phase::end_of_step).add_ref(obs);
    md.build();
    EXPECT_TRUE(md.is_mounted(ATV::PluginIdentity{&obs}));
    md.unfix("counted_obs");
    EXPECT_FALSE(md.is_mounted(ATV::PluginIdentity{&obs}));
}

TEST(TestCudaDriverUnfix, UnknownNameThrowsHowTo) {
    ATV::cuda::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    md.build();
    try {
        md.unfix("no_such_plugin");
        FAIL() << "expected unknown-name how-to";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("no plugin named"),
                  std::string::npos)
            << e.what();
        EXPECT_NE(std::string(e.what()).find("mounted:"), std::string::npos)
            << e.what();
    }
}

TEST(TestCudaDriverUnfix, UnfixLastForceSourceThrows) {
    ATV::cuda::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    md.build();
    try {
        md.unfix("stub_probe_force");
        FAIL() << "expected missing force source";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("force source"), std::string::npos)
            << e.what();
    }
    EXPECT_FALSE(md.is_mounted(ATV::PluginIdentity{&force}));
}

TEST(TestCudaDriverUnfix, RemountAfterUnfixRebuilds) {
    ATV::cuda::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    CountedObs obs;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    md.on(ATV::md::Phase::end_of_step).add_ref(obs);
    md.build();
    md.unfix("counted_obs");
    md.on(ATV::md::Phase::end_of_step).add_ref(obs);
    EXPECT_NO_THROW(md.build());
    EXPECT_TRUE(md.is_mounted(ATV::PluginIdentity{&obs}));
}
