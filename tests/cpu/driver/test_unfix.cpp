/// @file tests/cpu/driver/test_unfix.cpp
/// @brief Single-method gates for Driver::unfix.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/observer.h"
#include "atomiverse/cpu/component/recorder.h"
#include "atomiverse/cpu/dynamics/md.h"

#include "cpu/driver_test_helpers.h"

using atv_test::make_trivial_frame;
using atv_test::StubIntegrator;
using atv_test::StubProbeForce;

namespace {

class CountedObs : public ATV::cpu::Observer {
  public:
    std::string name() const override { return "counted_obs"; }
    void build(const ATV::cpu::BuildContext&) override { ++builds; }
    void destroy(const ATV::cpu::DestroyContext&) override { ++destroys; }
    void launch(const ATV::cpu::LaunchContext&) override { ++launches; }
    int builds = 0;
    int destroys = 0;
    int launches = 0;
};

class FakeRecorder : public ATV::cpu::Recorder {
  public:
    std::string name() const override { return "fake_rec"; }
    void launch(const ATV::cpu::LaunchContext&) override { buf += "rec-1\n"; }
    void destroy(const ATV::cpu::DestroyContext&) override {}
    std::string buf;
    int retract_count = 0;
};

class LaunchUnfixer : public ATV::cpu::Observer {
  public:
    ATV::cpu::md::MD* host = nullptr;
    std::string name() const override { return "launch_unfixer"; }
    void launch(const ATV::cpu::LaunchContext&) override {
        if (host)
            host->unfix("counted_obs");
    }
};

} // namespace

TEST(TestDriverUnfix, UnfixObserverRemovesAndDestroys) {
    ATV::cpu::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    CountedObs obs;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    md.on(ATV::md::Phase::end_of_step).add_ref(obs);
    md.build();
    auto frame = make_trivial_frame();
    md.run(frame, 1);
    EXPECT_EQ(obs.launches, 1);
    md.unfix("counted_obs");
    EXPECT_EQ(obs.destroys, 1);
    EXPECT_FALSE(md.is_mounted(ATV::PluginIdentity{&obs}));
    md.run(frame, 1);
    EXPECT_EQ(obs.launches, 1);
}

TEST(TestDriverUnfix, UnknownNameThrowsHowTo) {
    ATV::cpu::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    md.build();
    try {
        md.unfix("nope");
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("no plugin named"),
                  std::string::npos)
            << e.what();
    }
}

TEST(TestDriverUnfix, LastForceSourceThrowsAfterUnmount) {
    ATV::cpu::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    md.build();
    try {
        md.unfix("stub_probe_force");
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("missing required"),
                  std::string::npos)
            << e.what();
        EXPECT_TRUE(
            std::string(e.what()).find("force source") != std::string::npos
            || std::string(e.what()).find("set_evaluator") != std::string::npos
            || std::string(e.what()).find("on(") != std::string::npos)
            << e.what();
    }
    EXPECT_FALSE(md.is_mounted(ATV::PluginIdentity{&force}));
}

TEST(TestDriverUnfix, RecorderKeepsWrittenBytes) {
    ATV::cpu::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    FakeRecorder rec;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    md.on(ATV::md::Phase::end_of_step).add_ref(rec);
    md.build();
    auto frame = make_trivial_frame();
    md.run(frame, 1);
    md.unfix("fake_rec");
    EXPECT_EQ(rec.buf, "rec-1\n");
    EXPECT_EQ(rec.retract_count, 0);
}

TEST(TestDriverUnfix, UnfixFromLaunchThrows) {
    ATV::cpu::md::MD md;
    StubIntegrator integ;
    StubProbeForce force;
    CountedObs obs;
    LaunchUnfixer unfixer;
    unfixer.host = &md;
    md.on(ATV::md::Phase::initial_integrate).add_ref(integ);
    md.on(ATV::md::Phase::force).add_ref(force);
    md.on(ATV::md::Phase::end_of_step).add_ref(obs);
    md.on(ATV::md::Phase::end_of_step).add_ref(unfixer);
    md.build();
    auto frame = make_trivial_frame();
    EXPECT_THROW(md.run(frame, 1), std::runtime_error);
}
