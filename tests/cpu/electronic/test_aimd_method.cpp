#include <gtest/gtest.h>

#include "atomiverse/cpu/driver/backend_traits.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/electronic/aimd_method.h"

namespace {
// A final, move-only method can be composed but cannot be subclassed.
struct MethodProbe final {
    MethodProbe() = default;
    MethodProbe(const MethodProbe&) = delete;
    MethodProbe(MethodProbe&&) = default;
    bool cpmd = false, cpmd_thermostat = false, xl_bomd = false;
    double cpmd_mu = 0, cpmd_ekin_target_ha = 0, cpmd_tau_e = 0, dt = 0;
    int xl_bomd_k = 0, builds = 0, launches = 0, destroys = 0;
    bool provides_energy() const { return true; }
    bool provides_force() const { return true; }
    bool requires_full_frame() const { return true; }
    bool requires_tag_index() const { return false; }
    int default_budget() const { return 37; }
    bool timestep_required() const { return cpmd; }
    void set_timestep(double value) { dt = value; }
    void build_on(ATV::cpu::Runtime&) { ++builds; }
    int run_nested_erased(ATV::cpu::Context&, ATV::cpu::Frame&, int budget) {
        ++launches;
        return budget;
    }
    void destroy_on(ATV::cpu::Runtime&) { ++destroys; }
};
} // namespace

TEST(AimdMethod, ComposesOneMethodThroughNodeAndClockProtocols) {
    ATV::cpu::electronic::CarParrinelloMethod<MethodProbe> cp{MethodProbe{},
                                                              1000};
    ATV::cpu::DriverNode& node = cp;
    ASSERT_NE(node.timestep_consumer(), nullptr);
    node.timestep_consumer()->set_timestep(0.25);
    EXPECT_TRUE(node.timestep_consumer()->timestep_required());
    EXPECT_EQ(node.default_budget(), 37);
    ATV::cpu::Runtime runtime;
    ATV::cpu::Context context;
    ATV::cpu::Frame frame;
    node.build_on(runtime);
    EXPECT_EQ(node.run_nested_erased(context, frame, 3), 3);
    node.destroy_on(runtime);
    EXPECT_DOUBLE_EQ(cp.method().cpmd_mu, 1000);
    EXPECT_DOUBLE_EQ(cp.method().dt, 0.25);
    EXPECT_EQ(cp.method().builds, 1);
    EXPECT_EQ(cp.method().launches, 1);
    EXPECT_EQ(cp.method().destroys, 1);
}

TEST(AimdMethod, XlPolicyStaysOnTheOwnedMethod) {
    ATV::cpu::electronic::XlBomdMethod<MethodProbe> xl{MethodProbe{}, 7};
    EXPECT_TRUE(xl.method().xl_bomd);
    EXPECT_EQ(xl.method().xl_bomd_k, 7);
    EXPECT_FALSE(xl.timestep_required());
}

TEST(AimdMethod, RejectsInvalidElectronicMass) {
    EXPECT_THROW((ATV::cpu::electronic::CarParrinelloMethod<MethodProbe>{
                     MethodProbe{}, -1}),
                 std::invalid_argument);
}

namespace {
struct ThrowingNode : ATV::cpu::DriverNode {
    void build_on(ATV::cpu::Runtime&) override {}
    void destroy_on(ATV::cpu::Runtime&) override {}
    int run_nested_erased(ATV::cpu::Context&, ATV::cpu::Frame&, int) override {
        throw std::runtime_error("nested failure");
    }
};
} // namespace

TEST(AimdMethod, NestedFailureRestoresParentForceRequest) {
    ThrowingNode node;
    auto slot = ATV::backend_traits<ATV::cpu::device_tag>::nested_slot(
        std::unique_ptr<ThrowingNode>{}, &node, 1, true, true);
    ATV::cpu::Context context;
    ATV::cpu::Frame frame;
    context.exec.forces_requested = true;
    const ATV::cpu::LaunchContext launch{
        context, frame, ATV::slot_of(ATV::evaluate::Phase::evaluate), 0, false};
    EXPECT_THROW(slot->launch(launch), std::runtime_error);
    EXPECT_TRUE(context.exec.forces_requested);
}

namespace {
struct ClockNode : ThrowingNode, ATV::cpu::TimestepConsumer {
    double dt = 0;
    bool timestep_required() const override { return true; }
    void set_timestep(double value) override { dt = value; }
};
} // namespace

TEST(AimdMethod, NestedConcreteNodeDiscoversInheritedClockProtocol) {
    ClockNode node;
    auto slot = ATV::backend_traits<ATV::cpu::device_tag>::nested_slot(
        std::unique_ptr<ClockNode>{}, &node, 1, false, false);
    auto* clock = slot->as_timestep_consumer();
    ASSERT_NE(clock, nullptr);
    EXPECT_TRUE(clock->timestep_required());
    clock->set_timestep(0.125);
    EXPECT_DOUBLE_EQ(node.dt, 0.125);
}
