#include <gtest/gtest.h>

#include "atomiverse/core/driver_template_impl.h"

namespace {
struct ProbeDevice {};
struct ProbeFrame {
    int steps = 0;
};
struct ProbeTimer {};
struct ProbeNode {};
struct ProbeComponent {
    virtual ~ProbeComponent() = default;
    virtual std::string name() const { return "probe"; }
    ATV::Scheduling scheduling() const { return ATV::Scheduling::blocking; }
    int frequency() const { return 0; }
};
struct ProbeBackend {
    void shutdown() {}
};
} // namespace

namespace ATV {
template <>
struct backend_traits<ProbeDevice> {
    using component_type = ProbeComponent;
    using backend_type = ProbeBackend;
    using driver_node_type = ProbeNode;
    using frame_type = ProbeFrame;
    using timer_type = ProbeTimer;
    static constexpr bool has_device_timing = false;
    static void run(backend_type&, frame_type& frame,
                    const Plan<component_type>& plan, timer_type*, int steps) {
        EXPECT_EQ(plan.slots[slot_of(evaluate::Phase::evaluate)].ops.size(),
                  1u);
        frame.steps += steps;
    }
};
} // namespace ATV

TEST(DriverContract, BackendOwnsFrameAndExecution) {
    struct ProbeDriver : ATV::Driver<ATV::evaluate::Phase, ProbeDevice> {
        using ATV::Driver<ATV::evaluate::Phase, ProbeDevice>::run;
    } driver;
    driver.on(ATV::evaluate::Phase::evaluate).add(ProbeComponent{});
    driver.build();
    ProbeFrame frame;
    driver.run(frame, 3);
    EXPECT_EQ(frame.steps, 3);
    driver.enable_timing();
    EXPECT_TRUE(driver.summary().empty());
}
