#include <gtest/gtest.h>

#include "atomiverse/core/driver_template_impl.h"

namespace {
struct ProbeDevice {};
struct ProbeFrame {
    int steps = 0;
    ATV::units::Style style = ATV::units::Style::lj;
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
                    const Plan<component_type>& plan, timer_type*,
                    const units::UnitSystem& units, int steps) {
        frame.style = units.style;
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

TEST(DriverContract, UnitsStyleIsChosenOnceAndPublishedToTheBackend) {
    struct ProbeDriver : ATV::Driver<ATV::evaluate::Phase, ProbeDevice> {
        using ATV::Driver<ATV::evaluate::Phase, ProbeDevice>::run;
    } driver;
    EXPECT_EQ(driver.units().style, ATV::units::Style::lj); // LAMMPS default
    EXPECT_THROW(driver.set_units("kcal"), std::invalid_argument);
    driver.set_units("real");
    EXPECT_STREQ(driver.units().name(), "real");
    EXPECT_DOUBLE_EQ(driver.units().qqr2e, 332.06371);
    driver.on(ATV::evaluate::Phase::evaluate).add(ProbeComponent{});
    driver.build();
    ProbeFrame frame;
    driver.run(frame, 1);
    EXPECT_EQ(frame.style, ATV::units::Style::real);
}
