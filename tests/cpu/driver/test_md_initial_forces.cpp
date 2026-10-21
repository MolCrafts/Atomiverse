/// @file tests/cpu/driver/test_md_initial_forces.cpp
/// @brief Runtime::setup_md_forces dispatches only the initial force pass.
#include <gtest/gtest.h>

#include "atomiverse/core/plan.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/runtime.h"

#include <string>
#include <vector>

using namespace ATV;

namespace {

class PhaseProbe : public cpu::Component {
  public:
    explicit PhaseProbe(std::vector<std::string>& events,
                        const std::string& label)
        : events_(events), label_(label) {}
    std::string name() const override { return label_; }
    void launch(const cpu::LaunchContext& ctx) override {
        events_.push_back(label_);
        EXPECT_EQ(ctx.step, 0);
        if (ctx.at(md::Phase::force)) {
            ctx.frame.span(kAtom_fx)[0] = 3.0;
            ctx.frame.meta.get(kEnergy) = 7.0;
            ctx.frame.meta.get(kStress)[0] = 11.0;
        }
    }

  private:
    std::vector<std::string>& events_;
    std::string label_;
};

class InitialComm : public cpu::Comm {
  public:
    explicit InitialComm(std::vector<std::string>& events) : events_(events) {}
    void setup(const cpu::LaunchContext&) override {}
    void forward_comm(const cpu::LaunchContext&) override {
        events_.push_back("forward");
    }
    void exchange(const cpu::LaunchContext&) override {
        events_.push_back("exchange");
    }
    void borders(const cpu::LaunchContext&) override {
        events_.push_back("borders");
    }
    void reverse_comm(const cpu::LaunchContext&) override {
        events_.push_back("reverse");
    }
    void forward_comm(cpu::Packer&) override {}
    void set_ghost_cutoff(double) override {}

  private:
    std::vector<std::string>& events_;
};

} // namespace

TEST(MdInitialForces,
     SeedsForceAndScalarsWithoutIntegrationOrCommunicationSetup) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(20.0);
    frame.span(kAtom_x)[0] = 4.0;
    frame.span(kAtom_y)[0] = 4.0;
    frame.span(kAtom_z)[0] = 4.0;
    frame.span(kAtom_mass)[0] = 2.0;
    frame.span(kAtom_vx)[0] = 0.25;
    frame.span(kAtom_vy)[0] = 0.0;
    frame.span(kAtom_vz)[0] = 0.0;
    frame.meta.get(kKinetic) = -123.0;

    std::vector<std::string> events;
    PhaseProbe initial(events, "initial"), pre_exchange(events, "pre_exchange"),
        neighbor(events, "neighbor"), pre_force(events, "pre_force"),
        force(events, "force"), post_force(events, "post_force"),
        final(events, "final"), end(events, "end");
    PlanBuilder<cpu::Component> builder;
    auto add = [&](md::Phase phase, cpu::Component& c) {
        builder.add({slot_of(phase), 0, &c});
    };
    add(md::Phase::initial_integrate, initial);
    add(md::Phase::pre_exchange, pre_exchange);
    add(md::Phase::neighbor, neighbor);
    add(md::Phase::pre_force, pre_force);
    add(md::Phase::force, force);
    add(md::Phase::post_force, post_force);
    add(md::Phase::final_integrate, final);
    add(md::Phase::end_of_step, end);
    auto plan = builder.build(slot_count<md::Phase>());
    plan.kind = PhaseKind::MD;

    InitialComm comm(events);
    cpu::Runtime runtime;
    runtime.init();
    runtime.load(plan);
    runtime.set_frame(frame);
    runtime.context().md.comm = &comm;
    runtime.build_components();
    events.clear();

    runtime.setup_md_forces();

    EXPECT_EQ(events, (std::vector<std::string>{"pre_force", "force", "reverse",
                                                "post_force"}));
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 4.0);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_vx)[0], 0.25);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 3.0);
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 7.0);
    EXPECT_DOUBLE_EQ(frame.meta.get(kStress)[0], 11.0);
    EXPECT_DOUBLE_EQ(frame.meta.get(kKinetic),
                     0.5 * units::mass_v2_to_energy * 2.0 * 0.25 * 0.25);
    runtime.destroy_components();
}
