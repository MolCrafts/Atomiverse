/// @file tests/cpu/driver/test_verlet_comm.cpp
/// @brief MD `Runtime::step` is LAMMPS Verlet::run comm on `ctx.md.comm`.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "atomiverse/core/plan.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

class SpyComm : public cpu::Comm {
  public:
    int n_setup = 0;
    int n_forward = 0;
    int n_reverse = 0;
    int n_exchange = 0;
    int n_borders = 0;
    std::vector<std::string>* log = nullptr;

    void setup(const cpu::LaunchContext&) override { ++n_setup; }
    void forward_comm(const cpu::LaunchContext&) override {
        ++n_forward;
        if (log)
            log->push_back("forward");
    }
    void reverse_comm(const cpu::LaunchContext&) override {
        ++n_reverse;
        if (log)
            log->push_back("reverse");
    }
    void exchange(const cpu::LaunchContext&) override {
        ++n_exchange;
        if (log)
            log->push_back("exchange");
    }
    void borders(const cpu::LaunchContext&) override {
        ++n_borders;
        if (log)
            log->push_back("borders");
    }
    void forward_comm(cpu::Packer&) override {}
    void set_ghost_cutoff(double) override {}
};

class SpyNeighbor : public cpu::Component, public cpu::NeighborProvider {
  public:
    int n_decide = 0;
    int n_launch = 0;
    int delay = 10;
    std::string name() const override { return "spy_neighbor"; }
    bool verlet_engine() const override { return true; }
    NeighborProvider* neighbor_provider() override { return this; }
    const NeighborProvider* as_neighbor_provider() const override {
        return this;
    }
    void forbid_row_reordering() override {}
    bool decide_reneighbor(cpu::Context&, cpu::Frame&) override {
        ++n_decide;
        return n_decide >= delay;
    }
    void launch(const cpu::LaunchContext&) override { ++n_launch; }
};

class EngineStub : public cpu::Component {
  public:
    int n_launch = 0;
    std::string name() const override { return "engine_stub"; }
    bool verlet_engine() const override { return true; }
    void launch(const cpu::LaunchContext&) override { ++n_launch; }
};

class Hook : public cpu::Component {
  public:
    int n_launch = 0;
    std::vector<std::string>* log = nullptr;
    std::string name() const override { return "hook"; }
    void launch(const cpu::LaunchContext&) override {
        ++n_launch;
        if (log)
            log->push_back("hook");
    }
};

cpu::Frame one_atom() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 1.0;
    f.span(kAtom_y)[0] = 1.0;
    f.span(kAtom_z)[0] = 1.0;
    f.span(kAtom_mass)[0] = 1.0;
    f.box = cpu::Box::cubic(10.0);
    return f;
}

template <typename PhaseEnum>
void fill_slots(Plan<cpu::Component>& plan) {
    plan.slots.resize(static_cast<std::size_t>(ATV::slot_count<PhaseEnum>()));
    for (int p = 0; p < ATV::slot_count<PhaseEnum>(); ++p)
        plan.slots[static_cast<std::size_t>(p)].phase = p;
}

template <typename PhaseEnum>
void add_op(Plan<cpu::Component>& plan, PhaseEnum phase, cpu::Component& c) {
    Operation<cpu::Component> op;
    op.component = &c;
    plan.slots[static_cast<std::size_t>(ATV::slot_of(phase))].ops.push_back(op);
}

Plan<cpu::Component> md_plan(cpu::Component& neigh, cpu::Component& hook,
                             cpu::Component* engine = nullptr) {
    Plan<cpu::Component> plan;
    plan.kind = PhaseKind::MD;
    fill_slots<md::Phase>(plan);
    add_op(plan, md::Phase::neighbor, neigh);
    add_op(plan, md::Phase::end_of_step, hook);
    if (engine) {
        add_op(plan, md::Phase::pre_exchange, *engine);
        add_op(plan, md::Phase::pre_neighbor, *engine);
        add_op(plan, md::Phase::post_force, *engine);
    }
    return plan;
}

void bind_spy(cpu::Runtime& rt, SpyComm& spy) { rt.context().md.comm = &spy; }

} // namespace

TEST(TestVerletComm, NflagZeroIsForwardCommOnly) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neigh;
    EngineStub engine;
    Hook hook;
    auto plan = md_plan(neigh, hook, &engine);

    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();

    for (int s = 0; s < 9; ++s)
        rt.step(s);

    EXPECT_EQ(neigh.n_decide, 9);
    EXPECT_EQ(neigh.n_launch, 0);
    EXPECT_EQ(comm.n_forward, 9);
    EXPECT_EQ(comm.n_borders, 0);
    EXPECT_EQ(comm.n_exchange, 0);
    EXPECT_EQ(comm.n_reverse, 9);
    EXPECT_EQ(comm.n_forward + comm.n_borders, 9);
    EXPECT_EQ(engine.n_launch, 0);
    EXPECT_EQ(hook.n_launch, 9);
}

TEST(TestVerletComm, NflagOneIsExchangeBordersBuild) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neigh;
    EngineStub engine;
    Hook hook;
    auto plan = md_plan(neigh, hook, &engine);

    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();

    for (int s = 0; s < 10; ++s)
        rt.step(s);

    EXPECT_EQ(neigh.n_decide, 10);
    EXPECT_EQ(neigh.n_launch, 1);
    EXPECT_EQ(comm.n_forward, 9);
    EXPECT_EQ(comm.n_borders, 1);
    EXPECT_EQ(comm.n_exchange, 1);
    EXPECT_EQ(comm.n_reverse, 10);
    EXPECT_EQ(comm.n_forward + comm.n_borders, 10);
    EXPECT_EQ(engine.n_launch, 0);
    EXPECT_EQ(hook.n_launch, 10);
}

TEST(TestVerletComm, ReverseCommEveryStepNotGatedOnNflag) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neigh;
    Hook hook;
    auto plan = md_plan(neigh, hook);

    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();

    constexpr int n_steps = 10;
    for (int s = 0; s < n_steps; ++s)
        rt.step(s);

    EXPECT_EQ(comm.n_reverse, n_steps);
    EXPECT_EQ(comm.n_forward + comm.n_borders, n_steps);
}

TEST(TestVerletComm, SerialSetupDoesNotCallComm) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neigh;
    Hook hook;
    auto plan = md_plan(neigh, hook);

    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();

    EXPECT_EQ(rt.context().mpi_size, 1);
    EXPECT_EQ(comm.n_exchange, 0);
    EXPECT_EQ(comm.n_borders, 0);
    EXPECT_EQ(comm.n_forward, 0);
    EXPECT_EQ(neigh.n_launch, 0);
}

TEST(TestVerletComm, MpiSetupUsesCommThenNeighbor) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neigh;
    Hook hook;
    auto plan = md_plan(neigh, hook);

    cpu::Runtime rt;
    rt.init();
    rt.context().mpi_size = 2;
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();

    EXPECT_EQ(comm.n_exchange, 1);
    EXPECT_EQ(comm.n_borders, 1);
    EXPECT_EQ(neigh.n_launch, 1);
    EXPECT_EQ(comm.n_forward, 0);
    EXPECT_EQ(comm.n_reverse, 0);
}

TEST(TestVerletComm, EvaluateWrapsCommAroundForceSlot) {
    auto frame = one_atom();
    SpyComm comm;
    EngineStub engine;
    Hook hook;
    std::vector<std::string> log;
    comm.log = &log;
    hook.log = &log;

    Plan<cpu::Component> plan;
    plan.kind = PhaseKind::Evaluate;
    fill_slots<evaluate::Phase>(plan);
    add_op(plan, evaluate::Phase::evaluate, engine);
    add_op(plan, evaluate::Phase::evaluate, hook);

    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    rt.step(0);

    EXPECT_EQ(engine.n_launch, 0);
    EXPECT_EQ(hook.n_launch, 1);
    EXPECT_EQ(comm.n_reverse, 1);
    EXPECT_TRUE(
        (comm.n_forward == 1 && comm.n_exchange == 0 && comm.n_borders == 0)
        || (comm.n_forward == 0 && comm.n_exchange == 1
            && comm.n_borders == 1));
    ASSERT_GE(log.size(), 2u);
    EXPECT_EQ(log.back(), "reverse");
    bool hook_before_reverse = false;
    for (std::size_t i = 0; i + 1 < log.size(); ++i) {
        if (log[i] == "hook" && log[i + 1] == "reverse")
            hook_before_reverse = true;
    }
    EXPECT_TRUE(hook_before_reverse);
    EXPECT_TRUE(log.front() == "forward" || log.front() == "exchange");
}

TEST(TestVerletComm, MinimizeWrapsCommAroundForceSlot) {
    auto frame = one_atom();
    SpyComm comm;
    EngineStub engine;
    Hook hook;
    std::vector<std::string> log;
    comm.log = &log;
    hook.log = &log;

    Plan<cpu::Component> plan;
    plan.kind = PhaseKind::Minimizer;
    fill_slots<minimizer::Phase>(plan);
    add_op(plan, minimizer::Phase::force, engine);
    add_op(plan, minimizer::Phase::force, hook);

    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    rt.step(0);

    EXPECT_EQ(engine.n_launch, 0);
    EXPECT_EQ(hook.n_launch, 1);
    EXPECT_EQ(comm.n_reverse, 1);
    EXPECT_TRUE(
        (comm.n_forward == 1 && comm.n_exchange == 0 && comm.n_borders == 0)
        || (comm.n_forward == 0 && comm.n_exchange == 1
            && comm.n_borders == 1));
    ASSERT_GE(log.size(), 2u);
    EXPECT_EQ(log.back(), "reverse");
    bool hook_before_reverse = false;
    for (std::size_t i = 0; i + 1 < log.size(); ++i) {
        if (log[i] == "hook" && log[i + 1] == "reverse")
            hook_before_reverse = true;
    }
    EXPECT_TRUE(hook_before_reverse);
}

TEST(TestVerletComm, MinimizeRunStageWrapsComm) {
    auto frame = one_atom();
    SpyComm comm;
    EngineStub engine;
    Hook hook;
    std::vector<std::string> log;
    comm.log = &log;
    hook.log = &log;

    Plan<cpu::Component> plan;
    plan.kind = PhaseKind::Minimizer;
    fill_slots<minimizer::Phase>(plan);
    add_op(plan, minimizer::Phase::force, engine);
    add_op(plan, minimizer::Phase::force, hook);

    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    rt.run_stage(minimizer::Phase::force, 0);

    EXPECT_EQ(engine.n_launch, 0);
    EXPECT_EQ(hook.n_launch, 1);
    EXPECT_EQ(comm.n_reverse, 1);
    EXPECT_TRUE(
        (comm.n_forward == 1 && comm.n_exchange == 0 && comm.n_borders == 0)
        || (comm.n_forward == 0 && comm.n_exchange == 1
            && comm.n_borders == 1));
    ASSERT_GE(log.size(), 2u);
    EXPECT_EQ(log.back(), "reverse");
}

TEST(TestVerletComm, MdRefreshSequenceIsForwardThenReverse) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neigh;
    Hook hook;
    std::vector<std::string> log;
    comm.log = &log;
    auto plan = md_plan(neigh, hook);

    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    rt.step(0);

    ASSERT_EQ(log.size(), 2u);
    EXPECT_EQ(log[0], "forward");
    EXPECT_EQ(log[1], "reverse");
}

TEST(TestVerletComm, MdRebuildSequenceIsExchangeBordersReverse) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neigh;
    Hook hook;
    std::vector<std::string> log;
    comm.log = &log;
    auto plan = md_plan(neigh, hook);

    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    for (int s = 0; s < 10; ++s)
        rt.step(s);

    std::vector<std::string> rebuild;
    for (std::size_t i = 0; i + 2 < log.size(); ++i) {
        if (log[i] == "exchange" && log[i + 1] == "borders") {
            rebuild.assign(log.begin() + static_cast<std::ptrdiff_t>(i),
                           log.begin() + static_cast<std::ptrdiff_t>(i) + 3);
            break;
        }
    }
    ASSERT_EQ(rebuild.size(), 3u);
    EXPECT_EQ(rebuild[0], "exchange");
    EXPECT_EQ(rebuild[1], "borders");
    EXPECT_EQ(rebuild[2], "reverse");
}
