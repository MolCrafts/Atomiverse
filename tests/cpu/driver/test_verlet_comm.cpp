/// @file tests/cpu/driver/test_verlet_comm.cpp
/// @brief MD `Runtime::step` is LAMMPS Verlet::run comm on `ctx.md.comm`.

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/plan.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/dynamics/atom_sort.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

using namespace ATV;

namespace {

class SpyComm : public cpu::Comm {
  public:
    int n_setup = 0;
    int n_forward = 0;
    int n_reverse = 0;
    int n_exchange = 0;
    int n_borders = 0;
    double ghost_cutoff = 0;
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
    void reverse_comm(cpu::ReversePacker&) override {}
    void set_ghost_cutoff(double reach) override { ghost_cutoff = reach; }

    /// Forget the calls made so far (the setup's), keeping the log binding.
    void reset() {
        n_setup = n_forward = n_reverse = n_exchange = n_borders = 0;
    }
};

class SpyNeighbor : public cpu::Component, public cpu::NeighborProvider {
  public:
    int n_decide = 0;
    int n_launch = 0;
    int delay = 10;
    double reach = 0;
    const cpu::md::PairList* published = nullptr;
    std::string name() const override { return "spy_neighbor"; }
    bool verlet_engine() const override { return true; }
    NeighborProvider* neighbor_provider() override { return this; }
    const NeighborProvider* as_neighbor_provider() const override {
        return this;
    }
    void forbid_row_reordering() override {}
    double supported_cutoff() const override { return std::min(2.5, reach); }
    double neighbor_reach() const override { return reach; }
    bool decide_reneighbor(cpu::Context&, cpu::Frame&, bool) override {
        ++n_decide;
        return n_decide >= delay;
    }
    void launch(const cpu::LaunchContext& ctx) override {
        ++n_launch;
        if (published)
            ctx.runtime.md.pair_list = published;
    }
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
    std::string label = "hook";
    bool initialize = false;
    bool fail = false;
    bool saw_initializing = false;
    std::function<void(const cpu::LaunchContext&)> action;
    std::string name() const override { return label; }
    void launch(const cpu::LaunchContext& ctx) override {
        ++n_launch;
        saw_initializing = ctx.runtime.exec.initializing_forces;
        if (log)
            log->push_back(label);
        if (fail)
            throw std::runtime_error("hook failure");
        if (action)
            action(ctx);
    }
    void initialize_forces(const cpu::LaunchContext& ctx) override {
        if (initialize)
            launch(ctx);
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
    rt.prepare_run();
    // Count the steps only; the setup's pbc + exchange + borders + build
    // is SerialSetupWrapsThenUsesCommThenNeighbor's.
    comm.reset();
    neigh.n_launch = 0;

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
    rt.prepare_run();
    // Count the steps only; the setup's pbc + exchange + borders + build
    // is SerialSetupWrapsThenUsesCommThenNeighbor's.
    comm.reset();
    neigh.n_launch = 0;

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
    rt.prepare_run();
    // Count the steps only; the setup's pbc + exchange + borders + build
    // is SerialSetupWrapsThenUsesCommThenNeighbor's.
    comm.reset();
    neigh.n_launch = 0;

    constexpr int n_steps = 10;
    for (int s = 0; s < n_steps; ++s)
        rt.step(s);

    EXPECT_EQ(comm.n_reverse, n_steps);
    EXPECT_EQ(comm.n_forward + comm.n_borders, n_steps);
}

TEST(TestVerletComm, SerialSetupWrapsThenUsesCommThenNeighbor) {
    // LAMMPS Verlet::setup runs domain->pbc() + exchange + borders +
    // neighbor->build on every rank count. An owned row just outside the box
    // is wrapped in before the neighbor list is rebuilt from it.
    auto frame = one_atom();
    frame.span(kAtom_x)[0] = -0.5;
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
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 9.5);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_y)[0], 1.0);
    EXPECT_EQ(comm.n_exchange, 1);
    EXPECT_EQ(comm.n_borders, 1);
    EXPECT_EQ(neigh.n_launch, 1);
    EXPECT_EQ(comm.n_forward, 0);
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

TEST(TestVerletComm, RebuildRefreshesSlabsAfterExternalCellMutation) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neighbor;
    neighbor.delay = 1;
    Hook hook, force;
    auto plan = md_plan(neighbor, hook);
    add_op(plan, md::Phase::force, force);
    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    rt.prepare_run();
    EXPECT_EQ(force.n_launch, 1);
    ASSERT_NE(rt.context().md.dd, nullptr);
    const auto* dd = rt.context().md.dd;
    auto cartesian_slab = [&] {
        return frame.box.cartesian_bounds(
            Vec3{dd->sublo[0], dd->sublo[1], dd->sublo[2]},
            Vec3{dd->subhi[0], dd->subhi[1], dd->subhi[2]});
    };
    EXPECT_DOUBLE_EQ(dd->subhi[0], 1.0);
    EXPECT_DOUBLE_EQ(cartesian_slab().second[0], 10.0);
    EXPECT_EQ(comm.n_borders, 1);

    auto cell = frame.box.cell();
    cell(0, 1) = 1.5;
    frame.box.set_cell(cell);
    rt.prepare_run();
    EXPECT_EQ(force.n_launch, 2);
    for (int d = 0; d < 3; ++d) {
        EXPECT_DOUBLE_EQ(rt.context().md.dd->sublo[d], 0.0);
        EXPECT_DOUBLE_EQ(rt.context().md.dd->subhi[d], 1.0);
    }
    EXPECT_DOUBLE_EQ(cartesian_slab().second[0], 11.5);
    EXPECT_EQ(comm.n_borders, 2);
    frame.box = cpu::Box::cubic(12.0);
    rt.prepare_run();
    EXPECT_EQ(force.n_launch, 3);
    EXPECT_EQ(comm.n_borders, 3);
    const auto bounds = cartesian_slab();
    for (int d = 0; d < 3; ++d) {
        EXPECT_DOUBLE_EQ(dd->sublo[d], 0.0);
        EXPECT_DOUBLE_EQ(dd->subhi[d], 1.0);
        EXPECT_DOUBLE_EQ(bounds.first[d], 0.0);
        EXPECT_DOUBLE_EQ(bounds.second[d], 12.0);
    }
}

TEST(TestVerletComm, RemovingNeighborClearsCommunicationReach) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neighbor;
    neighbor.reach = 3.0;
    Hook hook;
    auto plan = md_plan(neighbor, hook);
    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    rt.prepare_run();
    EXPECT_DOUBLE_EQ(comm.ghost_cutoff, 3.0);

    Plan<cpu::Component> without_neighbor;
    without_neighbor.kind = PhaseKind::MD;
    fill_slots<md::Phase>(without_neighbor);
    add_op(without_neighbor, md::Phase::end_of_step, hook);
    rt.load(without_neighbor);
    rt.prepare_run();
    EXPECT_DOUBLE_EQ(comm.ghost_cutoff, 0.0);
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
    // The force slot's comm only; the setup's is not part of it.
    comm.reset();
    log.clear();
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
    rt.prepare_run();
    hook.n_launch = 0;
    // The force slot's comm only; the setup's is not part of it.
    comm.reset();
    log.clear();
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
    // The force slot's comm only; the setup's is not part of it.
    comm.reset();
    log.clear();
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
    rt.prepare_run();
    log.clear(); // the setup's exchange + borders are not this step's
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
    rt.prepare_run();
    log.clear(); // the setup's exchange + borders are not this step's
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

TEST(TestVerletComm, PrepareRunSeedsForcesOnceWithoutIntegrationOrObservers) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neigh;
    Hook observer, integrate, pre, force, post;
    std::vector<std::string> log;
    comm.log = &log;
    pre.label = "pre";
    pre.log = &log;
    pre.initialize = true;
    force.label = "force";
    force.log = &log;
    post.label = "post";
    post.log = &log;
    post.initialize = true;
    auto plan = md_plan(neigh, observer);
    add_op(plan, md::Phase::initial_integrate, integrate);
    add_op(plan, md::Phase::pre_force, pre);
    add_op(plan, md::Phase::force, force);
    add_op(plan, md::Phase::post_force, post);
    // An ordinary observer in post_force must not run during initialization.
    add_op(plan, md::Phase::post_force, observer);
    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    comm.reset();
    log.clear();
    rt.prepare_run();
    const std::vector<std::string> expected{"pre", "force", "reverse", "post"};
    EXPECT_EQ(log, expected);
    EXPECT_TRUE(pre.saw_initializing);
    EXPECT_TRUE(force.saw_initializing);
    EXPECT_TRUE(post.saw_initializing);
    EXPECT_FALSE(rt.context().exec.initializing_forces);
    EXPECT_EQ(observer.n_launch, 0);
    EXPECT_EQ(integrate.n_launch, 0);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 1.0);
    rt.prepare_run();
    EXPECT_EQ(log, expected);
    EXPECT_EQ(comm.n_reverse, 1);
}

TEST(TestVerletComm, NestedMdUsesVerletOrderingAndRestoresParentBindings) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor parent_neighbor, child_neighbor;
    cpu::md::PairList parent_pairs, child_pairs;
    parent_neighbor.published = &parent_pairs;
    child_neighbor.published = &child_pairs;
    Hook parent_observer, child_observer, parent_force, child_force;
    std::vector<std::string> log;
    comm.log = &log;
    parent_force.label = "parent_force";
    parent_force.log = &log;
    child_force.label = "child_force";
    child_force.log = &log;
    child_observer.label = "child_end";
    child_observer.log = &log;
    auto parent = md_plan(parent_neighbor, parent_observer);
    auto child = md_plan(child_neighbor, child_observer);
    add_op(parent, md::Phase::force, parent_force);
    add_op(child, md::Phase::force, child_force);
    cpu::Runtime rt;
    rt.init();
    rt.load(parent);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    rt.prepare_run();
    rt.build_plan(child);
    const auto* parent_ghost = rt.context().md.ghost;
    log.clear();
    comm.reset();
    rt.step_plan(child, 0);
    const std::vector<std::string> expected{
        "exchange", "borders",     "child_force", "reverse",
        "forward",  "child_force", "reverse",     "child_end"};
    EXPECT_EQ(log, expected);
    EXPECT_EQ(child_neighbor.n_decide, 1);
    EXPECT_EQ(parent_neighbor.n_decide, 0);
    EXPECT_EQ(rt.context().md.comm, &comm);
    EXPECT_EQ(rt.context().md.pair_list, &parent_pairs);
    EXPECT_EQ(rt.context().md.ghost, parent_ghost);
    EXPECT_FALSE(rt.context().exec.initializing_forces);
    // Restoring a physical parent also invalidates its halo and forces:
    // they must be rebuilt before a subsequent integration can use them.
    log.clear();
    rt.prepare_run();
    const std::vector<std::string> restored{"exchange", "borders",
                                            "parent_force", "reverse"};
    EXPECT_EQ(log, restored);
    EXPECT_EQ(parent_observer.n_launch, 0);
    rt.destroy_plan(child);
}

TEST(TestVerletComm,
     NestedForceFailureRestoresParentAndCanRetryInitialization) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor parent_neighbor, child_neighbor;
    cpu::md::PairList parent_pairs, child_pairs;
    parent_neighbor.published = &parent_pairs;
    child_neighbor.published = &child_pairs;
    Hook parent_observer, child_observer, parent_force, child_force;
    child_force.fail = true;
    auto parent = md_plan(parent_neighbor, parent_observer);
    auto child = md_plan(child_neighbor, child_observer);
    add_op(parent, md::Phase::force, parent_force);
    add_op(child, md::Phase::force, child_force);
    cpu::Runtime rt;
    rt.init();
    rt.load(parent);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    rt.prepare_run();
    rt.build_plan(child);
    const auto* parent_ghost = rt.context().md.ghost;
    EXPECT_THROW(rt.step_plan(child, 0), std::runtime_error);
    EXPECT_EQ(rt.context().md.comm, &comm);
    EXPECT_EQ(rt.context().md.pair_list, &parent_pairs);
    EXPECT_EQ(rt.context().md.ghost, parent_ghost);
    EXPECT_FALSE(rt.context().exec.initializing_forces);
    EXPECT_EQ(child_observer.n_launch, 0);
    const int before = parent_force.n_launch;
    rt.prepare_run();
    EXPECT_EQ(parent_force.n_launch, before + 1);
    rt.prepare_run();
    EXPECT_EQ(parent_force.n_launch, before + 1);
    child_force.fail = false;
    EXPECT_NO_THROW(rt.step_plan(child, 0));
    EXPECT_EQ(child_observer.n_launch, 1);
    EXPECT_EQ(rt.context().md.pair_list, &parent_pairs);
    rt.destroy_plan(child);
}

TEST(TestVerletComm, PhysicalChildCannotEnterAnActiveForcePipeline) {
    for (bool child_is_md : {false, true}) {
        for (bool through_scf : {false, true}) {
            SCOPED_TRACE(child_is_md ? "MD child" : "Minimizer child");
            SCOPED_TRACE(through_scf ? "through SCF" : "direct");
            auto frame = one_atom();
            SpyComm comm;
            SpyNeighbor parent_neighbor, child_neighbor;
            Hook parent_observer, child_observer, parent_force, child_force,
                bridge;
            auto parent = md_plan(parent_neighbor, parent_observer);
            add_op(parent, md::Phase::force, parent_force);
            Plan<cpu::Component> child;
            if (child_is_md) {
                child = md_plan(child_neighbor, child_observer);
                add_op(child, md::Phase::force, child_force);
            } else {
                child.kind = PhaseKind::Minimizer;
                fill_slots<minimizer::Phase>(child);
                add_op(child, minimizer::Phase::force, child_force);
            }
            Plan<cpu::Component> scf_plan;
            scf_plan.kind = PhaseKind::SCF;
            fill_slots<scf::Phase>(scf_plan);
            add_op(scf_plan, scf::Phase::fock, bridge);
            cpu::Runtime rt;
            rt.init();
            rt.load(parent);
            rt.set_frame(frame);
            bind_spy(rt, comm);
            rt.build_components();
            rt.build_plan(child);
            rt.build_plan(scf_plan);
            child_force.action = [](const cpu::LaunchContext& ctx) {
                ctx.frame.span(kAtom_x)[0] = 5.0;
            };
            bridge.action = [&](const cpu::LaunchContext&) {
                rt.step_plan(child, 0);
            };
            parent_force.action = [&](const cpu::LaunchContext&) {
                rt.step_plan(through_scf ? scf_plan : child, 0);
            };
            comm.reset();
            EXPECT_THROW(rt.prepare_run(), std::runtime_error);
            EXPECT_EQ(comm.n_setup, 0);
            EXPECT_EQ(comm.n_exchange, 0);
            EXPECT_EQ(comm.n_borders, 0);
            EXPECT_EQ(comm.n_forward, 0);
            EXPECT_EQ(comm.n_reverse, 0);
            EXPECT_EQ(child_force.n_launch, 0);
            EXPECT_EQ(child_observer.n_launch, 0);
            EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 1.0);
            EXPECT_FALSE(rt.context().exec.initializing_forces);
            // The rejected child must leave the parent initialization retryable.
            parent_force.action = {};
            EXPECT_NO_THROW(rt.prepare_run());
            EXPECT_EQ(comm.n_reverse, 1);
            rt.destroy_plan(scf_plan);
            rt.destroy_plan(child);
        }
    }
}

TEST(TestVerletComm, ObserverOnlyRefreshPreservesInitialForcesAndHistory) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neighbor;
    Hook observer, added_observer, history, force, added_force;
    history.initialize = true;
    auto plan = md_plan(neighbor, observer);
    add_op(plan, md::Phase::pre_force, history);
    add_op(plan, md::Phase::force, force);
    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    rt.prepare_run();
    EXPECT_EQ(history.n_launch, 1);
    EXPECT_EQ(force.n_launch, 1);

    Plan<cpu::Component> additions;
    additions.lifecycle_only = {&added_observer};
    rt.build_plan(additions);
    add_op(plan, md::Phase::end_of_step, added_observer);
    rt.load(plan);
    rt.prepare_run();
    EXPECT_EQ(history.n_launch, 1);
    EXPECT_EQ(force.n_launch, 1);
    EXPECT_EQ(added_observer.n_launch, 0);

    additions.lifecycle_only = {&added_force};
    rt.build_plan(additions);
    add_op(plan, md::Phase::force, added_force);
    rt.load(plan);
    rt.prepare_run();
    EXPECT_EQ(history.n_launch, 2);
    EXPECT_EQ(force.n_launch, 2);
    EXPECT_EQ(added_force.n_launch, 1);
    EXPECT_EQ(added_observer.n_launch, 0);
    rt.prepare_run();
    EXPECT_EQ(history.n_launch, 2);
    EXPECT_EQ(force.n_launch, 2);
    EXPECT_EQ(added_force.n_launch, 1);
}

TEST(TestVerletComm, NestedMdRejectsForceVetoAndRestoresCallerWindow) {
    class NestedMd : public cpu::Driver<md::Phase> {
        void validate_components() const override {}
    } child;
    Hook force, integrate;
    child.on(md::Phase::force).add_ref(force);
    child.on(md::Phase::initial_integrate).add_ref(integrate);
    auto node = backend_traits<cpu::device_tag>::nested_slot(
        std::unique_ptr<NestedMd>{}, &child, 1, true, true);
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(10.0);
    Plan<cpu::Component> parent;
    parent.kind = PhaseKind::SCF;
    fill_slots<scf::Phase>(parent);
    cpu::Runtime runtime;
    runtime.init();
    runtime.load(parent);
    runtime.set_frame(frame);
    node->build({runtime.context(), frame});
    runtime.context().exec.forces_requested = true;
    frame.begin_resolution_window(123);
    const auto before = frame.resolution_window();
    const cpu::LaunchContext request{runtime.context(), frame, 0, 0, false};
    EXPECT_THROW(node->launch(request), std::runtime_error);
    EXPECT_EQ(force.n_launch, 0);
    EXPECT_EQ(integrate.n_launch, 0);
    EXPECT_TRUE(runtime.context().exec.forces_requested);
    EXPECT_FALSE(runtime.context().exec.initializing_forces);
    const auto after = frame.resolution_window();
    EXPECT_EQ(after.active, before.active);
    EXPECT_EQ(after.budget, before.budget);
    EXPECT_EQ(after.count, before.count);
    node->destroy({runtime.context(), frame});
}

TEST(TestVerletComm, NestedMdKeepsItsClockAcrossInvocations) {
    class NestedMd : public cpu::Driver<md::Phase> {
        void validate_components() const override {}
    } child;
    Hook force, observer;
    std::vector<int> setup_steps, force_steps, observer_steps;
    force.action = [&](const cpu::LaunchContext& ctx) {
        (ctx.runtime.exec.initializing_forces ? setup_steps : force_steps)
            .push_back(ctx.step);
    };
    observer.action = [&](const cpu::LaunchContext& ctx) {
        observer_steps.push_back(ctx.step);
    };
    child.on(md::Phase::force).add_ref(force);
    child.on(md::Phase::end_of_step).add_ref(observer);
    auto frame = one_atom();
    Plan<cpu::Component> parent;
    parent.kind = PhaseKind::SCF;
    fill_slots<scf::Phase>(parent);
    cpu::Runtime runtime;
    runtime.init();
    runtime.load(parent);
    runtime.set_frame(frame);
    child.build_on(runtime);
    EXPECT_EQ(child.run_nested(runtime.context(), frame, 1), 1);
    EXPECT_EQ(child.run_nested(runtime.context(), frame, 2), 2);
    EXPECT_EQ(setup_steps, (std::vector<int>{0, 1}));
    EXPECT_EQ(force_steps, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(observer_steps, (std::vector<int>{0, 1, 2}));
    child.destroy_on(runtime);
    child.build_on(runtime);
    EXPECT_EQ(child.run_nested(runtime.context(), frame, 0), 0);
    EXPECT_EQ(setup_steps, (std::vector<int>{0, 1, 0}));
    EXPECT_EQ(observer_steps.size(), 3u);
    child.destroy_on(runtime);
}

TEST(TestVerletComm, InitialForceSetupDoesNotPolluteStepTiming) {
    auto frame = one_atom();
    SpyComm comm;
    SpyNeighbor neighbor;
    Hook observer, force;
    auto plan = md_plan(neighbor, observer);
    add_op(plan, md::Phase::force, force);
    cpu::Runtime runtime;
    runtime.init();
    runtime.enable_timing(true);
    runtime.load(plan);
    runtime.set_frame(frame);
    bind_spy(runtime, comm);
    runtime.build_components();
    force.fail = true;
    EXPECT_THROW(runtime.prepare_run(), std::runtime_error);
    EXPECT_TRUE(runtime.timing().enabled());
    force.fail = false;
    runtime.prepare_run();
    for (const auto& sample : runtime.timing().per_component())
        EXPECT_EQ(sample.calls, 0u);
    EXPECT_EQ(runtime.timing().steps(), 0u);
    runtime.step(0);
    EXPECT_EQ(runtime.timing().steps(), 1u);
    const auto& keys = runtime.timing().keys();
    const auto& samples = runtime.timing().per_component();
    std::uint64_t forwards = 0, reverses = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (keys[i].component == "comm_forward")
            forwards += samples[i].calls;
        if (keys[i].component == "comm_reverse")
            reverses += samples[i].calls;
    }
    EXPECT_EQ(forwards, 1u);
    EXPECT_EQ(reverses, 1u);
}

TEST(TestVerletComm, LinearNeighborDecisionSeesLineSearchMovement) {
    for (const int mode : {0, 1, 2}) {
        auto frame = one_atom();
        class PositionNeighbor : public SpyNeighbor {
          public:
            double seen = 0;
            bool decide_reneighbor(cpu::Context&, cpu::Frame& frame,
                                   bool) override {
                ++n_decide;
                seen = frame.span(kAtom_x)[0];
                return seen > 2.0;
            }
        } neighbor;
        Hook move, force;
        move.action = [](const cpu::LaunchContext& ctx) {
            ctx.frame.span(kAtom_x)[0] = 3.0;
        };
        SpyComm comm;
        Plan<cpu::Component> plan;
        plan.kind = PhaseKind::Minimizer;
        fill_slots<minimizer::Phase>(plan);
        add_op(plan, minimizer::Phase::line_search, move);
        add_op(plan, minimizer::Phase::force, neighbor);
        add_op(plan, minimizer::Phase::force, force);
        cpu::Runtime runtime;
        runtime.init();
        runtime.load(plan);
        runtime.set_frame(frame);
        bind_spy(runtime, comm);
        runtime.build_components();
        runtime.prepare_run();
        comm.reset();
        neighbor.n_decide = 0;
        if (mode == 2) {
            runtime.run_stage(minimizer::Phase::line_search, 0);
            runtime.run_stage(minimizer::Phase::force, 0);
        } else if (mode == 1) {
            for (int slot = 0; slot < slot_count<minimizer::Phase>(); ++slot)
                runtime.step_stage(slot, 0);
        } else
            runtime.step(0);
        EXPECT_EQ(neighbor.n_decide, 1);
        EXPECT_DOUBLE_EQ(neighbor.seen, 3.0);
        EXPECT_EQ(comm.n_exchange, 1);
        EXPECT_EQ(comm.n_forward, 0);
        runtime.destroy_components();
    }
}

TEST(TestVerletComm, RuntimeSortsBetweenExchangeAndBorders) {
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(10);
    frame.span(kAtom_x)[0] = 8;
    frame.span(kAtom_x)[1] = 1;
    frame.span(kAtom_tag)[0] = 80;
    frame.span(kAtom_tag)[1] = 10;
    class SortingNeighbor : public SpyNeighbor {
      public:
        cpu::md::AtomSortState sorting;
        void activate(cpu::Context& context) override {
            context.md.atom_sort = &sorting;
        }
        void launch(const cpu::LaunchContext& ctx) override {
            EXPECT_EQ(ctx.frame.span(kAtom_tag)[0], 10);
            SpyNeighbor::launch(ctx);
        }
    } neighbor;
    neighbor.reach = 2;
    neighbor.sorting.configure(1, 1);
    neighbor.sorting.next_sort = 0;
    class OrderingComm : public SpyComm {
      public:
        void exchange(const cpu::LaunchContext& ctx) override {
            EXPECT_EQ(ctx.frame.span(kAtom_tag)[0], 80);
            SpyComm::exchange(ctx);
        }
        void borders(const cpu::LaunchContext& ctx) override {
            EXPECT_EQ(ctx.frame.span(kAtom_tag)[0], 10);
            SpyComm::borders(ctx);
        }
    } comm;
    Hook hook;
    auto plan = md_plan(neighbor, hook);
    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    bind_spy(rt, comm);
    rt.build_components();
    rt.prepare_run();
    EXPECT_EQ(comm.n_exchange, 1);
    EXPECT_EQ(comm.n_borders, 1);
    EXPECT_EQ(neighbor.n_launch, 1);
    EXPECT_EQ(neighbor.sorting.next_sort, 1);
}

TEST(TestVerletComm, DistinctNeighborOwnersAreRejected) {
    auto frame = one_atom();
    SpyNeighbor first, second;
    Hook hook;
    auto plan = md_plan(first, hook);
    add_op(plan, md::Phase::neighbor, second);
    cpu::Runtime rt;
    rt.init();
    rt.load(plan);
    rt.set_frame(frame);
    EXPECT_THROW(rt.build_components(), std::runtime_error);
}

TEST(TestVerletComm, SamplingScoresRebuildAfterProposalAndRollback) {
    auto frame = one_atom();
    SpyNeighbor neighbor;
    SpyComm comm;
    Hook score;
    score.action = [&](const cpu::LaunchContext&) {
        EXPECT_EQ(neighbor.n_launch, comm.n_borders);
        EXPECT_EQ(comm.n_borders, score.n_launch);
    };
    Plan<cpu::Component> plan;
    plan.kind = PhaseKind::Sampling;
    fill_slots<sampling::Phase>(plan);
    add_op(plan, sampling::Phase::energy, neighbor);
    add_op(plan, sampling::Phase::energy, score);
    cpu::Runtime runtime;
    runtime.init();
    runtime.load(plan);
    runtime.set_frame(frame);
    bind_spy(runtime, comm);
    runtime.build_components();
    EXPECT_EQ(neighbor.n_launch, 0);
    frame.span(kAtom_x)[0] = 3;
    runtime.step_stage(slot_of(sampling::Phase::energy), 0);
    frame.span(kAtom_x)[0] = 1;
    runtime.step_stage(slot_of(sampling::Phase::energy), 0);
    EXPECT_EQ(neighbor.n_launch, 2);
    EXPECT_EQ(comm.n_exchange, 2);
    EXPECT_EQ(comm.n_borders, 2);
    EXPECT_EQ(comm.n_reverse, 2);
    EXPECT_EQ(score.n_launch, 2);
}

namespace {
class CutoffConsumer : public cpu::Component, public cpu::NeighborConsumer {
  public:
    double cutoff = 2.7;
    std::string name() const override { return "cutoff_consumer"; }
    const cpu::NeighborConsumer* as_neighbor_consumer() const override {
        return this;
    }
    AccReal neighbor_cutoff() const override { return cutoff; }
    // Halo width is a different contract; a wider halo cannot repair the CSR.
    AccReal halo_cutoff() const override { return 5; }
    void launch(const cpu::LaunchContext&) override {}
};
class ContractDriver : public cpu::Driver<md::Phase> {
  public:
    ~ContractDriver() { this->flush_live_(); }
    void run(cpu::Frame& frame, int) { this->begin_run_(frame); }

  protected:
    void validate_components() const override {}
};
class ContractChild : public cpu::Driver<scf::Phase> {
    void validate_components() const override {}
};
} // namespace

TEST(TestVerletComm, InteractionCutoffExcludesSkinAtInitialAndLiveBuild) {
    SpyNeighbor neighbor;
    neighbor.reach = 2.8;
    CutoffConsumer consumer, added;
    auto frame = one_atom();
    ContractDriver driver;
    driver.on(md::Phase::neighbor).add_ref(neighbor);
    driver.on(md::Phase::force).add_ref(consumer);
    EXPECT_THROW(driver.build(), std::runtime_error);
    consumer.cutoff = 2.5;
    ASSERT_NO_THROW(driver.build());
    ASSERT_NO_THROW(driver.run(frame, 0));
    driver.on(md::Phase::force).add_ref(added);
    EXPECT_THROW(driver.run(frame, 0), std::runtime_error);
}

TEST(TestVerletComm, RawRuntimeValidatesInteractionCutoffNotEnumerationReach) {
    for (double cutoff : {2.5, 2.7}) {
        SpyNeighbor neighbor;
        neighbor.reach = 2.8;
        CutoffConsumer consumer;
        consumer.cutoff = cutoff;
        Hook hook;
        SpyComm comm;
        auto plan = md_plan(neighbor, hook);
        add_op(plan, md::Phase::force, consumer);
        auto frame = one_atom();
        cpu::Runtime runtime;
        runtime.init();
        runtime.load(plan);
        runtime.set_frame(frame);
        bind_spy(runtime, comm);
        if (cutoff == 2.5) {
            EXPECT_NO_THROW(runtime.build_components());
            // halo dependency 5 plus the actual .3 skin, not ten percent.
            EXPECT_NEAR(comm.ghost_cutoff, 5.3, 1e-14);
        } else
            EXPECT_THROW(runtime.build_components(), std::runtime_error);
        runtime.destroy_components();
    }
}

TEST(TestVerletComm, ParentValidatesBorrowedNonphysicalDescendantCutoff) {
    SpyNeighbor neighbor;
    neighbor.reach = 2.8;
    CutoffConsumer consumer;
    ContractChild child;
    child.on(scf::Phase::check).add_ref(consumer);
    ContractDriver parent;
    parent.on(md::Phase::neighbor).add_ref(neighbor);
    parent.on(md::Phase::force).add_ref(child, 0, 1);
    parent.build();
    auto frame = one_atom();
    cpu::Runtime runtime;
    runtime.init();
    runtime.load(parent.plan());
    runtime.set_frame(frame);
    EXPECT_THROW(runtime.build_components(), std::runtime_error);
}
