/// @file cpu/multiscale/test_qm_lifecycle.cpp
/// @brief QM subsystem context and failure contracts, without an electronic solve.
#include <stdexcept>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/multiscale/internal/qm_subsystem.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"

using namespace ATV;
using cpu::multiscale::internal::QmSubsystem;

namespace {

cpu::Frame make_frame() {
    cpu::Frame frame(2);
    frame.blocks.at("atoms").insert(
        "qm", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                              kDynamicFieldId});
    frame.span(cpu::kAtomQm)[0] = 1;
    frame.span(cpu::kAtomQm)[1] = 0;
    frame.span(kAtom_atomic_number)[0] = 8; // the QM atom's species
    frame.span(kAtom_tag)[0] = 11;
    frame.span(kAtom_tag)[1] = 22;
    frame.span(kAtom_x)[0] = 1.25;
    frame.span(kAtom_y)[0] = -0.5;
    frame.span(kAtom_z)[0] = 0.75;
    frame.span(kAtom_x)[1] = 5;
    frame.box = cpu::Box::cubic(20);
    return frame;
}

struct ProbeEngine : cpu::QmEngine {
    cpu::Context* prepared = nullptr;
    cpu::Context* evaluated = nullptr;
    bool fail_prepare = false;
    bool fail_evaluate = false;
    bool force = true, stress = true, initial = false, trial = false;
    int stage = -1;
    BigInt step = -1;
    int resets = 0;

    std::string qm_engine_name() const override { return "context_probe"; }
    void prepare(const cpu::BuildContext& ctx) override {
        prepared = &ctx.runtime;
        EXPECT_EQ(ctx.frame.nlocal(), 1);
        EXPECT_EQ(ctx.frame.span(kAtom_atomic_number)[0], 8);
        EXPECT_DOUBLE_EQ(ctx.frame.span(kAtom_x)[0], 1.25);
        EXPECT_DOUBLE_EQ(ctx.frame.span(kAtom_y)[0], -0.5);
        EXPECT_DOUBLE_EQ(ctx.frame.span(kAtom_z)[0], 0.75);
        if (fail_prepare)
            throw std::runtime_error("probe prepare failure");
    }
    void evaluate(const cpu::LaunchContext& ctx) override {
        evaluated = &ctx.runtime;
        force = ctx.forces_requested;
        stress = ctx.stress_requested;
        initial = ctx.runtime.exec.initializing_forces;
        trial = ctx.runtime.exec.trial_evaluation;
        stage = ctx.stage;
        step = ctx.step;
        if (fail_evaluate)
            throw std::runtime_error("probe evaluation failure");
        ctx.frame.meta.get(kEnergy) = 2.0;
        if (force)
            ctx.frame.span(kAtom_fx)[0] = 3.0;
    }
    void reset_history() override { ++resets; }
};

} // namespace

TEST(QmLifecycle, PrepareUsesFilledPrivateFrameAndFailureCannotLaunch) {
    auto frame = make_frame();
    cpu::Runtime runtime;
    auto& context = runtime.context();
    context.exec.runtime = &runtime;
    ProbeEngine engine;
    QmSubsystem sub;
    sub.set_engine_ref(engine);
    engine.fail_prepare = true;
    EXPECT_THROW(sub.build({context, frame}, 1, false), std::runtime_error);
    double energy = -1;
    EXPECT_THROW(sub.launch({context, frame, 4, 37}, energy),
                 std::runtime_error);
    EXPECT_EQ(engine.evaluated, nullptr);
    engine.fail_prepare = false;
    EXPECT_NO_THROW(sub.build({context, frame}, 1, false));
    EXPECT_EQ(engine.prepared, &context);
    EXPECT_EQ(engine.prepared->exec.runtime, &runtime);
    EXPECT_NE(sub.private_frame(), &frame);
    EXPECT_NO_THROW(sub.launch({context, frame, 4, 37}, energy));
    EXPECT_DOUBLE_EQ(energy, 2);
}

TEST(QmLifecycle, ForwardsParentRequestsWithoutAdvancingResetHistory) {
    auto frame = make_frame();
    cpu::Context context;
    IterationStatus status;
    ConvergenceMetrics metrics;
    context.loop = {&status, &metrics};
    ProbeEngine engine;
    QmSubsystem sub;
    sub.set_engine_ref(engine);
    sub.reset_history_each_launch() = true;
    sub.build({context, frame}, 1, false);
    double energy = 0;
    for (int mode = 0; mode < 3; ++mode) {
        context.exec.initializing_forces = mode == 0;
        context.exec.trial_evaluation = mode == 1;
        context.exec.forces_requested = false;
        context.exec.stress_requested = false;
        frame.span(kAtom_fx)[0] = 7;
        sub.launch({context, frame, 8, 91, true, true}, energy);
        EXPECT_EQ(engine.evaluated, &context);
        EXPECT_FALSE(engine.force);
        EXPECT_FALSE(engine.stress);
        EXPECT_EQ(engine.initial, mode == 0);
        EXPECT_EQ(engine.trial, mode == 1);
        EXPECT_EQ(engine.stage, 8);
        EXPECT_EQ(engine.step, 91);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 7);
        EXPECT_EQ(context.loop.status, &status);
        EXPECT_EQ(context.loop.metrics, &metrics);
        EXPECT_EQ(engine.resets, mode == 2 ? 1 : 0);
    }
    context.exec.forces_requested = true;
    context.exec.stress_requested = true;
    sub.launch({context, frame, 9, 92, false, false}, energy);
    EXPECT_FALSE(engine.force);
    EXPECT_FALSE(engine.stress);
}

TEST(QmLifecycle, EvaluationFailureDoesNotScatterAndCanRetry) {
    auto frame = make_frame();
    cpu::Context context;
    context.exec.trial_evaluation = true;
    ProbeEngine engine;
    QmSubsystem sub;
    sub.set_engine_ref(engine);
    sub.build({context, frame}, 1, false);
    frame.meta.get(kEnergy) = 11;
    frame.span(kAtom_fx)[0] = 7;
    double energy = -1;
    engine.fail_evaluate = true;
    EXPECT_THROW(sub.launch({context, frame, 0, 0}, energy),
                 std::runtime_error);
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 11);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 7);
    EXPECT_DOUBLE_EQ(energy, -1);
    EXPECT_TRUE(context.exec.trial_evaluation);
    engine.fail_evaluate = false;
    sub.launch({context, frame, 0, 0}, energy);
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 13);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 10);
}
