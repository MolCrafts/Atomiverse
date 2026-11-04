#include <gtest/gtest.h>
#include <limits>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/iteration_status.h"

#include "atomiverse/core/stage.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/minimize/cell_relax.h"
#include "atomiverse/cpu/minimize/cg.h"
#include "atomiverse/cpu/minimize/fire.h"
#include "atomiverse/cpu/minimize/joint_cell_relax.h"
#include "atomiverse/cpu/minimize/lbfgs.h"

using namespace ATV;

TEST(OptimizerMetrics, NonfiniteForceFailsBeforeCoordinatesOrMetricsChange) {
    auto check = [](auto& optimizer) {
        SCOPED_TRACE(optimizer.name());
        cpu::Frame frame(1);
        frame.box = cpu::Box::cubic(10);
        frame.span(kAtom_x)[0] = 2;
        frame.span(kAtom_fx)[0] = std::numeric_limits<double>::quiet_NaN();
        cpu::Context context;
        ConvergenceMetrics metrics;
        metrics.max_force = 123;
        context.loop.metrics = &metrics;
        optimizer.build({context, frame});
        for (auto phase :
             {minimizer::Phase::search_direction, minimizer::Phase::line_search,
              minimizer::Phase::convergence_check}) {
            EXPECT_THROW(optimizer.launch({context, frame, slot_of(phase), 0}),
                         std::runtime_error);
            EXPECT_EQ(frame.span(kAtom_x)[0], 2);
            EXPECT_EQ(metrics.max_force, 123);
        }
    };
    cpu::minimizer::CG cg;
    cpu::minimizer::LBFGS lbfgs;
    cpu::minimizer::FIRE fire;
    cpu::minimizer::JointIsotropicRelax joint;
    check(cg);
    check(lbfgs);
    check(fire);
    check(joint);
}

TEST(OptimizerMetrics, JointReportsMeasuredForceAndDisplacement) {
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(10);
    frame.span(kAtom_fx)[0] = 3;
    frame.span(kAtom_fy)[0] = 4;
    frame.span(kAtom_fx)[1] = 10;
    cpu::Context context;
    ConvergenceMetrics metrics;
    context.loop.metrics = &metrics;
    cpu::minimizer::JointIsotropicRelax joint(0.0, 0.1);
    joint.build({context, frame});
    joint.launch(
        {context, frame, slot_of(minimizer::Phase::search_direction), 0});
    joint.launch({context, frame, slot_of(minimizer::Phase::line_search), 0});
    joint.launch(
        {context, frame, slot_of(minimizer::Phase::convergence_check), 0});
    EXPECT_NEAR(frame.span(kAtom_x)[0], 0.03, 1e-14);
    EXPECT_NEAR(frame.span(kAtom_y)[0], 0.04, 1e-14);
    EXPECT_NEAR(frame.span(kAtom_x)[1], 0.1, 1e-14);
    EXPECT_EQ(metrics.max_force, 10);
    EXPECT_NEAR(metrics.rms_force, std::sqrt(125.0 / 6), 1e-14);
    EXPECT_NEAR(metrics.max_displacement, 0.1, 1e-14);
    EXPECT_NEAR(metrics.rms_displacement, std::sqrt(0.0125 / 6), 1e-14);
    EXPECT_TRUE(metrics.has_displacement);
}

TEST(OptimizerMetrics,
     JointRejectsNonfinitePressureBeforeGeometryOrMetricsChange) {
    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()}) {
        cpu::Frame frame(1);
        frame.box = cpu::Box::cubic(10);
        frame.span(kAtom_x)[0] = 2;
        frame.span(kAtom_fx)[0] = 1;
        cpu::Context context;
        ConvergenceMetrics metrics;
        metrics.max_force = 123;
        context.loop.metrics = &metrics;
        cpu::minimizer::JointIsotropicRelax joint;
        joint.build({context, frame});
        joint.launch(
            {context, frame, slot_of(minimizer::Phase::search_direction), 0});
        frame.meta.get(kStress)[0] = bad;
        for (auto phase :
             {minimizer::Phase::search_direction, minimizer::Phase::line_search,
              minimizer::Phase::convergence_check}) {
            EXPECT_THROW(joint.launch({context, frame, slot_of(phase), 0}),
                         std::runtime_error);
            EXPECT_EQ(frame.span(kAtom_x)[0], 2);
            EXPECT_EQ(frame.box.volume(), 1000);
            EXPECT_EQ(metrics.max_force, 123);
            EXPECT_EQ(joint.pressure(), 0);
        }
    }
}

TEST(OptimizerMetrics,
     JointRejectsInvalidMutableParametersBeforeGeometryChange) {
    using Joint = cpu::minimizer::JointIsotropicRelax;
    for (double Joint::* member :
         {&Joint::P_target, &Joint::atom_step, &Joint::alpha0,
          &Joint::max_scale, &Joint::bulk_mod}) {
        cpu::Frame frame(1);
        frame.box = cpu::Box::cubic(10);
        frame.span(kAtom_x)[0] = 2;
        frame.span(kAtom_fx)[0] = 1;
        cpu::Context context;
        Joint joint;
        joint.build({context, frame});
        joint.launch(
            {context, frame, slot_of(minimizer::Phase::search_direction), 0});
        joint.*member = std::numeric_limits<double>::quiet_NaN();
        EXPECT_THROW(joint.launch({context, frame,
                                   slot_of(minimizer::Phase::line_search), 0}),
                     std::runtime_error);
        EXPECT_EQ(frame.span(kAtom_x)[0], 2);
        EXPECT_EQ(frame.box.volume(), 1000);
    }
}

TEST(OptimizerMetrics, JointUsesForceMagnitudeAndMaximumDisplacementComponent) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    frame.span(kAtom_fx)[0] = 3;
    frame.span(kAtom_fy)[0] = 4;
    cpu::Context context;
    ConvergenceMetrics metrics;
    context.loop.metrics = &metrics;
    cpu::minimizer::JointIsotropicRelax joint(0.0, 0.1);
    joint.build({context, frame});
    joint.launch(
        {context, frame, slot_of(minimizer::Phase::search_direction), 0});
    joint.launch({context, frame, slot_of(minimizer::Phase::line_search), 0});
    joint.launch(
        {context, frame, slot_of(minimizer::Phase::convergence_check), 0});
    EXPECT_EQ(metrics.max_force, 5);
    EXPECT_NEAR(metrics.max_displacement, 0.08, 1e-14);
    EXPECT_NEAR(metrics.rms_displacement, 0.1 / std::sqrt(3.0), 1e-14);
}

TEST(OptimizerMetrics,
     IsotropicCellRejectsBadPressureOrParameterBeforeChangingGeometry) {
    for (bool bad_pressure : {false, true}) {
        for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
            cpu::Frame frame(1);
            frame.box = cpu::Box::cubic(10);
            frame.span(kAtom_x)[0] = 2;
            cpu::Context context;
            ConvergenceMetrics metrics;
            metrics.max_displacement = 123;
            context.loop.metrics = &metrics;
            cpu::minimizer::IsotropicCellRelax cell;
            cell.build({context, frame});
            cell.launch({context, frame,
                         slot_of(minimizer::Phase::search_direction), 0});
            if (bad_pressure)
                frame.meta.get(kStress)[0] = bad;
            else
                cell.bulk_mod = bad;
            for (auto phase : {minimizer::Phase::search_direction,
                               minimizer::Phase::line_search,
                               minimizer::Phase::convergence_check}) {
                EXPECT_THROW(cell.launch({context, frame, slot_of(phase), 0}),
                             std::runtime_error);
                EXPECT_EQ(frame.span(kAtom_x)[0], 2);
                EXPECT_EQ(frame.box.volume(), 1000);
                EXPECT_EQ(metrics.max_displacement, 123);
                EXPECT_EQ(cell.pressure(), 0);
            }
        }
    }
}

TEST(OptimizerMetrics,
     IsotropicCellPublishesPhysicalForceAndDisplacementUnits) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    frame.span(kAtom_x)[0] = 2;
    frame.span(kAtom_y)[0] = 3;
    frame.span(kAtom_fx)[0] = 3;
    frame.span(kAtom_fy)[0] = 4;
    frame.meta.get(kStress)[0] = 3000; // P=1, scale=1.05.
    cpu::Context context;
    context.md.reneighbor = false;
    ConvergenceMetrics metrics;
    context.loop.metrics = &metrics;
    cpu::minimizer::IsotropicCellRelax cell;
    cell.build({context, frame});
    cell.launch(
        {context, frame, slot_of(minimizer::Phase::search_direction), 0});
    cell.launch({context, frame, slot_of(minimizer::Phase::line_search), 0});
    cell.launch(
        {context, frame, slot_of(minimizer::Phase::convergence_check), 0});
    EXPECT_NEAR(frame.span(kAtom_x)[0], 2.1, 1e-14);
    EXPECT_NEAR(frame.span(kAtom_y)[0], 3.15, 1e-14);
    EXPECT_NEAR(frame.box.length(0), 10.5, 1e-14);
    EXPECT_TRUE(context.md.reneighbor);
    EXPECT_EQ(metrics.max_force, 5);
    EXPECT_NEAR(metrics.rms_force, 5 / std::sqrt(3.0), 1e-14);
    EXPECT_NEAR(metrics.max_displacement, 0.15, 1e-14);
    EXPECT_NEAR(metrics.rms_displacement, std::sqrt(0.0325 / 3), 1e-14);
    EXPECT_TRUE(metrics.has_displacement);
    EXPECT_NEAR(cell.pressure(), 1000 / frame.box.volume(), 1e-14);
}

TEST(OptimizerMetrics, IsotropicCellRejectsNonfiniteForceBeforeUpdatingState) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    frame.span(kAtom_x)[0] = 2;
    cpu::Context context;
    ConvergenceMetrics metrics;
    metrics.max_force = 123;
    context.loop.metrics = &metrics;
    cpu::minimizer::IsotropicCellRelax cell;
    cell.build({context, frame});
    cell.launch(
        {context, frame, slot_of(minimizer::Phase::search_direction), 0});
    frame.meta.get(kStress)[0] = 3000;
    frame.span(kAtom_fx)[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(cell.launch({context, frame,
                              slot_of(minimizer::Phase::convergence_check), 0}),
                 std::runtime_error);
    EXPECT_EQ(cell.pressure(), 0);
    EXPECT_EQ(cell.pressure_residual(), 0);
    EXPECT_EQ(metrics.max_force, 123);
    EXPECT_EQ(frame.span(kAtom_x)[0], 2);
    EXPECT_EQ(frame.box.volume(), 1000);
}

#include "atomiverse/cpu/driver/runtime.h"
namespace {
class PreparationProbe : public cpu::Component {
  public:
    std::string name() const override { return "preparation_probe"; }
    void launch(const cpu::LaunchContext& ctx) override {
        ++calls;
        initializing = ctx.runtime.exec.initializing_forces;
    }
    int calls = 0;
    bool initializing = false;
};
} // namespace
TEST(OptimizerMetrics, PreparationRunsOnlyForceSlotOnce) {
    PreparationProbe force, direction, output;
    Plan<cpu::Component> plan;
    plan.kind = PhaseKind::Minimizer;
    for (int i = 0; i < slot_count<minimizer::Phase>(); ++i)
        plan.slots.push_back({i, {}});
    plan.slots[slot_of(minimizer::Phase::force)].ops.push_back({&force});
    plan.slots[slot_of(minimizer::Phase::search_direction)].ops.push_back(
        {&direction});
    plan.slots[slot_of(minimizer::Phase::end_of_step)].ops.push_back({&output});
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    cpu::Runtime runtime;
    runtime.init();
    runtime.load(plan);
    runtime.set_frame(frame);
    runtime.build_components();
    runtime.prepare_run();
    runtime.prepare_run();
    EXPECT_EQ(force.calls, 1);
    EXPECT_TRUE(force.initializing);
    EXPECT_EQ(direction.calls, 0);
    EXPECT_EQ(output.calls, 0);
    EXPECT_FALSE(runtime.context().exec.initializing_forces);
    runtime.destroy_components();
    runtime.shutdown();
}

TEST(OptimizerMetrics, FirstMeasurementHasNoEnergyDifference) {
    auto check = [](auto& optimizer) {
        SCOPED_TRACE(optimizer.name());
        cpu::Frame frame(1);
        frame.box = cpu::Box::cubic(10);
        frame.span(kAtom_fx)[0] = 1;
        cpu::Context context;
        ConvergenceMetrics metrics;
        context.loop.metrics = &metrics;
        optimizer.build({context, frame});
        optimizer.launch(
            {context, frame, slot_of(minimizer::Phase::search_direction), 0});
        optimizer.launch(
            {context, frame, slot_of(minimizer::Phase::convergence_check), 0});
        EXPECT_GT(metrics.energy_change, 1e20);
        optimizer.destroy({context, frame});
    };
    cpu::minimizer::CG cg;
    cpu::minimizer::LBFGS lbfgs;
    cpu::minimizer::FIRE fire;
    check(cg);
    check(lbfgs);
    check(fire);
}

namespace {

/// Harmonic well E = ½ Σ x² (forces −x) with an optional energy penalty that
/// makes every displaced trial look uphill, so a line search must fail.
struct HarmonicWell {
    cpu::Frame& frame;
    bool uphill = false;
    double origin_energy = 0;
    void evaluate() {
        double e = 0;
        for (int i = 0; i < frame.nlocal(); ++i) {
            for (auto [x, fx] :
                 {std::pair{kAtom_x, kAtom_fx}, std::pair{kAtom_y, kAtom_fy},
                  std::pair{kAtom_z, kAtom_fz}}) {
                const double v = frame.span(x)[i];
                e += 0.5 * v * v;
                frame.span(fx)[i] = -v;
            }
        }
        frame.meta.get(kEnergy) = uphill ? origin_energy + 1.0 : e;
    }
};

template <typename Optimizer>
void lbfgs_like_step(Optimizer& optimizer, cpu::Context& context,
                     cpu::Frame& frame, HarmonicWell& well) {
    optimizer.launch(
        {context, frame, slot_of(minimizer::Phase::search_direction), 0});
    optimizer.launch(
        {context, frame, slot_of(minimizer::Phase::line_search), 0});
    well.evaluate();
    optimizer.launch(
        {context, frame, slot_of(minimizer::Phase::convergence_check), 0});
}

} // namespace

TEST(OptimizerMetrics, FailedLineSearchRecoversThenStopsWithoutThrowing) {
    auto check = [](auto& optimizer, const char* reason) {
        SCOPED_TRACE(optimizer.name());
        cpu::Frame frame(2);
        frame.box = cpu::Box::cubic(50);
        frame.span(kAtom_x)[0] = 1.0;
        frame.span(kAtom_y)[1] = -0.5;
        frame.span(kAtom_z)[1] = 0.25;
        cpu::Context context;
        ConvergenceMetrics metrics;
        IterationStatus status;
        context.loop.metrics = &metrics;
        context.loop.status = &status;
        HarmonicWell well{frame};
        context.opt.reeval_forces = [&](bool) { well.evaluate(); };
        optimizer.build({context, frame});
        well.evaluate();
        // Two ordinary steps build a curvature history and lower the energy.
        for (int s = 0; s < 2; ++s)
            lbfgs_like_step(optimizer, context, frame, well);
        EXPECT_FALSE(status.stop_requested);
        // Now every trial is uphill: the search fails along the quasi-Newton
        // direction, retries steepest descent, fails again, and stops.
        const double x0 = frame.span(kAtom_x)[0];
        const double y1 = frame.span(kAtom_y)[1];
        well.origin_energy = frame.meta.get(kEnergy);
        well.uphill = true;
        optimizer.launch(
            {context, frame, slot_of(minimizer::Phase::search_direction), 0});
        EXPECT_NO_THROW(optimizer.launch(
            {context, frame, slot_of(minimizer::Phase::line_search), 0}));
        EXPECT_TRUE(status.stop_requested);
        EXPECT_FALSE(status.converged);
        EXPECT_EQ(status.reason, reason);
        // The origin, its forces and its energy are restored.
        EXPECT_EQ(frame.span(kAtom_x)[0], x0);
        EXPECT_EQ(frame.span(kAtom_y)[1], y1);
        EXPECT_EQ(frame.span(kAtom_fx)[0], -x0);
        EXPECT_EQ(frame.meta.get(kEnergy), well.origin_energy);
    };
    cpu::minimizer::LBFGS lbfgs;
    check(lbfgs, "lbfgs_line_search_failed");
    EXPECT_EQ(lbfgs.line_search_recoveries(), 1);
    cpu::minimizer::CG cg;
    check(cg, "cg_line_search_failed");
}

TEST(OptimizerMetrics, LbfgsHarmonicWellStillConvergesWithoutRecovery) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(50);
    frame.span(kAtom_x)[0] = 0.15;
    frame.span(kAtom_y)[0] = -0.1;
    cpu::Context context;
    ConvergenceMetrics metrics;
    IterationStatus status;
    context.loop.metrics = &metrics;
    context.loop.status = &status;
    HarmonicWell well{frame};
    context.opt.reeval_forces = [&](bool) { well.evaluate(); };
    cpu::minimizer::LBFGS lbfgs;
    lbfgs.build({context, frame});
    well.evaluate();
    for (int s = 0; s < 6; ++s)
        lbfgs_like_step(lbfgs, context, frame, well);
    EXPECT_FALSE(status.stop_requested);
    EXPECT_EQ(lbfgs.line_search_recoveries(), 0);
    EXPECT_LT(metrics.max_force, 1e-8);
}
