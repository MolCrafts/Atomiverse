/// @file tests/cpu/minimize/test_anisotropic_cell.cpp
/// @brief Unit gates for AnisotropicCellRelax (mask, stress convention) and
///        the StressConverged criterion. GPW-driven e2e lives in
///        regressions/gpw/variable-cell/.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/anisotropic_cell_relax.h"

using namespace ATV;

namespace {

cpu::Frame one_atom_box(double L) {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0.25 * L;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(L);
    return f;
}

} // namespace

TEST(AnisotropicCellRelax, Name) {
    cpu::minimizer::AnisotropicCellRelax a(0.0, 0.05);
    EXPECT_EQ(a.name(), "anisotropic_cell_relax");
}

TEST(AnisotropicCellRelax, IntensiveStressStepAndMask) {
    // relax_x only: the scale step is exactly 1 + alpha0*sxx/bulk_mod;
    // frozen y/z stay
    // bit-identical even though syy/szz are nonzero.
    const double L0 = 10.0;
    auto frame = one_atom_box(L0);

    cpu::Context context;
    cpu::minimizer::AnisotropicCellRelax cell(0.0, /*alpha*/ 0.5);
    cell.bulk_mod = 1.0;
    cell.max_scale = 0.05;
    cell.relax_y = false;
    cell.relax_z = false;
    frame.meta.get(kStress)[0] = 0.02 * frame.box.volume();
    frame.meta.get(kStress)[1] = 0.04 * frame.box.volume();
    frame.meta.get(kStress)[2] = -0.04 * frame.box.volume();
    cell.build({context, frame});
    cell.launch(
        {context, frame, slot_of(minimizer::Phase::search_direction), 0});
    cell.launch({context, frame, slot_of(minimizer::Phase::line_search), 0});

    const double expected = L0 * (1.0 + 0.5 * 0.02 / 1.0); // 10.1, unclamped
    EXPECT_NEAR(frame.box.length(0), expected, 1e-12 * L0);
    EXPECT_EQ(frame.box.length(1), L0); // masked: bit-identical
    EXPECT_EQ(frame.box.length(2), L0);
    // Residual reads the relaxed axis only (0.02), not the masked 0.04.
    EXPECT_NEAR(cell.pressure_residual(), 0.02, 1e-15);
}

TEST(AnisotropicCellRelax, MaxScaleClampsIntensiveStep) {
    const double L0 = 10.0;
    auto frame = one_atom_box(L0);

    cpu::Context context;
    cpu::minimizer::AnisotropicCellRelax cell(0.0, /*alpha*/ 1.0);
    cell.bulk_mod = 1.0;
    cell.max_scale = 0.03;
    cell.relax_y = false;
    cell.relax_z = false;
    frame.meta.get(kStress)[0] = -5.0 * frame.box.volume();
    cell.build({context, frame});
    cell.launch(
        {context, frame, slot_of(minimizer::Phase::search_direction), 0});
    cell.launch({context, frame, slot_of(minimizer::Phase::line_search), 0});

    EXPECT_NEAR(frame.box.length(0), L0 * (1.0 - 0.03), 1e-12 * L0);
}

TEST(AnisotropicCellRelax, AllAxesMaskedThrowsAtComponentBuild) {
    auto frame = one_atom_box(10.0);
    cpu::Context context;
    cpu::minimizer::AnisotropicCellRelax cell(0.0, 0.05);
    cell.relax_x = cell.relax_y = cell.relax_z = false;
    EXPECT_THROW(cell.build({context, frame}), std::invalid_argument);
}

TEST(StressConverged, Name) {
    cpu::StressConverged sc(0.0, 1e-4);
    EXPECT_EQ(sc.name(), "stress_converged");
}

TEST(StressConverged, VerdictMaskedPerAxis) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10.0);
    const double V = frame.box.volume(); // 1000 Å³
    // kStress is the extensive Voigt virial [eV]; the criterion judges the
    // intensive W_dd/V [eV/Å³] against its tolerance, so the stresses under
    // test are published as σ·V.
    auto& st = frame.meta.get(kStress);
    st[0] = 5e-5 * V; // σ_xx under tol
    st[1] = 0.5 * V;  // way out — but masked below
    st[2] = -0.5 * V; // way out — masked
    cpu::Context rt;
    ConvergenceMetrics metrics;
    cpu::LoopState s{/*iteration=*/0, frame, rt, metrics};

    cpu::StressConverged x_only(0.0, 1e-4, true, false, false);
    EXPECT_EQ(x_only.verdict(s), Verdict::Converged);

    cpu::StressConverged all(0.0, 1e-4);
    EXPECT_EQ(all.verdict(s), Verdict::Continue);

    st[0] = 2e-4 * V; // σ_xx now out of tol
    EXPECT_EQ(x_only.verdict(s), Verdict::Continue);
}

TEST(StressConverged, InvalidParametersThrow) {
    EXPECT_THROW(cpu::StressConverged(0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(cpu::StressConverged(0.0, 1e-4, false, false, false),
                 std::invalid_argument);
}

TEST(AnisotropicCellRelax, RejectsBadPressureOrParameterBeforeGeometryChanges) {
    for (bool bad_pressure : {false, true}) {
        for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
            auto frame = one_atom_box(10);
            cpu::Context context;
            ConvergenceMetrics metrics;
            metrics.max_displacement = 123;
            context.loop.metrics = &metrics;
            cpu::minimizer::AnisotropicCellRelax cell;
            cell.build({context, frame});
            cell.launch({context, frame,
                         slot_of(minimizer::Phase::search_direction), 0});
            if (bad_pressure)
                frame.meta.get(kStress)[0] = bad;
            else
                cell.alpha0 = bad;
            for (auto phase : {minimizer::Phase::search_direction,
                               minimizer::Phase::line_search,
                               minimizer::Phase::convergence_check}) {
                EXPECT_THROW(cell.launch({context, frame, slot_of(phase), 0}),
                             std::runtime_error);
                EXPECT_EQ(frame.span(kAtom_x)[0], 2.5);
                EXPECT_EQ(frame.box.volume(), 1000);
                EXPECT_EQ(metrics.max_displacement, 123);
                EXPECT_EQ(cell.pressure(), 0);
            }
        }
    }
}

TEST(AnisotropicCellRelax, PublishesPhysicalMetricsAndInvalidatesNeighbors) {
    auto frame = one_atom_box(10);
    frame.span(kAtom_x)[0] = 2;
    frame.span(kAtom_y)[0] = 3;
    frame.span(kAtom_z)[0] = 4;
    frame.span(kAtom_fx)[0] = 3;
    frame.span(kAtom_fy)[0] = 4;
    frame.meta.get(kStress)[0] = 1000;
    frame.meta.get(kStress)[1] = -1000;
    frame.meta.get(kStress)[2] = 9000;
    cpu::Context context;
    context.md.reneighbor = false;
    ConvergenceMetrics metrics;
    context.loop.metrics = &metrics;
    cpu::minimizer::AnisotropicCellRelax cell;
    cell.relax_z = false;
    cell.build({context, frame});
    cell.launch(
        {context, frame, slot_of(minimizer::Phase::search_direction), 0});
    cell.launch({context, frame, slot_of(minimizer::Phase::line_search), 0});
    cell.launch(
        {context, frame, slot_of(minimizer::Phase::convergence_check), 0});
    EXPECT_NEAR(frame.span(kAtom_x)[0], 2.1, 1e-14);
    EXPECT_NEAR(frame.span(kAtom_y)[0], 2.85, 1e-14);
    EXPECT_EQ(frame.span(kAtom_z)[0], 4);
    EXPECT_NEAR(frame.box.length(0), 10.5, 1e-14);
    EXPECT_NEAR(frame.box.length(1), 9.5, 1e-14);
    EXPECT_EQ(frame.box.length(2), 10);
    EXPECT_TRUE(context.md.reneighbor);
    EXPECT_EQ(metrics.max_force, 5);
    EXPECT_NEAR(metrics.rms_force, 5 / std::sqrt(3.0), 1e-14);
    EXPECT_NEAR(metrics.max_displacement, 0.15, 1e-14);
    EXPECT_NEAR(metrics.rms_displacement, std::sqrt(0.0325 / 3), 1e-14);
    EXPECT_TRUE(metrics.has_displacement);
}

TEST(AnisotropicCellRelax, RejectsTriclinicCellBeforeAffineUpdate) {
    auto frame = one_atom_box(10);
    Mat3 cell = Mat3::diagonal(10, 10, 10);
    cell(0, 1) = 1;
    frame.box = cpu::Box(cell);
    cpu::Context context;
    cpu::minimizer::AnisotropicCellRelax optimizer;
    optimizer.build({context, frame});
    EXPECT_THROW(optimizer.launch({context, frame,
                                   slot_of(minimizer::Phase::line_search), 0}),
                 std::runtime_error);
    EXPECT_EQ(frame.span(kAtom_x)[0], 2.5);
    EXPECT_EQ(frame.box.volume(), 1000);
}
