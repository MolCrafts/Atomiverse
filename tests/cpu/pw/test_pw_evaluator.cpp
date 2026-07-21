/// @file test_pw_evaluator.cpp
/// @brief PWEvaluator — Evaluate-driver integration + PWSolver parity.
///
/// Verifies the plane-wave SCF is reachable through the canonical
/// `cpu::Evaluate + Evaluator-component` shape (taxonomy rule: SCF is an
/// inner solver behind an Evaluator, never an outer driver), and that
/// the component path reproduces the direct solver path bit-for-bit.
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/pw_evaluator.h"
#include "atomiverse/cpu/pw/pw_solver.h"
#include "atomiverse/cpu/pw/qe_units.h"
#include "atomiverse/facade/stage.h"
#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

using namespace ATV;
using namespace ATV::cpu::pw;

// Taxonomy guards: PWSolver is an inner solver (plain class), PWEvaluator
// is the Plan-facing component.
static_assert(!std::is_base_of_v<cpu::Component, PWSolver>,
              "PWSolver must not be a Plan component");
static_assert(std::is_base_of_v<cpu::Component, PWEvaluator>,
              "PWEvaluator must be a cpu::Component");

namespace {
cpu::Frame make_si_fcc(double a) {
    cpu::Frame f(2);
    f.span(ATV::kAtom_type)[0] = 14;
    f.span(ATV::kAtom_type)[1] = 14;
    f.span(ATV::kAtom_mass)[0] = 28.0855;
    f.span(ATV::kAtom_mass)[1] = 28.0855;
    f.span(ATV::kAtom_x)[0] = 0;
    f.span(ATV::kAtom_y)[0] = 0;
    f.span(ATV::kAtom_z)[0] = 0;
    f.span(ATV::kAtom_x)[1] = a / 4.0;
    f.span(ATV::kAtom_y)[1] = a / 4.0;
    f.span(ATV::kAtom_z)[1] = a / 4.0;
    double h = a / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    f.simbox.set_cell(cell);
    return f;
}

PWConfig small_config() {
    PWConfig cfg;
    cfg.ecut_ry = 10.0;
    cfg.k_mesh = 2;
    cfg.tol = 1e-6;
    cfg.max_iter = 20;
    return cfg;
}
} // namespace

TEST(PWEvaluator, IsPlaneWaveKind) {
    PWEvaluator pwe{small_config()};
    EXPECT_NE(dynamic_cast<const cpu::PlaneWave*>(&pwe), nullptr);
    EXPECT_EQ(pwe.role(), 0); // main-loop required
}

TEST(PWEvaluator, EvaluateDriverWritesFrameEnergyInEv) {
    auto frame = make_si_fcc(5.43);

    PWEvaluator pwe{small_config()};
    cpu::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(pwe);
    eval.build();
    eval.run(frame);

    const auto& r = pwe.last_result();
    EXPECT_TRUE(std::isfinite(r.energy));
    EXPECT_NEAR(r.energy_ha, r.energy * qe_units::rydberg_to_hartree, 1e-12);
    // frame.energy follows the project convention (eV).
    EXPECT_NEAR(frame.meta.get(ATV::kEnergy),
                r.energy * qe_units::rydberg_to_ev, 1e-9);
}

TEST(PWEvaluator, MatchesDirectSolver) {
    auto cfg = small_config();

    auto frame_solver = make_si_fcc(5.43);
    PWSolver solver{cfg};
    solver.build(frame_solver);
    const PWResult r_solver = solver.run(frame_solver);

    auto frame_eval = make_si_fcc(5.43);
    PWEvaluator pwe{cfg};
    cpu::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(pwe);
    eval.build();
    eval.run(frame_eval);

    EXPECT_TRUE(std::isfinite(r_solver.energy));
    EXPECT_NEAR(pwe.last_result().energy, r_solver.energy, 1e-10)
        << "Evaluate + PWEvaluator must reproduce the direct PWSolver";
    EXPECT_EQ(pwe.last_result().iterations, r_solver.iterations);
}
