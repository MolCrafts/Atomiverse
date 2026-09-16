/// @file tests/cpu/minimize/test_line_search_nested_node.cpp
/// @brief Unit: an L-BFGS line search re-enters a NESTED force source.
///
/// Mirrors `src/cpu/minimize/lbfgs.cpp` (the `line_search` phase) and the
/// replay it depends on, `Runtime::run_stage` → `Runtime::run_slot_`.
/// Spec: `.claude/specs/esmm-contract-03-runslot-nested` — ac-002 (U5).
///
/// `Minimize` publishes `reeval_forces` (`minimize.h`), and StrongWolfe — the
/// DEFAULT mode for LBFGS and CG — calls it once per trial geometry to read
/// phi(alpha) and phi'(alpha) back off the Frame. The replay walks the force
/// slot: Zero* → evaluator → reduce. A nested method node (`Driver<scf::Phase>`
/// + the pure `HF` kind tag, the shape `cpu::scf::RHF` presents) sits in that
/// slot as a driver op, and `run_slot_` skipped driver ops — so each trial saw
/// the zeroed frame, E = 0 and f = 0, and the line search was degenerate.
///
/// The witness is a call counter on the node: with a working replay the node
/// is entered once per iteration (from `Runtime::step`) PLUS once per trial.
/// The physics is a closed-form harmonic well, so the case is a unit, not an
/// SCF (no e2e in `tests/`).

#include <cmath>
#include <utility>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/line_search.h"
#include "atomiverse/cpu/minimize/minimize.h"

using namespace ATV;

namespace {

/// Anisotropic spring constants [eV/Å²]. Spread so the 6-dimensional
/// quadratic is not solved by one exact step and the loop keeps line-searching.
constexpr double kSpring[3] = {1.0, 2.5, 4.0};

/// Well centres [Å], one per atom.
constexpr double kCentre[2][3] = {{0.0, 0.0, 0.0}, {1.2, 0.0, 0.0}};

/// @brief A nested method node with closed-form physics.
///
/// Node shape only — `Driver<scf::Phase>` + `cpu::HF`, never a `Component` —
/// so `Minimize::validate_components` credits it as the force source and the
/// Plan holds it as a driver op. `run_nested_erased` is the single entry point
/// for BOTH dispatch paths (`Runtime::step` and `Runtime::run_slot_`), so
/// `calls` counts force evaluations exactly.
///
/// Energy [eV] and force [eV/Å] of an anisotropic harmonic well:
///   E = ½ Σ_i Σ_d k_d (r_id − c_id)²,   f_id = −k_d (r_id − c_id).
class HarmonicMethodNode : public cpu::Driver<ATV::scf::Phase>,
                           public cpu::HF {
  public:
    int calls = 0;

    int run_nested_erased(cpu::Context&, cpu::Frame& f, int /*budget*/) final {
        ++calls;
        // Park the resolution window like a real method node does
        // (`MeanFieldMethod::run`): a driver op is dispatched OUTSIDE the
        // per-component window, and a line-search trial re-enters it from
        // inside the optimizer's own window.
        const auto saved = f.resolution_window();
        const int n = f.nlocal();
        StorageReal* x = f.span(kAtom_x);
        StorageReal* y = f.span(kAtom_y);
        StorageReal* z = f.span(kAtom_z);
        AccReal* fx = f.span(kAtom_fx);
        AccReal* fy = f.span(kAtom_fy);
        AccReal* fz = f.span(kAtom_fz);
        double e = 0;
        for (int i = 0; i < n && i < 2; ++i) {
            const double dx = static_cast<double>(x[i]) - kCentre[i][0];
            const double dy = static_cast<double>(y[i]) - kCentre[i][1];
            const double dz = static_cast<double>(z[i]) - kCentre[i][2];
            e += 0.5
                 * (kSpring[0] * dx * dx + kSpring[1] * dy * dy
                    + kSpring[2] * dz * dz);
            // ZeroForces ran first in this slot (Minimize binds it at -1000),
            // on the replay path as well — assignment is the whole force.
            fx[i] = -kSpring[0] * dx;
            fy[i] = -kSpring[1] * dy;
            fz[i] = -kSpring[2] * dz;
        }
        f.meta.get(kEnergy) = e;
        f.restore_resolution_window(saved);
        return 1;
    }

  protected:
    void validate_components() const override {}
};

/// @brief Two atoms displaced off their well centres.
cpu::Frame displaced_pair() {
    cpu::Frame f(2);
    const double x0[2][3] = {{0.40, 0.30, -0.20}, {1.65, -0.25, 0.35}};
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = x0[i][0];
        f.span(kAtom_y)[i] = x0[i][1];
        f.span(kAtom_z)[i] = x0[i][2];
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 1;
    }
    f.box = cpu::Box::cubic(20.0);
    return f;
}

/// Tolerances far below anything this well reaches in four iterations, so the
/// loop always spends its full budget and the iteration count is a fixed
/// reference for the call count.
cpu::ForceConverged never_converges() {
    return cpu::ForceConverged{1e-12, 1e-12, 1e-12, 1e-12, 1e-14};
}

} // namespace

// ── ac-002: StrongWolfe trials re-enter the nested node ────────────────────
//
// RED today: `calls == iterations` — only `Runtime::step` dispatches the node,
// and every trial evaluation replayed through `run_stage` saw E = 0, f = 0.
TEST(MinimizeNestedNode, StrongWolfeTrialsReenterNestedNode) {
    cpu::Frame frame = displaced_pair();

    HarmonicMethodNode method;
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{}); // default mode: StrongWolfe
    opt.on(ATV::minimizer::Phase::force)
        .add_ref(method, /*order=*/0, /*budget=*/1);
    opt.add_criterion(never_converges());
    opt.build();

    const int iterations = opt.run(frame, /*max_iter=*/4);

    ASSERT_EQ(iterations, 4) << "the loop stopped early; the call-count "
                                "reference is no longer the full budget";
    EXPECT_GT(method.calls, iterations)
        << "line-search trials never re-entered the nested force source "
           "(calls=" << method.calls << ", iterations=" << iterations
        << ") — Runtime::run_slot_ skips nested driver nodes, so every "
           "StrongWolfe trial evaluated a zeroed frame";
}

// ── Control (green today and after the fix) ────────────────────────────────
//
// `LineSearchMode::Fixed` takes the step without evaluating any trial, so the
// node is entered exactly once per iteration. This is what makes the case
// above a measurement of TRIALS and not of some second dispatch of the step.
TEST(MinimizeNestedNode, FixedStepEntersTheNodeOncePerIteration) {
    cpu::Frame frame = displaced_pair();

    HarmonicMethodNode method;
    cpu::minimizer::LBFGS fixed;
    fixed.line_search(cpu::minimizer::LineSearchMode::Fixed);

    cpu::minimizer::Minimize opt;
    opt.set_optimizer(std::move(fixed));
    opt.on(ATV::minimizer::Phase::force)
        .add_ref(method, /*order=*/0, /*budget=*/1);
    opt.add_criterion(never_converges());
    opt.build();

    const int iterations = opt.run(frame, /*max_iter=*/4);

    ASSERT_EQ(iterations, 4);
    EXPECT_EQ(method.calls, iterations)
        << "Fixed mode evaluates no trial geometry: the only dispatch of the "
           "nested node is the one Runtime::step makes per iteration";
}
