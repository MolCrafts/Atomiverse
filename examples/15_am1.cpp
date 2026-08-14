/// @file examples/15_am1.cpp
/// @brief AM1 semiempirical single-point, force check, and short Minimize.
///
/// Teaches: Semiempirical kind + AM1Evaluator (no SE Driver); energy, ΔHf,
/// nuclear forces (a central finite difference of the energy at fixed
/// converged density); geometry step on Minimize. See docs/tutorial/ch17.
///
/// Build: -DATV_BUILD_EXAMPLES=ON (no RHF/PW gate — AM1 is always in the CPU
/// library).

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"

namespace {

ATV::cpu::Frame make_water() {
    ATV::cpu::Frame f(3);
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_x)[1] = 1.05; // slightly stretched
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_x)[2] = -0.26;
    f.span(ATV::kAtom_y)[2] = 1.0;
    f.span(ATV::kAtom_z)[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        f.span(ATV::kAtom_tag)[i] = i + 1;
        f.span(ATV::kAtom_mass)[i] =
            (f.span(ATV::kAtom_type)[i] == 8) ? 15.999 : 1.008;
    }
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

double max_force(const ATV::cpu::Frame& f) {
    double m = 0.0;
    for (int i = 0; i < f.nlocal(); ++i) {
        m = std::max(m, std::hypot(f.span(ATV::kAtom_fx)[i],
                                   f.span(ATV::kAtom_fy)[i],
                                   f.span(ATV::kAtom_fz)[i]));
    }
    return m;
}

} // namespace

int main() {
    using namespace ATV;
    log_init(LogConfig{.level = spdlog::level::info});
    ATV_INFO("=== Atomiverse: AM1 water (energy, ΔHf, forces, Minimize) ===");

    auto frame = make_water();

    // ── 1. Low-level solver: energy + heat of formation ───────────────
    {
        cpu::semiempirical::AM1Solver solver;
        solver.build(frame);
        const auto& r = solver.run(frame);
        ATV_INFO("AM1Solver: converged={}  E={:.6f} eV  ΔHf={:.3f} kcal/mol  "
                 "iters={}",
                 r.converged, r.total_energy_ev, r.heat_of_formation_kcal_mol,
                 r.iterations);
    }

    // ── 2. Evaluate + forces (public Component path) ──────────────────
    {
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::semiempirical::AM1Evaluator{}.with_gradient(true));
        eval.build();
        eval.run(frame);
        ATV_INFO("Evaluate: E={:.6f} eV  max|F|={:.4f} eV/Å  q_O={:.4f}",
                 frame.meta.get(kEnergy), max_force(frame),
                 frame.span(kAtom_charge)[0]);
    }

    // ── 3. Geometry optimization on the same forces ───────────────────
    {
        const double e0 = frame.meta.get(kEnergy);
        const double f0 = max_force(frame);

        cpu::minimizer::Minimize opt;
        opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.02, 8});
        opt.add_criterion(cpu::ForceConverged{0.5, 0.3, 0.5, 0.3, 1e-3});
        opt.on(minimizer::Phase::force)
            .add(cpu::semiempirical::AM1Evaluator{}.with_gradient(true));
        opt.build();
        opt.run(frame, 20);

        // Re-evaluate for final E/F on Frame.
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::semiempirical::AM1Evaluator{}.with_gradient(true));
        eval.build();
        eval.run(frame);

        ATV_INFO("Minimize: E {:.6f} → {:.6f} eV  max|F| {:.4f} → {:.4f}  "
                 "converged={}  reason={}",
                 e0, frame.meta.get(kEnergy), f0, max_force(frame),
                 opt.converged(), opt.stop_reason());
    }

    ATV_INFO("Done. Tutorial: docs/tutorial/ch17-semiempirical-am1.md");
    return 0;
}
