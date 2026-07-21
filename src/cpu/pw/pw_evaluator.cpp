/// @file cpu/pw/pw_evaluator.cpp
/// @brief PWEvaluator implementation — plane-wave energy (+ analytic forces).

#include "atomiverse/cpu/pw/pw_evaluator.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/pw/qe_units.h"

#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {

void PWEvaluator::build(const cpu::BuildContext& ctx) {
    solver_.build(ctx.frame);
}

void PWEvaluator::launch(const cpu::LaunchContext& ctx) {
    // Energy path (always). When forces are requested, analytic forces run
    // their own SCF + gradient (one SCF total in the force path); skip the
    // duplicate energy-only SCF by letting forces supply energy later if needed.
    // For simplicity keep energy SCF then analytic forces (2 SCF) until
    // workspace reuse lands — still O(1) vs FD's 6N.
    const auto& result = solver_.run(ctx.frame);
    ctx.frame.meta.get(kEnergy) = result.energy * qe_units::rydberg_to_ev;

    if (compute_forces_ && ctx.forces_requested) {
        const auto forces =
            pw_forces_analytic(ctx.frame, solver_.config()); // eV/Å
        const int n = ctx.frame.nlocal();
        if (static_cast<int>(forces.size()) != n)
            throw std::runtime_error("PWEvaluator: force count mismatch");

        auto fx = ctx.frame.span(kAtom_fx);
        auto fy = ctx.frame.span(kAtom_fy);
        auto fz = ctx.frame.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            fx[i] =
                static_cast<AccReal>(forces[static_cast<std::size_t>(i)][0]);
            fy[i] =
                static_cast<AccReal>(forces[static_cast<std::size_t>(i)][1]);
            fz[i] =
                static_cast<AccReal>(forces[static_cast<std::size_t>(i)][2]);
        }
    }

    if (compute_stress_) {
        const auto s = pw_stress(ctx.frame, solver_.config()); // eV/Å³ Voigt
        auto& st = ctx.frame.meta.get(kStress);
        for (int k = 0; k < 6; ++k)
            st[static_cast<std::size_t>(k)] = s[static_cast<std::size_t>(k)];
    }
}

} // namespace pw
} // namespace cpu
} // namespace ATV
