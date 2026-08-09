/// @file cpu/pw/pw_evaluator.cpp
/// @brief PWEvaluator implementation — plane-wave energy (+ analytic forces).

#include "atomiverse/cpu/planewave/pw_evaluator.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/qe_units.h"

#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {

void PWEvaluator::build(const cpu::BuildContext& ctx) {
    method_.build(ctx.frame);
}

void PWEvaluator::launch(const cpu::LaunchContext& ctx) {
    // Nested force/stress SCFs re-resolve Frame keys many times; park the
    // per-launch resolution window around them (same pattern as
    // mean_field_method) so the outer Evaluate budget stays component-sized.
    //
    // Variable-cell productization: stress-only launches use the production
    // SCF inside `pw_stress` for both energy and stress (one SCF class, not
    // energy-SCF + stress-SCF). Force paths still run energy then forces.
    const bool want_forces = compute_forces_ && ctx.forces_requested;
    const bool stress_only = compute_stress_ && !want_forces;

    if (stress_only) {
        const auto saved = ctx.frame.resolution_window();
        double energy_ry = 0.0;
        const auto s = pw_stress(ctx.frame, method_, &energy_ry);
        ctx.frame.restore_resolution_window(saved);
        ctx.frame.meta.get(kEnergy) += energy_ry * qe_units::rydberg_to_ev;
        ctx.frame.meta.get(kEnergyElectronic) +=
            energy_ry * qe_units::rydberg_to_ev;
        // kStress is the EXTENSIVE Voigt virial [eV] (see core/frame_schema.h):
        // pw_stress returns the intensive sigma [eV/A^3], so it is multiplied by
        // the cell volume on the way onto the shared, `+=`-accumulated key.
        const double vol = ctx.frame.box.volume();
        auto& st = ctx.frame.meta.get(kStress);
        for (int k = 0; k < 6; ++k)
            st[static_cast<std::size_t>(k)] +=
                s[static_cast<std::size_t>(k)] * vol;
        return;
    }

    const auto& result = method_.run(ctx.frame);
    ctx.frame.meta.get(kEnergy) += result.energy * qe_units::rydberg_to_ev;
    ctx.frame.meta.get(kEnergyElectronic) +=
        result.energy * qe_units::rydberg_to_ev;

    if (want_forces) {
        const auto saved = ctx.frame.resolution_window();
        const auto forces = pw_forces_analytic(ctx.frame, method_); // eV/Å
        ctx.frame.restore_resolution_window(saved);
        const int n = ctx.frame.nlocal();
        if (static_cast<int>(forces.size()) != n)
            throw std::runtime_error("PWEvaluator: force count mismatch");

        auto fx = ctx.frame.span(kAtom_fx);
        auto fy = ctx.frame.span(kAtom_fy);
        auto fz = ctx.frame.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            fx[i] +=
                static_cast<AccReal>(forces[static_cast<std::size_t>(i)][0]);
            fy[i] +=
                static_cast<AccReal>(forces[static_cast<std::size_t>(i)][1]);
            fz[i] +=
                static_cast<AccReal>(forces[static_cast<std::size_t>(i)][2]);
        }
    }

    if (compute_stress_) {
        const auto saved = ctx.frame.resolution_window();
        const auto s = pw_stress(ctx.frame, method_); // eV/Å³ Voigt
        ctx.frame.restore_resolution_window(saved);
        // Intensive σ → extensive virial for the shared kStress key [eV].
        const double vol = ctx.frame.box.volume();
        auto& st = ctx.frame.meta.get(kStress);
        for (int k = 0; k < 6; ++k)
            st[static_cast<std::size_t>(k)] +=
                s[static_cast<std::size_t>(k)] * vol;
    }
}

} // namespace pw
} // namespace cpu
} // namespace ATV
