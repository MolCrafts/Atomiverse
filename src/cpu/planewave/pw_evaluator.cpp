/// @file cpu/planewave/pw_evaluator.cpp
/// @brief PWEvaluator implementation — plane-wave energy (+ analytic forces).

#include "atomiverse/cpu/planewave/pw_evaluator.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/planewave/qe_units.h"

#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {

void PWEvaluator::build(const cpu::BuildContext& ctx) {
    method_.build(ctx.frame);
}

void PWEvaluator::launch(const cpu::LaunchContext& ctx) {
    const bool want_forces = compute_forces_ && ctx.forces_requested;
    const bool want_stress = compute_stress_ && ctx.stress_requested;
    std::vector<std::array<double, 3>> forces;
    std::array<double, 6> stress{};
    {
        // Nested method accesses are independent of the component access budget.
        struct RestoreWindow {
            Frame& frame;
            ResolutionWindow saved;
            ~RestoreWindow() { frame.restore_resolution_window(saved); }
        } restore{ctx.frame, ctx.frame.resolution_window()};
        ctx.frame.restore_resolution_window(ResolutionWindow{});
        method_.evaluate_(ctx.frame, want_forces ? &forces : nullptr,
                          want_stress ? &stress : nullptr);
    }
    const double energy =
        method_.last_result().energy * qe_units::rydberg_to_ev;
    ctx.frame.meta.get(kEnergy) += energy;
    ctx.frame.meta.get(kEnergyElectronic) += energy;
    if (want_forces) {
        const int n = ctx.frame.nlocal();
        if (static_cast<int>(forces.size()) != n)
            throw std::runtime_error("PWEvaluator: force count mismatch");
        auto fx = ctx.frame.span(kAtom_fx);
        auto fy = ctx.frame.span(kAtom_fy);
        auto fz = ctx.frame.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            fx[i] += static_cast<AccReal>(forces[i][0]);
            fy[i] += static_cast<AccReal>(forces[i][1]);
            fz[i] += static_cast<AccReal>(forces[i][2]);
        }
    }
    if (want_stress) {
        const double vol = ctx.frame.box.volume();
        auto& st = ctx.frame.meta.get(kStress);
        for (int k = 0; k < 6; ++k)
            st[k] += stress[k] * vol;
    }
}

} // namespace pw
} // namespace cpu
} // namespace ATV
