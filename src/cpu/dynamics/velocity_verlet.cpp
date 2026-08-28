/// @file cpu/md/velocity_verlet.cpp
/// @brief CPU VelocityVerlet implementation with OpenMP.

#include "atomiverse/cpu/dynamics/velocity_verlet.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/pbc.h"

#include <cmath>

namespace ATV {
namespace cpu {
namespace md {

// ── VelocityVerlet ───────────────────────────────────────────────

VelocityVerlet::VelocityVerlet(double dt) : dt_(dt) {}

void VelocityVerlet::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // Hoisted once for both branches (initial_integrate uses all seven;
    // final_integrate uses only mass/v/f) -- the launch-boundary rule.
    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    StorageReal* __restrict sp_vz = f.span(kAtom_vz);
    const AccReal* __restrict sp_fx = f.span(kAtom_fx);
    const AccReal* __restrict sp_fy = f.span(kAtom_fy);
    const AccReal* __restrict sp_fz = f.span(kAtom_fz);
    StorageReal* __restrict sp_x = f.span(kAtom_x);
    StorageReal* __restrict sp_y = f.span(kAtom_y);
    StorageReal* __restrict sp_z = f.span(kAtom_z);

    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        const double dtf = double(0.5) * dt_ * units::force_to_accel;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double dtfm = dtf / sp_mass[i];
            sp_vx[i] += dtfm * sp_fx[i];
            sp_vy[i] += dtfm * sp_fy[i];
            sp_vz[i] += dtfm * sp_fz[i];
            sp_x[i] += dt_ * sp_vx[i];
            sp_y[i] += dt_ * sp_vy[i];
            sp_z[i] += dt_ * sp_vz[i];
        }

        wrap_orthorhombic(f, n);

    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
        const double dtf = double(0.5) * dt_ * units::force_to_accel;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double dtfm = dtf / sp_mass[i];
            sp_vx[i] += dtfm * sp_fx[i];
            sp_vy[i] += dtfm * sp_fy[i];
            sp_vz[i] += dtfm * sp_fz[i];
        }
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
