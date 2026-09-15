/// @file cpu/dynamics/nve_limit.cpp
/// @brief CPU velocity-Verlet with a per-step displacement (distance)
///        cap. LAMMPS `fix nve/limit` (`lammps/src/fix_nve_limit.cpp`).
///
/// Audit verdict (spec lammps-api-align-02-nve): ALIGNED. The cap below
/// reproduces `fix_nve_limit.cpp` exactly — `vmax = xmax/dt` mirrors
/// `vlimitsq = (xlimit/dtv)^2` (L66), the Euclidean `vsq = v·v` test
/// matches L105/L168, the rescale `s = vmax/sqrt(vsq)` is algebraically
/// identical to LAMMPS `scale = sqrt(vlimitsq/vsq)`, and the cap is
/// applied in BOTH the `initial_integrate` and `final_integrate`
/// branches (matching `FixNVELimit::initial_integrate`/`final_integrate`).

#include "atomiverse/cpu/dynamics/nve_limit.h"

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

NVELimit::NVELimit(double dt, double xmax) : dt_(dt), xmax_(xmax) {}

void NVELimit::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const double vmax = xmax_ / dt_;
    const double vmax_sq = vmax * vmax;
    const double dtf = 0.5 * dt_ * units::force_to_accel;

    // Hoisted once for both branches -- the launch-boundary rule.
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
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const double dtfm = dtf / sp_mass[i];
            sp_vx[i] += dtfm * sp_fx[i];
            sp_vy[i] += dtfm * sp_fy[i];
            sp_vz[i] += dtfm * sp_fz[i];

            const double vsq =
                sp_vx[i] * sp_vx[i] + sp_vy[i] * sp_vy[i] + sp_vz[i] * sp_vz[i];
            if (vsq > vmax_sq) {
                const double s = vmax / std::sqrt(vsq);
                sp_vx[i] *= s;
                sp_vy[i] *= s;
                sp_vz[i] *= s;
            }
            sp_x[i] += dt_ * sp_vx[i];
            sp_y[i] += dt_ * sp_vy[i];
            sp_z[i] += dt_ * sp_vz[i];
        }
        wrap_orthorhombic(f, n);

    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const double dtfm = dtf / sp_mass[i];
            sp_vx[i] += dtfm * sp_fx[i];
            sp_vy[i] += dtfm * sp_fy[i];
            sp_vz[i] += dtfm * sp_fz[i];

            const double vsq =
                sp_vx[i] * sp_vx[i] + sp_vy[i] * sp_vy[i] + sp_vz[i] * sp_vz[i];
            if (vsq > vmax_sq) {
                const double s = vmax / std::sqrt(vsq);
                sp_vx[i] *= s;
                sp_vy[i] *= s;
                sp_vz[i] *= s;
            }
        }
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
