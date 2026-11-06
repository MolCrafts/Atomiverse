#include "atomiverse/core/dynamics_math.h"
/// @file cpu/dynamics/fix_langevin.cpp
/// @brief LAMMPS-equivalent Langevin thermostat (force-additive form).

#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/internal/gauss_stream.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <algorithm>
#include <cmath>

namespace ATV {
namespace cpu {
namespace md {

FixLangevin::FixLangevin(double T_start, double T_stop, double damp, double dt,
                         std::uint32_t seed)
    : T_start_(T_start),
      // A negative `T_stop` is the "no ramp" sentinel — fold it to
      // `T_start` so `compute_target()` sees `T_start == T_stop` and the
      // linear interpolation degenerates to the constant setpoint. This
      // mirrors LAMMPS `fix_langevin.cpp` ctor line `t_target = t_start`
      // (`fix_langevin.cpp:72`) when only one temperature is given.
      T_stop_(T_stop < 0.0 ? T_start : T_stop), damp_(damp), dt_(dt),
      seed_(seed) {
    T_live_ = T_start_;
}

void FixLangevin::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // ── Live setpoint: linear Tstart→Tstop ramp ─────────────────────
    //
    // Mirrors LAMMPS `fix_langevin.cpp::compute_target()`:
    //
    //     double delta = update->ntimestep - update->beginstep;
    //     if (delta != 0.0) delta /= update->endstep - update->beginstep;
    //     t_target = t_start + delta * (t_stop - t_start);
    //     tsqrt    = sqrt(t_target);
    //
    // `ctx.step` is the cumulative MD step index (continuous across
    // chunked `run()` calls); `[nfirst_, nlast_]` is the run span set
    // via `set_run_span()`. `delta` is clamped to `[0,1]` so stepping
    // past `nlast_` holds the setpoint at `T_stop_` rather than
    // extrapolating. With the default span (`nlast_ <= nfirst_`) or
    // `T_stop_ == T_start_`, `T_live_` stays at `T_start_` — the
    // no-ramp default.
    T_live_ =
        ATV::md::ramp_target(T_start_, T_stop_, ctx.step, nfirst_, nlast_);

    // Per-step seed mixed in. Atom identity = global tag, so the same
    // atom on different rank counts gets the same kick (modulo
    // floating-point reduction noise elsewhere).
    // Setup precedes the first production draw. A later re-preparation
    // reuses the preceding completed step, without shifting production RNG.
    const std::int64_t noise_step =
        static_cast<std::int64_t>(ctx.step)
        - (ctx.runtime.exec.initializing_forces ? 1 : 0);
    const std::uint64_t base =
        static_cast<std::uint64_t>(seed_) * 0x9E3779B97F4A7C15ULL
        + static_cast<std::uint64_t>(noise_step) * 0xBF58476D1CE4E5B9ULL;

    // gfactor1 multiplies v -> drag force (temperature-independent, as
    // LAMMPS `gfactor1`).
    // gfactor2_pre * sqrt(m) * gauss -> random force. The temperature
    // enters as `sqrt(T_live_)`, recomputed every step so the ramp is
    // honoured exactly like LAMMPS scaling `gfactor2` by `tsqrt`.
    const double inv_damp_ftm2v = 1.0 / (damp_ * ctx.runtime.units.ftm2v);
    const double gfactor2_pre =
        std::sqrt(2.0 * ctx.runtime.units.boltz * T_live_
                  / (damp_ * dt_ * ctx.runtime.units.mvv2e))
        / ctx.runtime.units.ftm2v;

    // Hoisted once before the parallel region -- never span() inside
    // #pragma omp parallel.
    const TagInt* __restrict sp_tag = f.span(kAtom_tag);
    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    const StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    const StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    const StorageReal* __restrict sp_vz = f.span(kAtom_vz);
    AccReal* __restrict sp_fx = f.span(kAtom_fx);
    AccReal* __restrict sp_fy = f.span(kAtom_fy);
    AccReal* __restrict sp_fz = f.span(kAtom_fz);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        internal::GaussStream g{base
                                + static_cast<std::uint64_t>(sp_tag[i])
                                      * 0xD6E8FEB86659FD93ULL};

        const double m = sp_mass[i];
        const double gfactor1 = -m * inv_damp_ftm2v;
        const double gfactor2 = gfactor2_pre * std::sqrt(m);

        const double rx = g.next();
        const double ry = g.next();
        const double rz = g.next();

        sp_fx[i] += gfactor1 * sp_vx[i] + gfactor2 * rx;
        sp_fy[i] += gfactor1 * sp_vy[i] + gfactor2 * ry;
        sp_fz[i] += gfactor1 * sp_vz[i] + gfactor2 * rz;
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
