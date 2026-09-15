/// @file cpu/dynamics/fix_langevin.cpp
/// @brief LAMMPS-equivalent Langevin thermostat (force-additive form).

#include "atomiverse/cpu/dynamics/fix_langevin.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <algorithm>
#include <cmath>

namespace ATV {
namespace cpu {
namespace md {

namespace {

/// @brief Per-atom Gaussian stream with Box-Muller pair caching.
///
/// Deliberately constructed once per atom inside the `launch()` loop so
/// each atom's kick is seeded from `(step, tag)` — two ranks that own
/// the same global atom (different row indices, same tag) produce the
/// same random force, preserving cross-rank reproducibility without
/// needing an MPI reduction. The construction cost is a single
/// `uint64_t` store on the stack; `next()`'s `log / cos / sqrt` triple
/// dominates per-atom cost.
///
/// Pair caching: a single Box-Muller call returns two IID gaussians
/// (the `sin` and `cos` partners); we consume one and stash the other,
/// halving the Box-Muller invocations per atom (3 gaussians needed ⇒
/// 2 invocations instead of 3). A dedicated HPC pass can drop the
/// `log / cos` entirely by switching to Marsaglia polar; keep the
/// current form here so the statistical properties stay identical to
/// the 12 existing Langevin tests.
struct GaussStream {
    std::uint64_t s;
    bool has_cached = false;
    double cached = 0.0;
    double next_u01() {
        s += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    }
    double next() {
        if (has_cached) {
            has_cached = false;
            return cached;
        }
        const double u = next_u01();
        const double v = next_u01();
        const double r = std::sqrt(-2.0 * std::log(u > 1e-300 ? u : 1e-300));
        const double theta = 2.0 * ATV::units::pi * v;
        cached = r * std::sin(theta);
        has_cached = true;
        return r * std::cos(theta);
    }
};

} // namespace

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
    double delta = 0.0;
    if (nlast_ > nfirst_) {
        delta = static_cast<double>(ctx.step - nfirst_)
                / static_cast<double>(nlast_ - nfirst_);
        delta = std::clamp(delta, 0.0, 1.0);
    }
    T_live_ = T_start_ + delta * (T_stop_ - T_start_);

    // Per-step seed mixed in. Atom identity = global tag, so the same
    // atom on different rank counts gets the same kick (modulo
    // floating-point reduction noise elsewhere).
    const std::uint64_t base =
        static_cast<std::uint64_t>(seed_) * 0x9E3779B97F4A7C15ULL
        + static_cast<std::uint64_t>(ctx.step) * 0xBF58476D1CE4E5B9ULL;

    // gfactor1 multiplies v -> drag force (temperature-independent, as
    // LAMMPS `gfactor1`).
    // gfactor2_pre * sqrt(m) * gauss -> random force. The temperature
    // enters as `sqrt(T_live_)`, recomputed every step so the ramp is
    // honoured exactly like LAMMPS scaling `gfactor2` by `tsqrt`.
    const double inv_damp_ftm2v = 1.0 / (damp_ * units::force_to_accel);
    const double gfactor2_pre =
        std::sqrt(2.0 * units::kB_ev_per_k * T_live_
                  / (damp_ * dt_ * units::mass_v2_to_energy))
        / units::force_to_accel;

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
        GaussStream g{base
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
