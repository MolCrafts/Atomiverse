/// @file cpu/md/init.cpp
/// @brief CPU MD infrastructure component implementations.
/// Zero* implementations moved to the backend root (`src/cpu/zero.cpp`).

#include "atomiverse/cpu/md/init.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <algorithm>
#include <cmath>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

// ── KineticEnergy ────────────────────────────────────────────────

void KineticEnergy::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx); // owned only -- ghosts duplicate owners' KE
    double ke = 0;

    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    const StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    const StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    const StorageReal* __restrict sp_vz = f.span(kAtom_vz);
#pragma omp parallel for schedule(static) reduction(+ : ke)
    for (int i = 0; i < n; ++i) {
        double v2 =
            sp_vx[i] * sp_vx[i] + sp_vy[i] * sp_vy[i] + sp_vz[i] * sp_vz[i];
        ke += sp_mass[i] * v2;
    }

    f.meta.get(kKinetic) = double(0.5) * units::mass_v2_to_energy * ke;
}

// ── ZeroMomentum ─────────────────────────────────────────────────

void ZeroMomentum::launch(const cpu::LaunchContext& ctx) {
    if (step_++ % freq_ != 0)
        return;

    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // Hoisted once for both the accumulation loop and the subtraction
    // loop below.
    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    StorageReal* __restrict sp_vz = f.span(kAtom_vz);

    double p[3] = {0.0, 0.0, 0.0};
    double mtot = 0.0;
    for (int i = 0; i < n; ++i) {
        const double m = sp_mass[i];
        p[0] += m * sp_vx[i];
        p[1] += m * sp_vy[i];
        p[2] += m * sp_vz[i];
        mtot += m;
    }

#ifdef ATV_HAS_MPI
    if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        double local[4] = {p[0], p[1], p[2], mtot};
        double global[4];
        MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
        p[0] = global[0];
        p[1] = global[1];
        p[2] = global[2];
        mtot = global[3];
    }
#endif
    if (mtot <= 0.0)
        return;

    const double vx = p[0] / mtot;
    const double vy = p[1] / mtot;
    const double vz = p[2] / mtot;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        sp_vx[i] -= vx;
        sp_vy[i] -= vy;
        sp_vz[i] -= vz;
    }
}

// ── InitVelocity ─────────────────────────────────────────────────

InitVelocity::InitVelocity(double temperature, unsigned seed, VelocityDist dist)
    : temperature_(temperature), seed_(seed), dist_(dist) {}

void InitVelocity::build(const cpu::BuildContext& ctx) {
    // Mirrors LAMMPS `Velocity::create()` (velocity.cpp:160):
    //   1. draw v from `dist` (uniform / gaussian), scale by 1/sqrt(m);
    //   2. remove COM velocity (`zero_momentum()`, velocity.cpp:738);
    //   3. rescale so the measured temperature equals the requested T
    //      (`rescale(t, t_desired)`, velocity.cpp:368).
    //
    // Phase 2: InitVelocity operates on locally-owned atoms only. COM
    // removal is *local* -- for multi-rank usage that requires a
    // globally-zero COM, the caller should set velocities pre-partition
    // on a single rank and let `Backend::set_frame()` partition the
    // already-initialised velocities. The temperature rescale below is
    // MPI-aware (it reduces the global KE) so the measured *global*
    // temperature is exact even though the COM removal stays local. A
    // proper multi-rank InitVelocity with MPI_Allreduce-based global COM
    // is tracked in Phase 3.
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // Hoisted once for the whole function -- every step below (draw,
    // COM removal, rescale) reads/writes the same mass/v columns, and
    // both the gaussian and uniform branches need them.
    const StorageReal* __restrict sp_mass = f.span(kAtom_mass);
    StorageReal* __restrict sp_vx = f.span(kAtom_vx);
    StorageReal* __restrict sp_vy = f.span(kAtom_vy);
    StorageReal* __restrict sp_vz = f.span(kAtom_vz);

    // ── 1. draw velocities from the chosen distribution ──────────────
    // LAMMPS scales the raw draw by `1/sqrt(m)` (velocity.cpp:287-291);
    // the absolute scale is then fixed by the temperature rescale in
    // step 3, so the per-component prefactor only needs to be consistent.
    std::mt19937 rng(seed_);
    if (dist_ == VelocityDist::gaussian) {
        std::normal_distribution<double> draw(0.0, 1.0); // N(0,1)
        for (int i = 0; i < n; ++i) {
            const double s = 1.0 / std::sqrt(sp_mass[i]);
            sp_vx[i] = draw(rng) * s;
            sp_vy[i] = draw(rng) * s;
            sp_vz[i] = draw(rng) * s;
        }
    } else { // VelocityDist::uniform
        // LAMMPS draws `random->uniform() - 0.5` → U(-0.5, 0.5).
        std::uniform_real_distribution<double> draw(-0.5, 0.5);
        for (int i = 0; i < n; ++i) {
            const double s = 1.0 / std::sqrt(sp_mass[i]);
            sp_vx[i] = draw(rng) * s;
            sp_vy[i] = draw(rng) * s;
            sp_vz[i] = draw(rng) * s;
        }
    }

    // ── 2. local COM velocity removal (LAMMPS `mom yes` default) ─────
    double vcm_x = 0, vcm_y = 0, vcm_z = 0, mtot = 0;
    for (int i = 0; i < n; ++i) {
        vcm_x += sp_mass[i] * sp_vx[i];
        vcm_y += sp_mass[i] * sp_vy[i];
        vcm_z += sp_mass[i] * sp_vz[i];
        mtot += sp_mass[i];
    }
    if (mtot > 0) {
        vcm_x /= mtot;
        vcm_y /= mtot;
        vcm_z /= mtot;
        for (int i = 0; i < n; ++i) {
            sp_vx[i] -= vcm_x;
            sp_vy[i] -= vcm_y;
            sp_vz[i] -= vcm_z;
        }
    }

    // ── 3. rescale so the measured temperature equals T exactly ──────
    // LAMMPS `rescale(t_old, t_new)`: factor = sqrt(t_new / t_old).
    // KE = 0.5 * units::mass_v2_to_energy * sum(m v^2); T = 2*KE / (Ndof * kB) with
    // Ndof = 3N - 3 (COM removed), matching `compute temp` extra_dof.
    double sum_mv2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v2 =
            sp_vx[i] * sp_vx[i] + sp_vy[i] * sp_vy[i] + sp_vz[i] * sp_vz[i];
        sum_mv2 += sp_mass[i] * v2;
    }
    long long n_global = n;
#ifdef ATV_HAS_MPI
    if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        double local = sum_mv2, global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
        sum_mv2 = global;
        long long n_local = n, ng = 0;
        MPI_Allreduce(&n_local, &ng, 1, MPI_LONG_LONG, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
        n_global = ng;
    }
#endif
    // Ndof: single owner, constraint-aware. @see CPUDriver::publish_dof_
    const long long ndof = ctx.runtime.md.n_dof_global;
    (void)n_global;
    if (ndof > 0 && sum_mv2 > 0.0) {
        const double ke = 0.5 * units::mass_v2_to_energy * sum_mv2;
        const double t_measured =
            2.0 * ke / (static_cast<double>(ndof) * units::kB_ev_per_k);
        const double factor = std::sqrt(temperature_ / t_measured);
        for (int i = 0; i < n; ++i) {
            sp_vx[i] *= factor;
            sp_vy[i] *= factor;
            sp_vz[i] *= factor;
        }
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
