/// @file cpu/potentials/bond_harmonic.cpp
/// @brief CPU BondHarmonic — V = K(r − r₀)² implementation.

#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/resolve_view.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

BondHarmonic::BondHarmonic(std::vector<double> K, std::vector<double> r0)
    : K_(std::move(K)), r0_(std::move(r0)) {}

void BondHarmonic::build(const BuildContext& ctx) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    // Per-thread accumulators are sized to the full backing-store capacity
    // (`nmax`); only the live `[0, n_force)` rows are touched per step (a
    // partner row may be a ghost, so the range must span owners + ghosts).
    scratch_.build(n_threads, ctx.frame.capacity());
}

void BondHarmonic::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const auto num_bond = f.span(kAtom_num_bond);
    const auto bond_atom = f.span(kAtom_bond_atom);
    const auto bond_type = f.span(kAtom_bond_type);

    // Single source of truth for owned/ghost counts + bounded tag→local
    // resolution. `n_local` = owners to iterate; `n_force` = owners + ghosts,
    // the extent the per-thread force buffers and the reduction touch (NOT
    // `Frame::n_total()`, which is the global atom count in multi-rank).
    // `resolve()` returns -1 for any tag that is absent or maps outside
    // `[0, n_force)` — a stale ghost partner is dropped, not read as garbage.
    const md::ResolveView rv(ctx);
    const int nlocal = rv.n_local();
    const int n_force = rv.n_force();

    // Tuple cache: tag→local hash resolution happens only when row
    // indices may have moved (migration / halo full rebuild), not every
    // step. Layout: {i, j, type} per bond.
    if (cache_.stale(ctx)) {
        cache_.flat.clear();
        cache_.dropped_at_build = 0;
        for (int i = 0; i < nlocal; ++i) {
            const int nb = num_bond[i];
            for (int m = 0; m < nb; ++m) {
                const int j = rv.resolve(bond_atom(i, m));
                if (j < 0) {
                    ++cache_.dropped_at_build;
                    continue;
                }
                cache_.flat.push_back(i);
                cache_.flat.push_back(j);
                cache_.flat.push_back(bond_type(i, m));
            }
        }
        cache_.mark_built(ctx);
        drops_.dropped += cache_.dropped_at_build;
    }
    const int n_tuples = static_cast<int>(cache_.flat.size()) / 3;
    if (n_tuples == 0)
        return;
    const int* __restrict tup = cache_.flat.data();

    double energy_total = 0;
    md::Voigt6 vir;

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);

    // Resolve the frame's force spans ONCE, before the parallel region opens
    // (never inside it — see BondedForceScratch::bind).
    scratch_.bind(f);

// Stiff bonds contribute significantly to the stress tensor. The per-pair
// virial accumulates into a thread-private `vir_local` and is combined under
// a critical section (once per thread per step) — an OMP user-defined
// reduction on Voigt6 would heap-allocate per step, violating the
// "no allocation in launch()" invariant.
#pragma omp parallel if (scratch_.n_threads() > 1) reduction(+ : energy_total)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const md::ThreadForce tf = scratch_.begin_thread(tid, n_force);
        md::Voigt6 vir_local;

        // Hot loop over pre-resolved tuples (Newton-bond-ON: each bond is
        // owned by its smaller-tag atom, stored once). Partner `j` may be
        // an owner or a ghost; ghost force halves ship home via CommBrick.
#pragma omp for schedule(static)
        for (int t = 0; t < n_tuples; ++t) {
            const int i = tup[3 * t + 0];
            const int j = tup[3 * t + 1];
            const int type = tup[3 * t + 2];

            double dx = x_ptr[i] - x_ptr[j];
            double dy = y_ptr[i] - y_ptr[j];
            double dz = z_ptr[i] - z_ptr[j];
            f.box.min_image(dx, dy, dz);

            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double dr = r - r0_[type];

            energy_total += K_[type] * dr * dr;

            const double fpair = -double(2) * K_[type] * dr / r;
            const double fx_val = fpair * dx;
            const double fy_val = fpair * dy;
            const double fz_val = fpair * dz;

            tf.fx[i] += fx_val;
            tf.fx[j] -= fx_val;
            tf.fy[i] += fy_val;
            tf.fy[j] -= fy_val;
            tf.fz[i] += fz_val;
            tf.fz[j] -= fz_val;

            vir_local.add_pair(dx, dy, dz, fx_val, fy_val, fz_val);
        }
#pragma omp critical
        vir += vir_local;
    }

    scratch_.reduce_into(n_force);

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyBond) += energy_total;
    vir.flush_to(f.meta.get(kStress));

    drops_.evaluated += static_cast<std::uint64_t>(n_tuples);
}

void BondHarmonic::destroy(const DestroyContext& /*ctx*/) {
    drops_.report("BondHarmonic");
    drops_.reset();
    cache_.clear();
    scratch_.clear();
    // Owned-tuple cache is owned by the cpu::Bond base; nothing to clear here.
}

} // namespace cpu
} // namespace ATV
