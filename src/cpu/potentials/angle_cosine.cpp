/// @file cpu/potentials/angle_cosine.cpp
/// @brief CPU AngleCosine — V = K(1 + cos θ).
///
/// Port of LAMMPS `src/MOLECULE/angle_cosine.cpp` (`AngleCosine::compute`),
/// checkout SHA e8c4c278a8. LAMMPS: Copyright (2003) Sandia Corporation,
/// GNU General Public License. Tuple resolution and the threaded force
/// scratch follow `cpu::AngleHarmonic`.

#include "atomiverse/cpu/potentials/angle_cosine.h"
#include "atomiverse/core/angle_cosine.h"
#include "atomiverse/cpu/potentials/resolve_view.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

AngleCosine::AngleCosine(std::vector<double> K) : K_(std::move(K)) {
    if (K_.empty())
        throw std::invalid_argument("AngleCosine: K must be non-empty");
    for (double k : K_)
        if (!std::isfinite(k))
            throw std::invalid_argument("AngleCosine: K must be finite");
}

void AngleCosine::build(const BuildContext& ctx) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (md::ChunkForces* arena = ctx.runtime.md.chunk_forces)
        arena->reserve(n_threads, ctx.frame.capacity());
}

void AngleCosine::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const auto num_angle = f.span(kAtom_num_angle);
    const auto angle_atom = f.span(kAtom_angle_atom);
    const auto angle_type = f.span(kAtom_angle_type);

    const md::ResolveView rv(ctx);
    const int nlocal = rv.n_local();

    // Layout {i, j, k, type} per angle, j = centre/owner (LAMMPS anglelist
    // order i1, i2, i3).
    if (cache_.stale(ctx)) {
        cache_.flat.clear();
        cache_.dropped_at_build = 0;
        for (int j = 0; j < nlocal; ++j) {
            const int na = num_angle[j];
            for (int m = 0; m < na; ++m) {
                const int i = rv.resolve(angle_atom(j, 2 * m + 0));
                const int k = rv.resolve(angle_atom(j, 2 * m + 1));
                if (i < 0 || k < 0) {
                    ++cache_.dropped_at_build;
                    continue;
                }
                cache_.flat.push_back(i);
                cache_.flat.push_back(j);
                cache_.flat.push_back(k);
                cache_.flat.push_back(angle_type(j, m));
            }
        }
        cache_.mark_built(ctx);
        drops_.dropped += cache_.dropped_at_build;
    }
    cache_.require_complete(ctx, "AngleCosine");
    const int n_tuples = static_cast<int>(cache_.flat.size()) / 4;
    if (n_tuples == 0)
        return;
    const int* __restrict tup = cache_.flat.data();
    const int n_types = static_cast<int>(K_.size());
    for (int t = 0; t < n_tuples; ++t) {
        const int type = tup[4 * t + 3];
        if (type < 0 || type >= n_types)
            throw std::runtime_error("AngleCosine: angle type "
                                     + std::to_string(type)
                                     + " outside the K table");
    }

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);

    md::ForcePass forces(ctx);
    const bool do_force = ctx.forces_requested;
    const bool do_virial = ctx.stress_requested;

    // Each chunk keeps its own energy and virial and stores them once;
    // ForcePass sums the chunks in order (deterministic per thread count).
    forces.for_each_chunk(
        n_tuples, [&](int c, int t_begin, int t_end, md::ForceRows tf) {
            double energy_total = 0;
            md::Voigt6 vir_local;

            for (int t = t_begin; t < t_end; ++t) {
                const int i = tup[4 * t + 0];
                const int j = tup[4 * t + 1];
                const int k = tup[4 * t + 2];
                const int type = tup[4 * t + 3];

                double d1[3] = {double(x_ptr[i]) - double(x_ptr[j]),
                                double(y_ptr[i]) - double(y_ptr[j]),
                                double(z_ptr[i]) - double(z_ptr[j])};
                f.box.min_image(d1[0], d1[1], d1[2]);
                double d2[3] = {double(x_ptr[k]) - double(x_ptr[j]),
                                double(y_ptr[k]) - double(y_ptr[j]),
                                double(z_ptr[k]) - double(z_ptr[j])};
                f.box.min_image(d2[0], d2[1], d2[2]);

                const AngleCosineTerm<double> term =
                    angle_cosine_term<double>(K_[type], d1, d2);
                energy_total += term.energy;

                if (do_force) {
                    tf.fx[i] += term.f1[0];
                    tf.fy[i] += term.f1[1];
                    tf.fz[i] += term.f1[2];
                    tf.fx[k] += term.f3[0];
                    tf.fy[k] += term.f3[1];
                    tf.fz[k] += term.f3[2];
                    tf.fx[j] -= term.f1[0] + term.f3[0];
                    tf.fy[j] -= term.f1[1] + term.f3[1];
                    tf.fz[j] -= term.f1[2] + term.f3[2];
                }
                if (do_virial) {
                    vir_local.add_pair(d1[0], d1[1], d1[2], term.f1[0],
                                       term.f1[1], term.f1[2]);
                    vir_local.add_pair(d2[0], d2[1], d2[2], term.f3[0],
                                       term.f3[1], term.f3[2]);
                }
            }
            forces.store(c, energy_total, vir_local);
        });
    const double energy_total = forces.energy();
    const md::Voigt6 vir = forces.virial();

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyAngle) += energy_total;
    if (do_virial)
        vir.flush_to(f.meta.get(kStress));

    drops_.evaluated += static_cast<std::uint64_t>(n_tuples);
}

void AngleCosine::destroy(const DestroyContext& /*ctx*/) {
    drops_.report("AngleCosine");
    drops_.reset();
    cache_.clear();
}

} // namespace cpu
} // namespace ATV
