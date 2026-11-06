/// @file cpu/potentials/angle_harmonic.cpp
/// @brief CPU AngleHarmonic — V = K(θ − θ₀)² implementation.

#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/resolve_view.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

AngleHarmonic::AngleHarmonic(std::vector<double> K, std::vector<double> theta0)
    : K_(std::move(K)), theta0_(std::move(theta0)) {}

void AngleHarmonic::build(const BuildContext& ctx) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (md::ChunkForces* arena = ctx.runtime.md.chunk_forces)
        arena->reserve(n_threads, ctx.frame.capacity());
}

void AngleHarmonic::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const auto num_angle = f.span(kAtom_num_angle);
    const auto angle_atom = f.span(kAtom_angle_atom);
    const auto angle_type = f.span(kAtom_angle_type);

    const md::ResolveView rv(ctx);
    const int nlocal = rv.n_local();

    // Tuple cache: tag→local hash resolution happens only when row
    // indices may have moved (migration / halo full rebuild), not every
    // step. Layout: {i, j, k, type} per angle (j = centre/owner).
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
    cache_.require_complete(ctx, "AngleHarmonic");
    const int n_tuples = static_cast<int>(cache_.flat.size()) / 4;
    if (n_tuples == 0)
        return;
    const int* __restrict tup = cache_.flat.data();

    // Angle virial: Σ d_ji ⊗ F_i + d_jk ⊗ F_k (centre force cancels by
    // Newton's 3rd, so this is translation-invariant and PBC-safe).

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);

    // Open this launch's chunks of the Runtime's accumulators once,
    // before the parallel region (see md::ForcePass).
    md::ForcePass forces(ctx);
    // The virial is only flushed when stress is requested; skip the two
    // outer products per angle otherwise (LAMMPS vflag).
    const bool do_virial = ctx.stress_requested;

    // Each chunk keeps its own energy and virial and stores them once;
    // ForcePass sums the chunks in order (deterministic per thread count).
    forces.for_each_chunk(n_tuples, [&](int c, int t_begin, int t_end,
                                        md::ForceRows tf) {
        double energy_total = 0;
        md::Voigt6 vir_local;

        // Hot loop over pre-resolved tuples: centre atom `j` owns the
        // angle; partners (i, k) may be owners or ghosts.
        for (int t = t_begin; t < t_end; ++t) {
            {
                const int i = tup[4 * t + 0];
                const int j = tup[4 * t + 1];
                const int k = tup[4 * t + 2];
                const int type = tup[4 * t + 3];

                const double ri[3] = {x_ptr[i], y_ptr[i], z_ptr[i]};
                const double rj[3] = {x_ptr[j], y_ptr[j], z_ptr[j]};
                const double rk[3] = {x_ptr[k], y_ptr[k], z_ptr[k]};

                double dx_ji = ri[0] - rj[0];
                double dy_ji = ri[1] - rj[1];
                double dz_ji = ri[2] - rj[2];
                f.box.min_image(dx_ji, dy_ji, dz_ji);

                double dx_jk = rk[0] - rj[0];
                double dy_jk = rk[1] - rj[1];
                double dz_jk = rk[2] - rj[2];
                f.box.min_image(dx_jk, dy_jk, dz_jk);

                const double r_ji =
                    std::sqrt(dx_ji * dx_ji + dy_ji * dy_ji + dz_ji * dz_ji);
                const double r_jk =
                    std::sqrt(dx_jk * dx_jk + dy_jk * dy_jk + dz_jk * dz_jk);

                double cos_theta =
                    (dx_ji * dx_jk + dy_ji * dy_jk + dz_ji * dz_jk)
                    / (r_ji * r_jk);
                cos_theta =
                    std::max(double(-1), std::min(double(1), cos_theta));
                const double theta = std::acos(cos_theta);

                const double dtheta = theta - theta0_[type];
                energy_total += K_[type] * dtheta * dtheta;

                const double dV = double(2) * K_[type] * dtheta;

                const double sin_theta =
                    std::sqrt(double(1) - cos_theta * cos_theta);
                if (sin_theta < double(1e-10))
                    continue;

                const double a_coeff = dV / (r_ji * sin_theta);
                const double c_coeff = dV / (r_jk * sin_theta);

                const double fi_x =
                    a_coeff * (dx_jk / r_jk - cos_theta * dx_ji / r_ji);
                const double fi_y =
                    a_coeff * (dy_jk / r_jk - cos_theta * dy_ji / r_ji);
                const double fi_z =
                    a_coeff * (dz_jk / r_jk - cos_theta * dz_ji / r_ji);

                const double fk_x =
                    c_coeff * (dx_ji / r_ji - cos_theta * dx_jk / r_jk);
                const double fk_y =
                    c_coeff * (dy_ji / r_ji - cos_theta * dy_jk / r_jk);
                const double fk_z =
                    c_coeff * (dz_ji / r_ji - cos_theta * dz_jk / r_jk);

                // Force on j: -(fi + fk) (Newton's 3rd law).
                if (ctx.forces_requested) {
                    tf.fx[i] += fi_x;
                    tf.fy[i] += fi_y;
                    tf.fz[i] += fi_z;
                    tf.fx[k] += fk_x;
                    tf.fy[k] += fk_y;
                    tf.fz[k] += fk_z;
                    tf.fx[j] -= fi_x + fk_x;
                    tf.fy[j] -= fi_y + fk_y;
                    tf.fz[j] -= fi_z + fk_z;
                }

                if (do_virial) {
                    vir_local.add_pair(dx_ji, dy_ji, dz_ji, fi_x, fi_y, fi_z);
                    vir_local.add_pair(dx_jk, dy_jk, dz_jk, fk_x, fk_y, fk_z);
                }
            }
        }
        forces.store(c, energy_total, vir_local);
    });
    const double energy_total = forces.energy();
    const md::Voigt6 vir = forces.virial();

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyAngle) += energy_total;
    if (ctx.stress_requested)
        vir.flush_to(f.meta.get(kStress));

    drops_.evaluated += static_cast<std::uint64_t>(n_tuples);
}

void AngleHarmonic::destroy(const DestroyContext& /*ctx*/) {
    drops_.report("AngleHarmonic");
    drops_.reset();
    cache_.clear();
    // Owned-tuple cache is owned by the cpu::Angle base.
}

} // namespace cpu
} // namespace ATV
