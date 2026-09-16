/// @file cpu/potentials/improper_harmonic.cpp
/// @brief CPU ImproperHarmonic — V = K(χ − χ₀)².

#include "atomiverse/cpu/potentials/improper_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_geom.h"
#include "atomiverse/cpu/potentials/resolve_view.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

ImproperHarmonic::ImproperHarmonic(std::vector<double> K,
                                   std::vector<double> chi0)
    : K_(std::move(K)), chi0_(std::move(chi0)) {
    if (K_.empty() || K_.size() != chi0_.size()) {
        throw std::invalid_argument(
            "ImproperHarmonic: K and chi0 must be non-empty and same size");
    }
    for (std::size_t t = 0; t < K_.size(); ++t) {
        if (!std::isfinite(K_[t]) || !std::isfinite(chi0_[t])) {
            throw std::invalid_argument(
                "ImproperHarmonic: non-finite K or chi0 at type "
                + std::to_string(t));
        }
    }
}

void ImproperHarmonic::build(const BuildContext& ctx) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    scratch_.build(n_threads, ctx.frame.capacity());
}

void ImproperHarmonic::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const auto num_improper = f.span(kAtom_num_improper);
    const auto improper_atom = f.span(kAtom_improper_atom);
    const auto improper_type = f.span(kAtom_improper_type);

    const md::ResolveView rv(ctx);
    const int nlocal = rv.n_local();
    const int n_force = rv.n_force();

    // Same layout as dihedrals: {i1, i2, i3, i4, type}; owner = i2 (central).
    if (cache_.stale(ctx)) {
        cache_.flat.clear();
        cache_.dropped_at_build = 0;
        for (int i2 = 0; i2 < nlocal; ++i2) {
            const int ni = num_improper[i2];
            for (int m = 0; m < ni; ++m) {
                const int i1 = rv.resolve(improper_atom(i2, 3 * m + 0));
                const int i3 = rv.resolve(improper_atom(i2, 3 * m + 1));
                const int i4 = rv.resolve(improper_atom(i2, 3 * m + 2));
                if (i1 < 0 || i3 < 0 || i4 < 0) {
                    ++cache_.dropped_at_build;
                    continue;
                }
                cache_.flat.push_back(i1);
                cache_.flat.push_back(i2);
                cache_.flat.push_back(i3);
                cache_.flat.push_back(i4);
                cache_.flat.push_back(improper_type(i2, m));
            }
        }
        cache_.mark_built(ctx);
        drops_.dropped += cache_.dropped_at_build;
    }
    const int n_tuples = static_cast<int>(cache_.flat.size()) / 5;
    if (n_tuples == 0)
        return;
    const int* __restrict tup = cache_.flat.data();

    double energy_total = 0;
    md::Voigt6 vir;

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);

    scratch_.bind(f);

#pragma omp parallel reduction(+ : energy_total)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const md::ThreadForce tf = scratch_.begin_thread(tid, n_force);
        md::Voigt6 vir_local;

#pragma omp for schedule(static)
        for (int t = 0; t < n_tuples; ++t) {
            const int i1 = tup[5 * t + 0];
            const int i2 = tup[5 * t + 1];
            const int i3 = tup[5 * t + 2];
            const int i4 = tup[5 * t + 3];
            const int type = tup[5 * t + 4];
            if (type < 0 || static_cast<std::size_t>(type) >= K_.size())
                continue;

            dihedral_geom::Geometry g;
            const bool ok = dihedral_geom::fill(
                x_ptr[i1], y_ptr[i1], z_ptr[i1], x_ptr[i2], y_ptr[i2],
                z_ptr[i2], x_ptr[i3], y_ptr[i3], z_ptr[i3], x_ptr[i4],
                y_ptr[i4], z_ptr[i4],
                [&](double& ax, double& ay, double& az) {
                    f.box.min_image(ax, ay, az);
                },
                g);
            if (!ok)
                continue;

            const double K = K_[static_cast<std::size_t>(type)];
            const double chi0 = chi0_[static_cast<std::size_t>(type)];
            const double chi = std::atan2(g.sin_phi, g.cos_phi);
            const double dchi = chi - chi0;
            const double energy = K * dchi * dchi;
            energy_total += energy;

            // dE/dχ = 2 K (χ − χ₀); d²E/dχ² = 2 K
            const double dE_dchi = 2.0 * K * dchi;
            const double d2E_dchi2 = 2.0 * K;
            const double dV_dcos = dihedral_geom::dV_dcos_from_dE_dphi(
                dE_dchi, g.sin_phi, g.cos_phi, d2E_dchi2);

            double f1x, f1y, f1z, f2x, f2y, f2z, f3x, f3y, f3z, f4x, f4y, f4z;
            dihedral_geom::forces_from_dV_dcos(g, dV_dcos, f1x, f1y, f1z, f2x,
                                               f2y, f2z, f3x, f3y, f3z, f4x,
                                               f4y, f4z);

            tf.fx[i1] += f1x;
            tf.fy[i1] += f1y;
            tf.fz[i1] += f1z;
            tf.fx[i2] += f2x;
            tf.fy[i2] += f2y;
            tf.fz[i2] += f2z;
            tf.fx[i3] += f3x;
            tf.fy[i3] += f3y;
            tf.fz[i3] += f3z;
            tf.fx[i4] += f4x;
            tf.fy[i4] += f4y;
            tf.fz[i4] += f4z;

            vir_local.add_pair(-g.b1x, -g.b1y, -g.b1z, f1x, f1y, f1z);
            vir_local.add_pair(g.b2x, g.b2y, g.b2z, f3x + f4x, f3y + f4y,
                               f3z + f4z);
            vir_local.add_pair(g.b3x, g.b3y, g.b3z, f4x, f4y, f4z);
        }
#pragma omp critical
        vir += vir_local;
    }

    scratch_.reduce_into(n_force);

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyImproper) += energy_total;
    vir.flush_to(f.meta.get(kStress));

    drops_.evaluated += static_cast<std::uint64_t>(n_tuples);
}

void ImproperHarmonic::destroy(const DestroyContext& /*ctx*/) {
    drops_.report("ImproperHarmonic");
    drops_.reset();
    cache_.clear();
    scratch_.clear();
}

} // namespace cpu
} // namespace ATV
