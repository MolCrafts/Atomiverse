/// @file cpu/potentials/dihedral_multi_harmonic.cpp
/// @brief CPU DihedralMultiHarmonic — V = Σₙ Cₙ cosⁿ(φ) implementation.

#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_geom.h"
#include "atomiverse/cpu/potentials/resolve_view.h"

#include <cstddef>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

DihedralMultiHarmonic::DihedralMultiHarmonic(
    std::vector<std::vector<double>> Cn)
    : Cn_(std::move(Cn)) {}

void DihedralMultiHarmonic::build(const BuildContext& ctx) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    scratch_.build(n_threads, ctx.frame.capacity());
}

void DihedralMultiHarmonic::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const auto num_dihedral = f.span(kAtom_num_dihedral);
    const auto dihedral_atom = f.span(kAtom_dihedral_atom);
    const auto dihedral_type = f.span(kAtom_dihedral_type);

    const md::ResolveView rv(ctx);
    const int nlocal = rv.n_local();
    const int n_force = rv.n_force();

    // Tuple cache: tag→local hash resolution happens only when row
    // indices may have moved (migration / halo full rebuild), not every
    // step. Layout: {i1, i2, i3, i4, type} per dihedral (i2 = owner).
    if (cache_.stale(ctx)) {
        cache_.flat.clear();
        cache_.dropped_at_build = 0;
        for (int i2 = 0; i2 < nlocal; ++i2) {
            const int nd = num_dihedral[i2];
            for (int m = 0; m < nd; ++m) {
                const int i1 = rv.resolve(dihedral_atom(i2, 3 * m + 0));
                const int i3 = rv.resolve(dihedral_atom(i2, 3 * m + 1));
                const int i4 = rv.resolve(dihedral_atom(i2, 3 * m + 2));
                if (i1 < 0 || i3 < 0 || i4 < 0) {
                    ++cache_.dropped_at_build;
                    continue;
                }
                cache_.flat.push_back(i1);
                cache_.flat.push_back(i2);
                cache_.flat.push_back(i3);
                cache_.flat.push_back(i4);
                cache_.flat.push_back(dihedral_type(i2, m));
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
    // Dihedral virial = (-b1)⊗F1 + b2⊗(F3+F4) + b3⊗F4 (with i2 as the
    // reference atom and Newton's 3rd F2 = -(F1+F3+F4)) — translation/PBC
    // invariant. Each term is one Voigt6::add_pair.
    md::Voigt6 vir;

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);

    // Resolve the frame's force spans ONCE, before the parallel region opens
    // (never inside it — see BondedForceScratch::bind).
    scratch_.bind(f);

// Thread-private virial combined under a critical section (no per-step heap
// alloc, unlike an OMP user-defined reduction on Voigt6).
#pragma omp parallel reduction(+ : energy_total)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const md::ThreadForce tf = scratch_.begin_thread(tid, n_force);
        md::Voigt6 vir_local;

        // Hot loop over pre-resolved tuples: atom `i2` (2nd of the
        // i-j-k-l quadruple) owns the dihedral; partners may be ghosts.
#pragma omp for schedule(static)
        for (int t = 0; t < n_tuples; ++t) {
            const int i1 = tup[5 * t + 0];
            const int i2 = tup[5 * t + 1];
            const int i3 = tup[5 * t + 2];
            const int i4 = tup[5 * t + 3];
            const int type = tup[5 * t + 4];

            dihedral_geom::Geometry g;
            const bool ok = dihedral_geom::fill(
                x_ptr[i1], y_ptr[i1], z_ptr[i1], x_ptr[i2], y_ptr[i2],
                z_ptr[i2], x_ptr[i3], y_ptr[i3], z_ptr[i3], x_ptr[i4],
                y_ptr[i4], z_ptr[i4],
                [&](double& ax, double& ay, double& az) {
                    f.simbox.min_image(ax, ay, az);
                },
                g);
            if (!ok)
                continue;

            // V = Σ Cn cos^n; dV/dcos = Σ n·Cn·cos^(n-1). Running
            // cos_pow_minus1 avoids dividing by cos(phi) (vanishes at ±90°).
            const auto& coeffs = Cn_[type];
            double energy = 0;
            double dV_dcos = 0;
            double cos_pow = 1;
            double cos_pow_minus1 = 0;
            for (std::size_t n = 0; n < coeffs.size(); ++n) {
                energy += coeffs[n] * cos_pow;
                if (n > 0)
                    dV_dcos +=
                        static_cast<double>(n) * coeffs[n] * cos_pow_minus1;
                cos_pow_minus1 = cos_pow;
                cos_pow *= g.cos_phi;
            }
            energy_total += energy;

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

            // Virial: (-b1)⊗F1 + b2⊗(F3+F4) + b3⊗F4.
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
    f.meta.get(kEnergyDihedral) += energy_total;
    vir.flush_to(f.meta.get(kStress));

    drops_.evaluated += static_cast<std::uint64_t>(n_tuples);
}

void DihedralMultiHarmonic::destroy(const DestroyContext& /*ctx*/) {
    drops_.report("DihedralMultiHarmonic");
    drops_.reset();
    cache_.clear();
    scratch_.clear();
}

} // namespace cpu
} // namespace ATV
