/// @file cpu/potentials/periodic_torsion.cpp
/// @brief Shared periodic torsion loop (charmm / fourier / cvff).

#include "atomiverse/cpu/potentials/periodic_torsion.h"
#include "atomiverse/core/dihedral_geom.h"
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
namespace md {

PeriodicTorsionSet::PeriodicTorsionSet(
    std::vector<std::vector<TorsionTerm>> terms, Topology topology,
    const char* style)
    : table_(terms, style), topology_(topology), style_(style) {}

void PeriodicTorsionSet::build(const BuildContext& ctx) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (ChunkForces* arena = ctx.runtime.md.chunk_forces)
        arena->reserve(n_threads, ctx.frame.capacity());
}

void PeriodicTorsionSet::rebuild_tuples_(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const bool improper = topology_ == Topology::Improper;
    const auto count =
        f.span(improper ? kAtom_num_improper : kAtom_num_dihedral);
    const auto atom =
        f.span(improper ? kAtom_improper_atom : kAtom_dihedral_atom);
    const auto type =
        f.span(improper ? kAtom_improper_type : kAtom_dihedral_type);
    const ResolveView rv(ctx);
    const int nlocal = rv.n_local();
    const int ntypes = table_.n_types();

    cache_.flat.clear();
    cache_.dropped_at_build = 0;
    // Layout {i1, i2, i3, i4, type}; owner = i2 for both topologies.
    for (int i2 = 0; i2 < nlocal; ++i2) {
        for (int m = 0; m < count[i2]; ++m) {
            const int t = type(i2, m);
            if (t < 0 || t >= ntypes)
                throw std::runtime_error(
                    std::string(style_) + ": topology type " + std::to_string(t)
                    + " has no coefficients (" + std::to_string(ntypes)
                    + " types given)");
            const int i1 = rv.resolve(atom(i2, 3 * m + 0));
            const int i3 = rv.resolve(atom(i2, 3 * m + 1));
            const int i4 = rv.resolve(atom(i2, 3 * m + 2));
            if (i1 < 0 || i3 < 0 || i4 < 0) {
                ++cache_.dropped_at_build;
                continue;
            }
            if (!table_.active[static_cast<std::size_t>(t)])
                continue; // all-zero type: exactly no energy or force
            cache_.flat.insert(cache_.flat.end(), {i1, i2, i3, i4, t});
        }
    }
    cache_.mark_built(ctx);
    drops_.dropped += cache_.dropped_at_build;
}

void PeriodicTorsionSet::compute(const LaunchContext& ctx) {
    if (cache_.stale(ctx))
        rebuild_tuples_(ctx);
    cache_.require_complete(ctx, style_);
    const int n_tuples = static_cast<int>(cache_.flat.size()) / 5;
    if (n_tuples == 0)
        return;

    auto& f = ctx.frame;
    const int* __restrict tup = cache_.flat.data();
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);

    const Box& box = f.box;
    auto min_image = [&box](double& ax, double& ay, double& az) {
        box.min_image(ax, ay, az);
    };
    const bool do_virial = compute_stress_ && ctx.stress_requested;
    const bool do_forces = ctx.forces_requested;

    const int* __restrict off = table_.offset.data();
    const double* __restrict Kt = table_.K.data();
    const int* __restrict nt = table_.n.data();
    const double* __restrict cd = table_.cos_delta.data();
    const double* __restrict sd = table_.sin_delta.data();

    // Each chunk keeps its own energy and virial and stores them once;
    // ForcePass sums the chunks in order (deterministic per thread count).
    ForcePass forces(ctx);
    forces.for_each_chunk(n_tuples, [&](int c, int q_begin, int q_end,
                                        ForceRows tf) {
        double energy_total = 0.0;
        Voigt6 vir_local;

        for (int q = q_begin; q < q_end; ++q) {
            const int i1 = tup[5 * q + 0];
            const int i2 = tup[5 * q + 1];
            const int i3 = tup[5 * q + 2];
            const int i4 = tup[5 * q + 3];
            const int t = tup[5 * q + 4];

            dihedral_geom::Geometry g;
            if (!dihedral_geom::fill(x[i1], y[i1], z[i1], x[i2], y[i2], z[i2],
                                     x[i3], y[i3], z[i3], x[i4], y[i4], z[i4],
                                     min_image, g))
                continue;

            // V = Σ K (1 + cos(nφ − δ)); dV/dφ = −Σ K n sin(nφ − δ).
            double energy, dE_dphi, d2E_dphi2;
            torsion_sum(off[t], off[t + 1], Kt, nt, cd, sd, g.cos_phi,
                        g.sin_phi, energy, dE_dphi, d2E_dphi2);
            energy_total += energy;

            const double dV_dcos = dihedral_geom::dV_dcos_from_dE_dphi(
                dE_dphi, g.sin_phi, g.cos_phi, d2E_dphi2);
            double f1x, f1y, f1z, f2x, f2y, f2z, f3x, f3y, f3z, f4x, f4y, f4z;
            dihedral_geom::forces_from_dV_dcos(g, dV_dcos, f1x, f1y, f1z, f2x,
                                               f2y, f2z, f3x, f3y, f3z, f4x,
                                               f4y, f4z);
            if (do_forces) {
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
            }
            if (do_virial) {
                vir_local.add_pair(-g.b1x, -g.b1y, -g.b1z, f1x, f1y, f1z);
                vir_local.add_pair(g.b2x, g.b2y, g.b2z, f3x + f4x, f3y + f4y,
                                   f3z + f4z);
                vir_local.add_pair(g.b3x, g.b3y, g.b3z, f4x, f4y, f4z);
            }
        }
        forces.store(c, energy_total, vir_local);
    });
    const double energy_total = forces.energy();
    const Voigt6 vir = forces.virial();

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(topology_ == Topology::Improper ? kEnergyImproper
                                               : kEnergyDihedral) +=
        energy_total;
    if (do_virial)
        vir.flush_to(f.meta.get(kStress));
    drops_.evaluated += static_cast<std::uint64_t>(n_tuples);
}

void PeriodicTorsionSet::destroy() {
    drops_.report(style_);
    drops_.reset();
    cache_.clear();
}

} // namespace md
} // namespace cpu
} // namespace ATV
