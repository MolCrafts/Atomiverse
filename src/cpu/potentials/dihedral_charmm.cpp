/// @file cpu/potentials/dihedral_charmm.cpp
/// @brief CPU DihedralCharmm — V = K(1 + cos(nφ − δ)).

#include "atomiverse/cpu/potentials/dihedral_charmm.h"
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

DihedralCharmm::DihedralCharmm(std::vector<double> K, std::vector<int> n,
                               std::vector<double> delta)
    : K_(std::move(K)), n_(std::move(n)), delta_(std::move(delta)) {
    if (K_.empty() || K_.size() != n_.size() || K_.size() != delta_.size()) {
        throw std::invalid_argument(
            "DihedralCharmm: K, n, delta must be non-empty and same size");
    }
    cos_delta_.resize(delta_.size());
    sin_delta_.resize(delta_.size());
    for (std::size_t t = 0; t < K_.size(); ++t) {
        if (!std::isfinite(K_[t]) || !std::isfinite(delta_[t])) {
            throw std::invalid_argument(
                "DihedralCharmm: non-finite K or delta at type "
                + std::to_string(t));
        }
        if (n_[t] < 1) {
            throw std::invalid_argument(
                "DihedralCharmm: multiplicity n must be >= 1 (type "
                + std::to_string(t) + ")");
        }
        cos_delta_[t] = std::cos(delta_[t]);
        sin_delta_[t] = std::sin(delta_[t]);
    }
}

void DihedralCharmm::build(const BuildContext& ctx) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    scratch_.build(n_threads, ctx.frame.capacity());
}

void DihedralCharmm::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const auto num_dihedral = f.span(kAtom_num_dihedral);
    const auto dihedral_atom = f.span(kAtom_dihedral_atom);
    const auto dihedral_type = f.span(kAtom_dihedral_type);

    const md::ResolveView rv(ctx);
    const int nlocal = rv.n_local();
    const int n_force = rv.n_force();

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
    md::Voigt6 vir;

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);

    // Hoist orthorhombic min-image constants (avoids per-bond virtual/ indirections).
    const double lx = f.box.length(0), ly = f.box.length(1),
                 lz = f.box.length(2);
    const double inv_lx = 1.0 / lx, inv_ly = 1.0 / ly, inv_lz = 1.0 / lz;
    auto min_image = [&](double& ax, double& ay, double& az) {
        ax -= lx * std::nearbyint(ax * inv_lx);
        ay -= ly * std::nearbyint(ay * inv_ly);
        az -= lz * std::nearbyint(az * inv_lz);
    };
    const bool do_virial = compute_stress_;

    const double* __restrict Ktab = K_.data();
    const int* __restrict ntab = n_.data();
    const double* __restrict cosd = cos_delta_.data();
    const double* __restrict sind = sin_delta_.data();
    const std::size_t ntypes = K_.size();

    scratch_.bind(f);

#pragma omp parallel if (scratch_.n_threads() > 1) reduction(+ : energy_total)
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
            if (type < 0 || static_cast<std::size_t>(type) >= ntypes)
                continue;
            // K = 0 contributes exactly zero energy and force — the type is
            // owned by the other member of a hybrid dihedral split.
            if (Ktab[static_cast<std::size_t>(type)] == 0.0)
                continue;

            dihedral_geom::Geometry g;
            const bool ok = dihedral_geom::fill(
                x_ptr[i1], y_ptr[i1], z_ptr[i1], x_ptr[i2], y_ptr[i2],
                z_ptr[i2], x_ptr[i3], y_ptr[i3], z_ptr[i3], x_ptr[i4],
                y_ptr[i4], z_ptr[i4], min_image, g);
            if (!ok)
                continue;

            const double K = Ktab[static_cast<std::size_t>(type)];
            const int n = ntab[static_cast<std::size_t>(type)];
            // cos(nφ), sin(nφ) by recurrence — no atan2 / sincos.
            // GAFF n is almost always 1..3; special-case those.
            const double c = g.cos_phi;
            const double s = g.sin_phi;
            double cn, sn;
            if (n == 1) {
                cn = c;
                sn = s;
            } else if (n == 2) {
                cn = c * c - s * s;
                sn = 2.0 * s * c;
            } else if (n == 3) {
                // cos3φ = 4c³−3c, sin3φ = 3s−4s³
                cn = (4.0 * c * c - 3.0) * c;
                sn = (3.0 - 4.0 * s * s) * s;
            } else {
                cn = 1.0;
                sn = 0.0;
                for (int k = 0; k < n; ++k) {
                    const double cn1 = cn * c - sn * s;
                    const double sn1 = sn * c + cn * s;
                    cn = cn1;
                    sn = sn1;
                }
            }
            // cos(nφ−δ) = cn cosδ + sn sinδ
            // sin(nφ−δ) = sn cosδ − cn sinδ
            const double cd = cosd[static_cast<std::size_t>(type)];
            const double sd = sind[static_cast<std::size_t>(type)];
            const double carg = cn * cd + sn * sd;
            const double sarg = sn * cd - cn * sd;

            // V = K(1 + cos(nφ − δ))
            energy_total += K * (1.0 + carg);

            // dE/dφ = −K n sin(nφ − δ)
            const double dE_dphi = -K * static_cast<double>(n) * sarg;
            const double d2E_dphi2 = -K * static_cast<double>(n * n) * carg;
            const double dV_dcos = dihedral_geom::dV_dcos_from_dE_dphi(
                dE_dphi, g.sin_phi, g.cos_phi, d2E_dphi2);

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

            if (do_virial) {
                vir_local.add_pair(-g.b1x, -g.b1y, -g.b1z, f1x, f1y, f1z);
                vir_local.add_pair(g.b2x, g.b2y, g.b2z, f3x + f4x, f3y + f4y,
                                   f3z + f4z);
                vir_local.add_pair(g.b3x, g.b3y, g.b3z, f4x, f4y, f4z);
            }
        }
        if (do_virial) {
#pragma omp critical
            vir += vir_local;
        }
    }

    scratch_.reduce_into(n_force);

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyDihedral) += energy_total;
    if (do_virial)
        vir.flush_to(f.meta.get(kStress));

    drops_.evaluated += static_cast<std::uint64_t>(n_tuples);
}

void DihedralCharmm::destroy(const DestroyContext& /*ctx*/) {
    drops_.report("DihedralCharmm");
    drops_.reset();
    cache_.clear();
    scratch_.clear();
}

} // namespace cpu
} // namespace ATV
