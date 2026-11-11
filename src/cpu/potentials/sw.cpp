/// @file src/cpu/potentials/sw.cpp
/// @brief cpu::StillingerWeber — three-body force style.
///
/// Thread-owned force buffers and short-neighbor traversal ported from LAMMPS
/// `src/OPENMP/pair_sw_omp.cpp` and fused radial/angular coefficients from
/// `src/MANYBODY/pair_sw.cpp`, checkout
/// `a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b`.
/// Copyright (2003) Sandia Corporation. Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software. Distributed under the GNU General
/// Public License (GPL-2.0-only).

#include "atomiverse/cpu/potentials/sw.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/sw_data/sw_tables.h"

namespace ATV {
namespace cpu {

SWPotential::SWPotential(molrs::Element element, std::string_view name) {
    const int z = static_cast<int>(element);
    for (unsigned long k = 0; k < sw_data::kStillingerWeberCount; ++k) {
        const sw_data::SWRow& row = sw_data::kStillingerWeber[k];
        if (row.z == z && name == row.name) {
            eps_ = row.eps;
            sigma_ = row.sigma;
            A_ = row.A;
            B_ = row.B;
            p_ = row.p;
            q_ = row.q;
            a_ = row.a;
            lambda_ = row.lambda;
            gamma_ = row.gamma;
            cos_theta0_ = row.cos_theta0;
            return;
        }
    }
    throw std::out_of_range("SWPotential: no parameterization '"
                            + std::string(name)
                            + "' for Z=" + std::to_string(z));
}

StillingerWeber::StillingerWeber(SWPotential potential) : pot_(potential) {}

void StillingerWeber::build(const cpu::BuildContext& ctx) {
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    if (md::ChunkForces* arena = ctx.runtime.md.chunk_forces)
        arena->reserve(threads, ctx.frame.capacity());
    short_neighbors_.resize(static_cast<std::size_t>(threads));
}

void StillingerWeber::launch(const cpu::LaunchContext& ctx) {
    cpu::Frame& f = ctx.frame;
    const cpu::md::PairList* pl = ctx.runtime.md.full_pair_list;
    if (pl == nullptr)
        throw std::runtime_error("StillingerWeber: no neighbor list published");

    const double rc = pot_.cutoff();
    const double rc2 = rc * rc;
    const double cos0 = pot_.cos_theta0();
    const double le = pot_.lambda_eps();
    const double sigma_gamma = pot_.sigma_ * pot_.gamma_;
    const bool derivatives = ctx.forces_requested || ctx.stress_requested;

    // Resolve every schema column ONCE at the launch boundary; the triple
    // loop below (owners x neighbors x neighbors) touches nothing but these
    // raw pointers.
    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    const TagInt* __restrict tag_ptr = f.span(kAtom_tag);
    md::ForcePass forces(ctx);
    int max_neighbors = 0;
    for (int ii = 0; ii < pl->inum; ++ii)
        max_neighbors = std::max(max_neighbors, pl->numneigh[pl->ilist[ii]]);
    for (auto& neighbors : short_neighbors_)
        neighbors.resize(static_cast<std::size_t>(max_neighbors));

    // One chunk per thread (static loop): thread t owns force slot t and sum
    // slot t, summed in slot order — deterministic per thread count.
    const int team = forces.team();
    forces.open(team);
#pragma omp parallel num_threads(team) if (team > 1)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const md::ForceRows force = forces.rows(tid);
        double e_total = 0.0;
        md::Voigt6 vir;
        double* v = vir.v;
        double* __restrict fx_ptr = force.fx;
        double* __restrict fy_ptr = force.fy;
        double* __restrict fz_ptr = force.fz;
        int* __restrict short_list = short_neighbors_[tid].data();

        // Only owners supply vertices. Keep every short neighbor for the
        // angular term, even when the two-body ownership rule rejects a pair.
#pragma omp for schedule(static)
        for (int ii = 0; ii < pl->inum; ++ii) {
            const int i = pl->ilist[ii];
            const int off = pl->firstneigh[i];
            const int nn = pl->numneigh[i];
            const TagInt itag = tag_ptr[i];
            const double xi = x_ptr[i], yi = y_ptr[i], zi = z_ptr[i];
            double fix = 0.0, fiy = 0.0, fiz = 0.0;
            int n_short = 0;

            for (int jj = 0; jj < nn; ++jj) {
                const int j = pl->neighbors[off + jj];
                double dx = xi - x_ptr[j];
                double dy = yi - y_ptr[j];
                double dz = zi - z_ptr[j];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc2)
                    continue;
                short_list[n_short++] = off + jj;

                // LAMMPS full-list Newton ownership. XOR gives the parity of
                // itag+jtag without overflowing a signed TagInt. Equal tags
                // use the unwrapped image coordinates, as in pair_sw_omp.cpp.
                const TagInt jtag = tag_ptr[j];
                const bool odd = ((itag ^ jtag) & TagInt{1}) != 0;
                if (itag > jtag) {
                    if (!odd)
                        continue;
                } else if (itag < jtag) {
                    if (odd)
                        continue;
                } else if (dz > 0 || (dz == 0 && dy > 0)
                           || (dz == 0 && dy == 0 && dx > 0)) {
                    continue;
                }

                const double r = std::sqrt(r2);
                double derivative;
                e_total +=
                    pot_.two_body(r, derivatives ? &derivative : nullptr);
                if (!derivatives)
                    continue;
                const double fpair = -derivative / r;
                if (ctx.forces_requested) {
                    fix += fpair * dx;
                    fiy += fpair * dy;
                    fiz += fpair * dz;
                    fx_ptr[j] -= fpair * dx;
                    fy_ptr[j] -= fpair * dy;
                    fz_ptr[j] -= fpair * dz;
                }
                if (ctx.stress_requested) {
                    v[0] += dx * fpair * dx;
                    v[1] += dy * fpair * dy;
                    v[2] += dz * fpair * dz;
                    v[3] += dx * fpair * dy;
                    v[4] += dx * fpair * dz;
                    v[5] += dy * fpair * dz;
                }
            }

            // Each unordered {j,k} is counted once at its owned vertex i.
            // LAMMPS's fused scalar coefficients share the radial exponential
            // between energy and force, and hoist the first leg out of kk.
            for (int jj = 0; jj + 1 < n_short; ++jj) {
                const int jentry = short_list[jj];
                const int j = pl->neighbors[jentry];
                double ax = xi - x_ptr[j];
                double ay = yi - y_ptr[j];
                double az = zi - z_ptr[j];
                ax = -ax;
                ay = -ay;
                az = -az;
                const double r1sq = ax * ax + ay * ay + az * az;
                const double r1 = std::sqrt(r1sq);
                const double rainv1 = 1.0 / (r1 - rc);
                const double g1 = std::exp(sigma_gamma * rainv1);
                const double radial1 = sigma_gamma * rainv1 * rainv1 / r1;
                const double rinvsq1 = 1.0 / r1sq;
                double fjx_sum = 0.0, fjy_sum = 0.0, fjz_sum = 0.0;

                for (int kk = jj + 1; kk < n_short; ++kk) {
                    const int kentry = short_list[kk];
                    const int k = pl->neighbors[kentry];
                    double bx = xi - x_ptr[k];
                    double by = yi - y_ptr[k];
                    double bz = zi - z_ptr[k];
                    bx = -bx;
                    by = -by;
                    bz = -bz;
                    const double r2sq = bx * bx + by * by + bz * bz;
                    const double r2 = std::sqrt(r2sq);
                    const double rainv2 = 1.0 / (r2 - rc);
                    const double g2 = std::exp(sigma_gamma * rainv2);
                    const double rinv12 = 1.0 / (r1 * r2);
                    const double cs = (ax * bx + ay * by + az * bz) * rinv12;
                    const double delta = cs - cos0;
                    const double prefactor = le * g1 * g2;
                    const double energy = prefactor * delta * delta;
                    e_total += energy;
                    if (!derivatives)
                        continue;

                    const double radial2 = sigma_gamma * rainv2 * rainv2 / r2;
                    const double angular = 2.0 * prefactor * delta;
                    const double angular12 = rinv12 * angular;
                    const double csangular = cs * angular;
                    const double coeff1 =
                        energy * radial1 + csangular * rinvsq1;
                    const double coeff2 = energy * radial2 + csangular / r2sq;
                    const double fjx = ax * coeff1 - bx * angular12;
                    const double fjy = ay * coeff1 - by * angular12;
                    const double fjz = az * coeff1 - bz * angular12;
                    const double fkx = bx * coeff2 - ax * angular12;
                    const double fky = by * coeff2 - ay * angular12;
                    const double fkz = bz * coeff2 - az * angular12;

                    if (ctx.forces_requested) {
                        fjx_sum += fjx;
                        fjy_sum += fjy;
                        fjz_sum += fjz;
                        fx_ptr[k] += fkx;
                        fy_ptr[k] += fky;
                        fz_ptr[k] += fkz;
                        fix -= fjx + fkx;
                        fiy -= fjy + fky;
                        fiz -= fjz + fkz;
                    }
                    // Leg vectors point from i to j/k, matching LAMMPS's
                    // three-body virial. Ghost forces use the existing reverse.
                    if (ctx.stress_requested) {
                        v[0] += ax * fjx + bx * fkx;
                        v[1] += ay * fjy + by * fky;
                        v[2] += az * fjz + bz * fkz;
                        v[3] += ax * fjy + bx * fky;
                        v[4] += ax * fjz + bx * fkz;
                        v[5] += ay * fjz + by * fkz;
                    }
                }
                if (ctx.forces_requested) {
                    fx_ptr[j] += fjx_sum;
                    fy_ptr[j] += fjy_sum;
                    fz_ptr[j] += fjz_sum;
                }
            }
            if (ctx.forces_requested) {
                fx_ptr[i] += fix;
                fy_ptr[i] += fiy;
                fz_ptr[i] += fiz;
            }
        }
        forces.store(tid, e_total, vir);
    } // chunk-owned Newton-on force accumulation
    const double e_total = forces.energy();
    f.meta.get(kEnergy) += e_total;
    f.meta.get(kEnergyPair) += e_total;
    if (ctx.stress_requested)
        forces.virial().flush_to(f.meta.get(kStress));
}

void StillingerWeber::destroy(const cpu::DestroyContext&) {
    short_neighbors_.clear();
}

} // namespace cpu
} // namespace ATV
