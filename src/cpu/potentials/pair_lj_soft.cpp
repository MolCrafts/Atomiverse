/// @file src/cpu/potentials/pair_lj_soft.cpp
/// @brief Beutler/GROMACS one-leg soft-core LJ (neighbor-list walk).

#include "atomiverse/cpu/potentials/pair_lj_soft.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

PairLJSoft::PairLJSoft(double epsilon, double sigma, double cutoff,
                       double alpha, double p)
    : Pair(cutoff), epsilon_(epsilon), sigma_(sigma), alpha_(alpha), p_(p) {
    if (!(epsilon > 0.0))
        throw std::invalid_argument("PairLJSoft: epsilon must be > 0");
    if (!(sigma > 0.0))
        throw std::invalid_argument("PairLJSoft: sigma must be > 0");
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairLJSoft: cutoff must be > 0");
    if (!(alpha > 0.0))
        throw std::invalid_argument("PairLJSoft: alpha must be > 0");
    if (!(p > 0.0))
        throw std::invalid_argument("PairLJSoft: p must be > 0");

    sigma6_ = sigma * sigma * sigma * sigma * sigma * sigma;
    cutoff_sq_ = cutoff * cutoff;
    const double r2inv = 1.0 / cutoff_sq_;
    const double r6inv = r2inv * r2inv * r2inv * sigma6_;
    v_shift_ = 4.0 * epsilon_ * (r6inv * r6inv - r6inv);
}

void PairLJSoft::build(const BuildContext& ctx) {
    n_threads_ = 1;
#ifdef _OPENMP
    n_threads_ = omp_get_max_threads();
#endif
    const int n = ctx.frame.capacity();
    if (md::ChunkForces* arena = ctx.runtime.md.chunk_forces)
        arena->reserve(n_threads_, n);
    dU_dlambda_ = 0;
}

void PairLJSoft::destroy(const DestroyContext&) {}

void PairLJSoft::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairLJSoft::compute: runtime.md.pair_list is null. Did you add a "
            "NeighborList ahead of the soft LJ potential?");
    }
    const auto& pairs = *ctx.runtime.md.pair_list;

    const int k_inum = pairs.inum;
    const int* __restrict ilist = pairs.ilist.data();
    const int* __restrict numneigh = pairs.numneigh.data();
    const int* __restrict firstneigh = pairs.firstneigh.data();
    const int* __restrict neighbors = pairs.neighbors.data();

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);

    const double rc_sq = cutoff_sq_;
    const double sig6 = sigma6_;
    const double eps4 = 4.0 * epsilon_;
    const double eps24 = 24.0 * epsilon_;
    const double vshift = v_shift_;
    const double alpha = alpha_;
    const double p = p_;
    // Clamp λ to a sane range for stability; user still owns the member.
    const double lam = std::min(1.0, std::max(0.0, lambda));
    const double one_m_lam = 1.0 - lam;
    const double soft0 = alpha * sig6 * std::pow(one_m_lam, p);
    const double dsoft0_dlam =
        (p == 1.0) ? (-alpha * sig6)
                   : (-alpha * sig6 * p * std::pow(one_m_lam, p - 1.0));

    if (!(lam > 0.0) && soft0 == 0.0) {
        dU_dlambda_ = 0;
        return;
    }

    // One chunk per thread (static rows): thread t owns force slot t and
    // sum slot t, summed in slot order — deterministic per thread count.
    md::ForcePass forces(ctx);
    const int team = forces.team();
    forces.open(team);
#pragma omp parallel num_threads(team) if (team > 1)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const md::ForceRows tf = forces.rows(tid);
        double energy_total = 0;
        double dU_total = 0;
        double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;
        double* __restrict fxt_ptr = tf.fx;
        double* __restrict fyt_ptr = tf.fy;
        double* __restrict fzt_ptr = tf.fz;

#pragma omp for schedule(static)
        for (int ii = 0; ii < k_inum; ++ii) {
            const int i = ilist[ii];
            const double xi = x_ptr[i], yi = y_ptr[i], zi = z_ptr[i];
            const int jnum = numneigh[i];
            const int off = firstneigh[i];
            double fi_x = 0, fi_y = 0, fi_z = 0;
            for (int jj = 0; jj < jnum; ++jj) {
                const int j = neighbors[off + jj];
                double dx, dy, dz;
                dx = xi - x_ptr[j];
                dy = yi - y_ptr[j];
                dz = zi - z_ptr[j];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc_sq || r2 == 0.0)
                    continue;

                const double r = std::sqrt(r2);
                const double r6 = r2 * r2 * r2;
                const double r_lam = std::pow(soft0 + r6, 1.0 / 6.0);
                const double r_lam2 = r_lam * r_lam;
                const double r2inv_l = 1.0 / r_lam2;
                const double r6inv_l = r2inv_l * r2inv_l * r2inv_l * sig6;

                // Energy: λ (V_LJ(r_λ) − V_shift)
                const double vlj = eps4 * (r6inv_l * r6inv_l - r6inv_l);
                const double vpair = vlj - vshift;
                energy_total += lam * vpair;

                // ∂U/∂λ = vpair + λ (dV_LJ/dr_λ)(dr_λ/dλ)
                // dV/dr = −fpair_h · r_λ  (fpair definition F = fpair·dx)
                const double fpair_h =
                    eps24 * r2inv_l * (2.0 * r6inv_l * r6inv_l - r6inv_l);
                const double dVdr = -fpair_h * r_lam;
                const double r_lam5 = r_lam2 * r_lam2 * r_lam; // r_λ^4 * r_λ
                const double drlam_dlam = dsoft0_dlam / (6.0 * r_lam5);
                dU_total += vpair + lam * dVdr * drlam_dlam;

                // Hard fpair at r_λ, then soft map: fpair_s = λ fpair_h (r/r_λ)^4
                if (lam > 0.0) {
                    const double ratio = r / r_lam;
                    const double ratio2 = ratio * ratio;
                    const double fpair = lam * fpair_h * ratio2 * ratio2;

                    const double fxp = fpair * dx;
                    const double fyp = fpair * dy;
                    const double fzp = fpair * dz;
                    fi_x += fxp;
                    fi_y += fyp;
                    fi_z += fzp;
                    if (ctx.forces_requested) {
                        fxt_ptr[j] -= fxp;
                        fyt_ptr[j] -= fyp;
                        fzt_ptr[j] -= fzp;
                    }

                    v_xx += dx * fxp;
                    v_yy += dy * fyp;
                    v_zz += dz * fzp;
                    v_xy += dx * fyp;
                    v_xz += dx * fzp;
                    v_yz += dy * fzp;
                }
            }
            if (ctx.forces_requested) {
                fxt_ptr[i] += fi_x;
                fyt_ptr[i] += fi_y;
                fzt_ptr[i] += fi_z;
            }
        }
        forces.store(tid, energy_total,
                     md::Voigt6{{v_xx, v_yy, v_zz, v_xy, v_xz, v_yz}},
                     dU_total);
    }
    const double energy_total = forces.energy();
    const double dU_total = forces.extra();
    const md::Voigt6 vir = forces.virial();

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyPair) += energy_total;
    if (ctx.stress_requested)
        vir.flush_to(f.meta.get(kStress));
    dU_dlambda_ = dU_total;
}

} // namespace cpu
} // namespace ATV
