/// @file src/cpu/potentials/pair_coul_soft.cpp
/// @brief Beutler/GROMACS one-leg soft-core Coulomb (neighbor-list walk).

#include "atomiverse/cpu/potentials/pair_coul_soft.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

PairCoulSoft::PairCoulSoft(double cutoff, double sigma_c, double alpha,
                           double p)
    : Pair(cutoff), sigma_c_(sigma_c), alpha_(alpha), p_(p) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairCoulSoft: cutoff must be > 0");
    if (!(sigma_c > 0.0))
        throw std::invalid_argument("PairCoulSoft: sigma_c must be > 0");
    if (!(alpha > 0.0))
        throw std::invalid_argument("PairCoulSoft: alpha must be > 0");
    if (!(p > 0.0))
        throw std::invalid_argument("PairCoulSoft: p must be > 0");

    sigma6_ = sigma_c * sigma_c * sigma_c * sigma_c * sigma_c * sigma_c;
    cutoff_sq_ = cutoff * cutoff;
    inv_rc_ = 1.0 / cutoff;
}

void PairCoulSoft::build(const BuildContext& ctx) {
    n_threads_ = 1;
#ifdef _OPENMP
    n_threads_ = omp_get_max_threads();
#endif
    const int n = ctx.frame.capacity();
    if (md::ThreadForces* arena = ctx.runtime.md.thread_forces)
        arena->reserve(n_threads_, n);
    dU_dlambda_ = 0;
}

void PairCoulSoft::destroy(const DestroyContext&) {}

void PairCoulSoft::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairCoulSoft::compute: runtime.md.pair_list is null");
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
    const StorageReal* __restrict q_ptr = f.span(kAtom_charge);

    const double rc_sq = cutoff_sq_;
    const double inv_rc = inv_rc_;
    const double sig6 = sigma6_;
    const double alpha = alpha_;
    const double p = p_;
    const double qqr2e = units::md_qqr2e;
    const double lam = std::min(1.0, std::max(0.0, lambda));
    const double one_m_lam = 1.0 - lam;
    const double soft0 = alpha * sig6 * std::pow(one_m_lam, p);
    // d soft0 / d λ = α σ⁶ p (1−λ)^{p−1} * (−1)
    const double dsoft0_dlam =
        (p == 1.0) ? (-alpha * sig6)
                   : (-alpha * sig6 * p * std::pow(one_m_lam, p - 1.0));

    double energy_total = 0;
    double dU_total = 0;
    double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;

    if (!(lam > 0.0) && soft0 == 0.0) {
        dU_dlambda_ = 0;
        return;
    }

    const md::ForcePass forces(ctx);
#pragma omp parallel num_threads(forces.team()) if (forces.team() > 1)         \
    reduction(+ : energy_total) reduction(+ : dU_total) reduction(+ : v_xx)    \
    reduction(+ : v_yy) reduction(+ : v_zz) reduction(+ : v_xy)                \
    reduction(+ : v_xz) reduction(+ : v_yz)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const md::ThreadForce tf = forces.thread(tid);
        double* __restrict fxt_ptr = tf.fx;
        double* __restrict fyt_ptr = tf.fy;
        double* __restrict fzt_ptr = tf.fz;

#pragma omp for schedule(static)
        for (int ii = 0; ii < k_inum; ++ii) {
            const int i = ilist[ii];
            const double xi = x_ptr[i], yi = y_ptr[i], zi = z_ptr[i];
            const double qi = q_ptr[i];
            if (qi == 0.0)
                continue;
            const int jnum = numneigh[i];
            const int off = firstneigh[i];
            double fi_x = 0, fi_y = 0, fi_z = 0;
            for (int jj = 0; jj < jnum; ++jj) {
                const int j = neighbors[off + jj];
                const double qj = q_ptr[j];
                if (qj == 0.0)
                    continue;
                double dx, dy, dz;
                dx = xi - x_ptr[j];
                dy = yi - y_ptr[j];
                dz = zi - z_ptr[j];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc_sq || r2 == 0.0)
                    continue;

                const double r6 = r2 * r2 * r2;
                const double r_lam = std::pow(soft0 + r6, 1.0 / 6.0);
                const double inv_rl = 1.0 / r_lam;
                const double qq = qqr2e * qi * qj;

                // V = λ qq (1/r_λ − 1/r_c)
                const double vpair = qq * (inv_rl - inv_rc);
                energy_total += lam * vpair;

                // ∂V/∂λ = vpair + λ qq (−1/r_λ²) ∂r_λ/∂λ
                // ∂r_λ/∂λ = (1/6)(soft0+r6)^{−5/6} ∂soft0/∂λ
                //         = dsoft0_dlam / (6 r_λ⁵)
                const double r_lam5 = r_lam * r_lam * r_lam * r_lam * r_lam;
                const double drlam_dlam = dsoft0_dlam / (6.0 * r_lam5);
                dU_total += vpair + lam * qq * (-inv_rl * inv_rl) * drlam_dlam;

                // fpair such that F_i = fpair * dx:
                // F_r = λ qq r⁵ / r_λ⁷  ⇒ fpair = λ qq r⁴ / r_λ⁷
                if (lam > 0.0) {
                    const double r4 = r2 * r2;
                    const double r_lam7 = r_lam5 * r_lam * r_lam;
                    const double fpair = lam * qq * r4 / r_lam7;
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
    }

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyPair) += energy_total;
    if (ctx.stress_requested) {
        f.meta.get(kStress)[0] += v_xx;
        f.meta.get(kStress)[1] += v_yy;
        f.meta.get(kStress)[2] += v_zz;
        f.meta.get(kStress)[3] += v_xy;
        f.meta.get(kStress)[4] += v_xz;
        f.meta.get(kStress)[5] += v_yz;
    }
    dU_dlambda_ = dU_total;
}

} // namespace cpu
} // namespace ATV
