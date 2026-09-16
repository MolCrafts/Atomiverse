/// @file src/cpu/potentials/pair_lj_soft.cpp
/// @brief Beutler/GROMACS one-leg soft-core LJ (neighbor-list walk).

#include "atomiverse/cpu/potentials/pair_lj_soft.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

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
    fx_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
    fy_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
    fz_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
    dU_dlambda_ = 0;
}

void PairLJSoft::destroy(const DestroyContext&) {
    fx_thr_.clear();
    fy_thr_.clear();
    fz_thr_.clear();
}

void PairLJSoft::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairLJSoft::compute: runtime.md.pair_list is null. Did you add a "
            "NeighborList ahead of the soft LJ potential?");
    }
    const auto& pairs = *ctx.runtime.md.pair_list;
    const auto& box = f.box;

    const auto* dd = ctx.runtime.md.dd;
#ifdef ATV_HAS_MPI
    const bool mpi = dd && ctx.runtime.mpi_size > 1;
    const int n_force = mpi ? dd->n_local + dd->n_ghost : f.nlocal();
#else
    (void)dd;
    const int n_force = f.nlocal();
#endif

    if (static_cast<int>(fx_thr_.empty() ? 0 : fx_thr_[0].size()) < n_force) {
        for (int t = 0; t < n_threads_; ++t) {
            fx_thr_[t].resize(n_force, 0.0);
            fy_thr_[t].resize(n_force, 0.0);
            fz_thr_[t].resize(n_force, 0.0);
        }
    }

    const int k_inum = pairs.inum;
    const int* __restrict ilist = pairs.ilist.data();
    const int* __restrict numneigh = pairs.numneigh.data();
    const int* __restrict firstneigh = pairs.firstneigh.data();
    const int* __restrict neighbors = pairs.neighbors.data();
    const bool has_shifts = pairs.has_shifts;
    const double* __restrict shx = has_shifts ? pairs.shift_x.data() : nullptr;
    const double* __restrict shy = has_shifts ? pairs.shift_y.data() : nullptr;
    const double* __restrict shz = has_shifts ? pairs.shift_z.data() : nullptr;

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    AccReal* __restrict fx_ptr = f.span(kAtom_fx);
    AccReal* __restrict fy_ptr = f.span(kAtom_fy);
    AccReal* __restrict fz_ptr = f.span(kAtom_fz);

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

    double energy_total = 0;
    double dU_total = 0;
    double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;

    if (!(lam > 0.0) && soft0 == 0.0) {
        dU_dlambda_ = 0;
        return;
    }

#pragma omp parallel reduction(+ : energy_total) reduction(+ : dU_total)       \
    reduction(+ : v_xx) reduction(+ : v_yy) reduction(+ : v_zz)                \
    reduction(+ : v_xy) reduction(+ : v_xz) reduction(+ : v_yz)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const bool direct = (n_threads_ == 1);
        double* __restrict fxt_ptr;
        double* __restrict fyt_ptr;
        double* __restrict fzt_ptr;
        if (direct) {
            fxt_ptr = fx_ptr;
            fyt_ptr = fy_ptr;
            fzt_ptr = fz_ptr;
        } else {
            auto& fxt = fx_thr_[tid];
            auto& fyt = fy_thr_[tid];
            auto& fzt = fz_thr_[tid];
            std::fill(fxt.begin(), fxt.begin() + n_force, 0.0);
            std::fill(fyt.begin(), fyt.begin() + n_force, 0.0);
            std::fill(fzt.begin(), fzt.begin() + n_force, 0.0);
            fxt_ptr = fxt.data();
            fyt_ptr = fyt.data();
            fzt_ptr = fzt.data();
        }

        const double lx = box.length(0), ly = box.length(1), lz = box.length(2);
        const double inv_lx = 1.0 / lx, inv_ly = 1.0 / ly, inv_lz = 1.0 / lz;

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
                if (has_shifts) {
                    dx = xi - (x_ptr[j] + shx[off + jj]);
                    dy = yi - (y_ptr[j] + shy[off + jj]);
                    dz = zi - (z_ptr[j] + shz[off + jj]);
                } else {
                    dx = xi - x_ptr[j];
                    dy = yi - y_ptr[j];
                    dz = zi - z_ptr[j];
                    dx -= lx * std::nearbyint(dx * inv_lx);
                    dy -= ly * std::nearbyint(dy * inv_ly);
                    dz -= lz * std::nearbyint(dz * inv_lz);
                }
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
                    fxt_ptr[j] -= fxp;
                    fyt_ptr[j] -= fyp;
                    fzt_ptr[j] -= fzp;

                    v_xx += dx * fxp;
                    v_yy += dy * fyp;
                    v_zz += dz * fzp;
                    v_xy += dx * fyp;
                    v_xz += dx * fzp;
                    v_yz += dy * fzp;
                }
            }
            fxt_ptr[i] += fi_x;
            fyt_ptr[i] += fi_y;
            fzt_ptr[i] += fi_z;
        }

        if (!direct) {
#pragma omp for schedule(static)
            for (int i = 0; i < n_force; ++i) {
                for (int t = 0; t < n_threads_; ++t) {
                    fx_ptr[i] += fx_thr_[t][i];
                    fy_ptr[i] += fy_thr_[t][i];
                    fz_ptr[i] += fz_thr_[t][i];
                }
            }
        }
    }

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyPair) += energy_total;
    f.meta.get(kStress)[0] += v_xx;
    f.meta.get(kStress)[1] += v_yy;
    f.meta.get(kStress)[2] += v_zz;
    f.meta.get(kStress)[3] += v_xy;
    f.meta.get(kStress)[4] += v_xz;
    f.meta.get(kStress)[5] += v_yz;
    dU_dlambda_ = dU_total;
}

} // namespace cpu
} // namespace ATV
