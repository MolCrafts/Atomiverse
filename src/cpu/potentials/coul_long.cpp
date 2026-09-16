/// @file src/cpu/potentials/coul_long.cpp
/// @brief Real-space screened Coulomb implementation (CPU).
#include "atomiverse/cpu/potentials/coul_long.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ATV_HAS_MPI
#include "atomiverse/cpu/communication/domain.h"
#endif

namespace ATV {
namespace cpu {

static constexpr double kEwaldF = 1.1283791670955126;

CoulLong::CoulLong(double alpha, double cutoff)
    : alpha_(alpha), cutoff_(cutoff), cutoff_sq_(cutoff * cutoff) {
    if (alpha <= 0.0)
        throw std::invalid_argument("CoulLong: alpha must be > 0");
    if (cutoff <= 0.0)
        throw std::invalid_argument("CoulLong: cutoff must be > 0");
}

CoulLong& CoulLong::set_special_bonds(double s12, double s13, double s14) & {
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    return *this;
}

CoulLong CoulLong::set_special_bonds(double s12, double s13, double s14) && {
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    return std::move(*this);
}

void CoulLong::build(const BuildContext& ctx) {
    n_threads_ = 1;
#ifdef _OPENMP
    n_threads_ = omp_get_max_threads();
#endif
    const int n = ctx.frame.capacity();
    fx_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
    fy_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
    fz_thr_.assign(n_threads_, std::vector<double>(n, 0.0));

    special_.s12 = s12_;
    special_.s13 = s13_;
    special_.s14 = s14_;
    any_special_ = special_.active();
    special_.clear();
    special_version_ = 0;
    if (!any_special_)
        return;
    const auto* dd = ctx.runtime.md.dd;
    const int n_owners =
        (dd && ctx.runtime.mpi_size > 1) ? dd->n_local : ctx.frame.nlocal();
    special_.rebuild(ctx.frame, n_owners);
}

void CoulLong::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error("CoulLong::launch: runtime.md.pair_list is "
                                 "null. Add a NeighborList "
                                 "to the force phase ahead of CoulLong.");
    }
    const auto& pairs = *ctx.runtime.md.pair_list;
    const auto& box = f.box;

    const auto* dd = ctx.runtime.md.dd;
    const bool mpi = dd && ctx.runtime.mpi_size > 1;
    const int n_force = mpi ? dd->n_local + dd->n_ghost : f.nlocal();

    if (static_cast<int>(fx_thr_.empty() ? 0 : fx_thr_[0].size()) < n_force) {
        for (int t = 0; t < n_threads_; ++t) {
            fx_thr_[t].resize(n_force, 0.0);
            fy_thr_[t].resize(n_force, 0.0);
            fz_thr_[t].resize(n_force, 0.0);
        }
    }

    if (any_special_ && pairs.version != special_version_) {
        special_.s12 = s12_;
        special_.s13 = s13_;
        special_.s14 = s14_;
        special_.rebuild(f, pairs.inum);
        special_version_ = pairs.version;
    }

    const int* __restrict ilist = pairs.ilist.data();
    const int* __restrict numneigh = pairs.numneigh.data();
    const int* __restrict firstneigh = pairs.firstneigh.data();
    const int* __restrict neighbors = pairs.neighbors.data();
    const bool has_shifts = pairs.has_shifts;
    const double* __restrict shx = has_shifts ? pairs.shift_x.data() : nullptr;
    const double* __restrict shy = has_shifts ? pairs.shift_y.data() : nullptr;
    const double* __restrict shz = has_shifts ? pairs.shift_z.data() : nullptr;
    const TagInt* __restrict tag_ptr =
        any_special_ ? f.span(kAtom_tag).data() : nullptr;

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    const StorageReal* __restrict q_ptr = f.span(kAtom_charge);
    AccReal* __restrict fx_ptr = f.span(kAtom_fx);
    AccReal* __restrict fy_ptr = f.span(kAtom_fy);
    AccReal* __restrict fz_ptr = f.span(kAtom_fz);

    const double alpha = alpha_;
    const double rc_sq = cutoff_sq_;
    const double qqr2e = ATV::units::coulomb;

    double energy_total = 0;
    double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;

#pragma omp parallel reduction(+ : energy_total) reduction(+ : v_xx)           \
    reduction(+ : v_yy) reduction(+ : v_zz) reduction(+ : v_xy)                \
    reduction(+ : v_xz) reduction(+ : v_yz)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double* __restrict fxt = fx_thr_[tid].data();
        double* __restrict fyt = fy_thr_[tid].data();
        double* __restrict fzt = fz_thr_[tid].data();
        std::fill(fx_thr_[tid].begin(), fx_thr_[tid].begin() + n_force, 0.0);
        std::fill(fy_thr_[tid].begin(), fy_thr_[tid].begin() + n_force, 0.0);
        std::fill(fz_thr_[tid].begin(), fz_thr_[tid].begin() + n_force, 0.0);

        const double lx = box.length(0), ly = box.length(1), lz = box.length(2);
        const double inv_lx = 1.0 / lx, inv_ly = 1.0 / ly, inv_lz = 1.0 / lz;

#pragma omp for schedule(static)
        for (int ii = 0; ii < pairs.inum; ++ii) {
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
                double sc = 1.0;
                if (any_special_) {
                    // Zero-scale (1-2/1-3) pairs stay: with a reciprocal
                    // partner they owe the -(1-f)·qq/r Ewald compensation.
                    sc = special_.scale_owner(i, tag_ptr[j]);
                }
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
                const double qj = q_ptr[j];
                if (qj == 0.0)
                    continue;

                const double r = std::sqrt(r2);
                const double r2inv = 1.0 / r2;
                const double grij = alpha * r;
                const double expm2 = std::exp(-grij * grij);
                const double erfcv = std::erfc(grij);
                // LAMMPS special convention with a kspace partner: erfc part
                // unscaled; scaled/excluded bonded pairs subtract the (1-f)
                // plain-Coulomb share the reciprocal sum added for them.
                const double prefactor = qqr2e * qi * qj / r;
                double forcecoul = prefactor * (erfcv + kEwaldF * grij * expm2);
                if (sc != 1.0)
                    forcecoul -= (1.0 - sc) * prefactor;
                const double fpair = forcecoul * r2inv;

                const double fxp = dx * fpair;
                const double fyp = dy * fpair;
                const double fzp = dz * fpair;
                fi_x += fxp;
                fi_y += fyp;
                fi_z += fzp;
                fxt[j] -= fxp;
                fyt[j] -= fyp;
                fzt[j] -= fzp;

                v_xx += dx * fxp;
                v_yy += dy * fyp;
                v_zz += dz * fzp;
                v_xy += dx * fyp;
                v_xz += dx * fzp;
                v_yz += dy * fzp;

                energy_total += sc != 1.0
                                    ? prefactor * erfcv - (1.0 - sc) * prefactor
                                    : prefactor * erfcv;
            }
            fxt[i] += fi_x;
            fyt[i] += fi_y;
            fzt[i] += fi_z;
        }

#pragma omp for schedule(static)
        for (int i = 0; i < n_force; ++i) {
            for (int t = 0; t < n_threads_; ++t) {
                fx_ptr[i] += fx_thr_[t][i];
                fy_ptr[i] += fy_thr_[t][i];
                fz_ptr[i] += fz_thr_[t][i];
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
}

void CoulLong::destroy(const DestroyContext&) {
    fx_thr_.clear();
    fy_thr_.clear();
    fz_thr_.clear();
    special_.clear();
    any_special_ = false;
    special_version_ = 0;
}

} // namespace cpu
} // namespace ATV
