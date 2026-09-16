/// @file src/cpu/potentials/pair_coul_cut.cpp
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/cpu/communication/domain.h"

namespace ATV {
namespace cpu {

PairCoulCut::PairCoulCut(double cutoff)
    : PairPotential(cutoff), cutoff_sq_(cutoff * cutoff) {
    if (cutoff <= 0.0)
        throw std::invalid_argument("PairCoulCut: cutoff must be > 0");
}

PairCoulCut& PairCoulCut::set_special_bonds(double s12, double s13,
                                            double s14) & {
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    return *this;
}

PairCoulCut PairCoulCut::set_special_bonds(double s12, double s13,
                                           double s14) && {
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    return std::move(*this);
}

void PairCoulCut::build(const BuildContext& ctx) {
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
    if (!any_special_)
        return;
    const auto* dd = ctx.runtime.md.dd;
    const int n_owners =
        (dd && ctx.runtime.mpi_size > 1) ? dd->n_local : ctx.frame.nlocal();
    special_.rebuild(ctx.frame, n_owners);
}

void PairCoulCut::rebuild_filtered_list(const Frame& f,
                                        const md::PairList& pairs) {
    const int n = pairs.inum;
    special_.s12 = s12_;
    special_.s13 = s13_;
    special_.s14 = s14_;
    special_.rebuild(f, n);

    filtered_inum_ = n;
    filtered_numneigh_.assign(n, 0);
    filtered_firstneigh_.assign(n, 0);
    filtered_neighbors_.clear();
    filtered_scale_.clear();
    filtered_neighbors_.reserve(pairs.total_pairs);
    filtered_scale_.reserve(pairs.total_pairs);
    filtered_has_shifts_ = pairs.has_shifts;
    if (filtered_has_shifts_) {
        filtered_shift_x_.clear();
        filtered_shift_y_.clear();
        filtered_shift_z_.clear();
        filtered_shift_x_.reserve(pairs.total_pairs);
        filtered_shift_y_.reserve(pairs.total_pairs);
        filtered_shift_z_.reserve(pairs.total_pairs);
    }

    const int* numneigh = pairs.numneigh.data();
    const int* firstneigh = pairs.firstneigh.data();
    const int* neighbors = pairs.neighbors.data();
    const double* sx = filtered_has_shifts_ ? pairs.shift_x.data() : nullptr;
    const double* sy = filtered_has_shifts_ ? pairs.shift_y.data() : nullptr;
    const double* sz = filtered_has_shifts_ ? pairs.shift_z.data() : nullptr;
    const TagInt* __restrict tag = f.span(kAtom_tag);

    for (int i = 0; i < n; ++i) {

        const int off = firstneigh[i];
        const int jn = numneigh[i];
        filtered_firstneigh_[i] = static_cast<int>(filtered_neighbors_.size());
        int kept = 0;
        for (int jj = 0; jj < jn; ++jj) {
            const int j = neighbors[off + jj];
            const double sc = special_.scale_owner(i, tag[j]);
            if (sc == 0.0)
                continue;
            filtered_neighbors_.push_back(j);
            filtered_scale_.push_back(sc);
            if (filtered_has_shifts_) {
                filtered_shift_x_.push_back(sx[off + jj]);
                filtered_shift_y_.push_back(sy[off + jj]);
                filtered_shift_z_.push_back(sz[off + jj]);
            }
            ++kept;
        }
        filtered_numneigh_[i] = kept;
    }
}

void PairCoulCut::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairCoulCut::launch: runtime.md.pair_list is null");
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

    const bool use_filtered = any_special_;
    if (use_filtered && pairs.version != filtered_version_) {
        rebuild_filtered_list(f, pairs);
        filtered_version_ = pairs.version;
    }
    const int k_inum = use_filtered ? filtered_inum_ : pairs.inum;
    const int* __restrict ilist = use_filtered ? nullptr : pairs.ilist.data();
    const int* __restrict numneigh =
        use_filtered ? filtered_numneigh_.data() : pairs.numneigh.data();
    const int* __restrict firstneigh =
        use_filtered ? filtered_firstneigh_.data() : pairs.firstneigh.data();
    const int* __restrict neighbors =
        use_filtered ? filtered_neighbors_.data() : pairs.neighbors.data();
    const double* __restrict scales =
        use_filtered ? filtered_scale_.data() : nullptr;
    const bool has_shifts =
        use_filtered ? filtered_has_shifts_ : pairs.has_shifts;
    const double* __restrict shx =
        !has_shifts
            ? nullptr
            : (use_filtered ? filtered_shift_x_.data() : pairs.shift_x.data());
    const double* __restrict shy =
        !has_shifts
            ? nullptr
            : (use_filtered ? filtered_shift_y_.data() : pairs.shift_y.data());
    const double* __restrict shz =
        !has_shifts
            ? nullptr
            : (use_filtered ? filtered_shift_z_.data() : pairs.shift_z.data());

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    const StorageReal* __restrict q_ptr = f.span(kAtom_charge);
    AccReal* __restrict fx_ptr = f.span(kAtom_fx);
    AccReal* __restrict fy_ptr = f.span(kAtom_fy);
    AccReal* __restrict fz_ptr = f.span(kAtom_fz);

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
        for (int ii = 0; ii < k_inum; ++ii) {
            const int i = use_filtered ? ii : ilist[ii];
            const double xi = x_ptr[i], yi = y_ptr[i], zi = z_ptr[i];
            const double qi = q_ptr[i];
            if (qi == 0.0)
                continue;
            const int jnum = numneigh[i];
            const int off = firstneigh[i];
            double fi_x = 0, fi_y = 0, fi_z = 0;
            for (int jj = 0; jj < jnum; ++jj) {
                const int j = neighbors[off + jj];
                const double sc = scales ? scales[off + jj] : 1.0;
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
                const double prefactor = sc * qqr2e * qi * qj / r;
                const double fpair = prefactor * r2inv;

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

                energy_total += prefactor;
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

void PairCoulCut::destroy(const DestroyContext&) {
    fx_thr_.clear();
    fy_thr_.clear();
    fz_thr_.clear();
    special_.clear();
    filtered_neighbors_.clear();
    filtered_scale_.clear();
    filtered_numneigh_.clear();
    filtered_firstneigh_.clear();
    filtered_shift_x_.clear();
    filtered_shift_y_.clear();
    filtered_shift_z_.clear();
    filtered_inum_ = 0;
    filtered_version_ = 0;
    any_special_ = false;
}

} // namespace cpu
} // namespace ATV
