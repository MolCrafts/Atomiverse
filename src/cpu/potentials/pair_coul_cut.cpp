/// @file src/cpu/potentials/pair_coul_cut.cpp
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/component/owned_range.h"

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

// Ket-loop vectorization, as in PairLJCut (NEON64 W=2, AVX2 W=4, AVX-512 W=8;
// W=1 degenerates to the scalar loop).
#include <xsimd/xsimd.hpp>

#include "atomiverse/cpu/communication/domain.h"

namespace ATV {
namespace cpu {

PairCoulCut::PairCoulCut(double cutoff)
    : PairPotential(cutoff), cutoff_sq_(cutoff * cutoff),
      qqr2e_(ATV::units::coulomb) {
    if (cutoff <= 0.0)
        throw std::invalid_argument("PairCoulCut: cutoff must be > 0");
}

PairCoulCut& PairCoulCut::set_coulomb_constant(double k) & {
    if (!(k > 0.0) || !std::isfinite(k))
        throw std::invalid_argument(
            "PairCoulCut: the Coulomb constant must be finite and > 0");
    qqr2e_ = k;
    return *this;
}

PairCoulCut PairCoulCut::set_coulomb_constant(double k) && {
    set_coulomb_constant(k);
    return std::move(*this);
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
    filtered_source_ = nullptr;
    part_source_ = nullptr;
    n_threads_ = 1;
#ifdef _OPENMP
    n_threads_ = omp_get_max_threads();
#endif
    chunks_.clear(); // sized on demand by compute()

    special_.s12 = s12_;
    special_.s13 = s13_;
    special_.s14 = s14_;
    any_special_ = special_.active();
    special_.clear();
    if (!any_special_)
        return;
    const int n_owners = ctx.frame.nlocal();
    special_.rebuild(ctx.frame, n_owners);
}

void PairCoulCut::rebuild_filtered_list(const Frame& f,
                                        const md::PairList& pairs) {
    const int n = pairs.inum;
    special_.s12 = s12_;
    special_.s13 = s13_;
    special_.s14 = s14_;
    special_.rebuild(f, f.nlocal());

    filtered_inum_ = n;
    filtered_numneigh_.assign(f.n_total(), 0);
    filtered_firstneigh_.assign(f.n_total(), 0);
    filtered_neighbors_.clear();
    filtered_scale_.clear();
    filtered_neighbors_.reserve(pairs.total_pairs);
    filtered_scale_.reserve(pairs.total_pairs);

    const int* numneigh = pairs.numneigh.data();
    const int* firstneigh = pairs.firstneigh.data();
    const int* neighbors = pairs.neighbors.data();
    const TagInt* __restrict tag = f.span(kAtom_tag);

    for (int ii = 0; ii < n; ++ii) {
        const int i = pairs.ilist[ii];
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

    const int n_force = cpu::force_count(ctx);

    const bool use_filtered = any_special_;
    if (use_filtered
        && (filtered_source_ != &pairs || pairs.version != filtered_version_)) {
        rebuild_filtered_list(f, pairs);
        filtered_version_ = pairs.version;
        filtered_source_ = &pairs;
    }
    const int k_inum = use_filtered ? filtered_inum_ : pairs.inum;
    const int* __restrict ilist = pairs.ilist.data();
    const int* __restrict numneigh =
        use_filtered ? filtered_numneigh_.data() : pairs.numneigh.data();
    const int* __restrict firstneigh =
        use_filtered ? filtered_firstneigh_.data() : pairs.firstneigh.data();
    const int* __restrict neighbors =
        use_filtered ? filtered_neighbors_.data() : pairs.neighbors.data();
    const double* __restrict scales =
        use_filtered ? filtered_scale_.data() : nullptr;

    // Team size, balanced rows and the touched-row bound follow the list.
    if (part_source_ != &pairs || pairs.version != part_version_) {
        partition_.rebuild(k_inum, ilist, numneigh, firstneigh, neighbors);
        part_source_ = &pairs;
        part_version_ = pairs.version;
    }
    const int team = partition_.team(n_threads_);
    // One thread accumulates straight into the frame (already zeroed by
    // ZeroForces / the halo), so neither the scratch zero nor the reduce runs.
    const bool direct = team == 1;
    // Rows past the list's highest reference receive nothing.
    const int n_touch = std::min(partition_.n_touch(), n_force);
    // Chunks are handed out dynamically; each owns its buffer and sums, so
    // the result does not depend on the thread that ran it.
    const int n_chunks = partition_.chunks(team);
    chunks_.prepare(n_chunks, direct ? 0 : n_touch);

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    const StorageReal* __restrict q_ptr = f.span(kAtom_charge);
    AccReal* __restrict fx_ptr = f.span(kAtom_fx);
    AccReal* __restrict fy_ptr = f.span(kAtom_fy);
    AccReal* __restrict fz_ptr = f.span(kAtom_fz);

    const double rc_sq = cutoff_sq_;
    const double qqr2e = qqr2e_;
    const bool do_forces = ctx.forces_requested;
    const bool do_virial = ctx.stress_requested;

#pragma omp parallel num_threads(team) if (team > 1)
    {
        using batch_t = xsimd::batch<double>;
        constexpr int W = static_cast<int>(batch_t::size);
        const batch_t v_rc_sq(rc_sq), v_one(1.0), v_zero(0.0);
        alignas(64) double xj_buf[16], yj_buf[16], zj_buf[16], qj_buf[16];
        alignas(64) double fxp_buf[16], fyp_buf[16], fzp_buf[16];

#pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < n_chunks; ++c) {
            double* fxc = fx_ptr;
            double* fyc = fy_ptr;
            double* fzc = fz_ptr;
            if (!direct)
                chunks_.begin_chunk(c, fxc, fyc, fzc);
            double* __restrict fxt = fxc;
            double* __restrict fyt = fyc;
            double* __restrict fzt = fzc;
            // Chunk-local sums, kept in registers across the force scatter.
            double e_thr = 0;
            double t_xx = 0, t_yy = 0, t_zz = 0, t_xy = 0, t_xz = 0, t_yz = 0;
            const auto [ii_begin, ii_end] = partition_.range(c, n_chunks);
            for (int ii = ii_begin; ii < ii_end; ++ii) {
                const int i = ilist[ii];
                const double xi = x_ptr[i], yi = y_ptr[i], zi = z_ptr[i];
                const double qi = q_ptr[i];
                if (qi == 0.0)
                    continue;
                const double qqi = qqr2e * qi;
                const int jnum = numneigh[i];
                const int off = firstneigh[i];
                const int* __restrict nb = neighbors + off;
                const double* __restrict sc_row =
                    scales ? scales + off : nullptr;
                int jj = 0;
                // ── SIMD ket loop: W partners at a time, same arithmetic as the
                // scalar tail; out-of-cutoff (and coincident) lanes are masked
                // to zero. Batch accumulators are reduced once per atom i.
                const batch_t v_xi(xi), v_yi(yi), v_zi(zi), v_qqi(qqi);
                batch_t acc_fx(0.0), acc_fy(0.0), acc_fz(0.0), acc_e(0.0);
                batch_t acc_xx(0.0), acc_yy(0.0), acc_zz(0.0);
                batch_t acc_xy(0.0), acc_xz(0.0), acc_yz(0.0);
                for (; jj + W <= jnum; jj += W) {
                    for (int lane = 0; lane < W; ++lane) {
                        const int j = nb[jj + lane];
                        xj_buf[lane] = x_ptr[j];
                        yj_buf[lane] = y_ptr[j];
                        zj_buf[lane] = z_ptr[j];
                        qj_buf[lane] = q_ptr[j];
                    }
                    const batch_t v_dx = v_xi - batch_t::load_aligned(xj_buf);
                    const batch_t v_dy = v_yi - batch_t::load_aligned(yj_buf);
                    const batch_t v_dz = v_zi - batch_t::load_aligned(zj_buf);
                    const batch_t v_r2 =
                        v_dx * v_dx + v_dy * v_dy + v_dz * v_dz;
                    const auto mask = (v_r2 < v_rc_sq) && (v_r2 != v_zero);
                    const batch_t v_sc =
                        sc_row ? batch_t::load_unaligned(sc_row + jj) : v_one;
                    const batch_t v_rinv =
                        v_one / xsimd::sqrt(xsimd::select(mask, v_r2, v_one));
                    const batch_t v_pref = xsimd::select(
                        mask,
                        v_sc * v_qqi * batch_t::load_aligned(qj_buf) * v_rinv,
                        v_zero);
                    const batch_t v_fpair = v_pref * v_rinv * v_rinv;
                    const batch_t v_fxp = v_dx * v_fpair;
                    const batch_t v_fyp = v_dy * v_fpair;
                    const batch_t v_fzp = v_dz * v_fpair;
                    acc_fx = acc_fx + v_fxp;
                    acc_fy = acc_fy + v_fyp;
                    acc_fz = acc_fz + v_fzp;
                    acc_e = acc_e + v_pref;
                    // Unconditional batch virial: see PairLJCut's ket loop for
                    // why a branch here is not worth its risk.
                    acc_xx = acc_xx + v_dx * v_fxp;
                    acc_yy = acc_yy + v_dy * v_fyp;
                    acc_zz = acc_zz + v_dz * v_fzp;
                    acc_xy = acc_xy + v_dx * v_fyp;
                    acc_xz = acc_xz + v_dx * v_fzp;
                    acc_yz = acc_yz + v_dy * v_fzp;
                    if (do_forces) {
                        v_fxp.store_aligned(fxp_buf);
                        v_fyp.store_aligned(fyp_buf);
                        v_fzp.store_aligned(fzp_buf);
                        for (int lane = 0; lane < W; ++lane) {
                            const int j = nb[jj + lane];
                            fxt[j] -= fxp_buf[lane];
                            fyt[j] -= fyp_buf[lane];
                            fzt[j] -= fzp_buf[lane];
                        }
                    }
                }
                double fi_x = xsimd::reduce_add(acc_fx);
                double fi_y = xsimd::reduce_add(acc_fy);
                double fi_z = xsimd::reduce_add(acc_fz);
                double e_i = xsimd::reduce_add(acc_e);
                double a_xx = 0, a_yy = 0, a_zz = 0, a_xy = 0, a_xz = 0,
                       a_yz = 0;
                if (do_virial) {
                    a_xx = xsimd::reduce_add(acc_xx);
                    a_yy = xsimd::reduce_add(acc_yy);
                    a_zz = xsimd::reduce_add(acc_zz);
                    a_xy = xsimd::reduce_add(acc_xy);
                    a_xz = xsimd::reduce_add(acc_xz);
                    a_yz = xsimd::reduce_add(acc_yz);
                }
                // ── Scalar tail (last jnum % W partners).
                for (; jj < jnum; ++jj) {
                    const int j = nb[jj];
                    const double dx = xi - x_ptr[j];
                    const double dy = yi - y_ptr[j];
                    const double dz = zi - z_ptr[j];
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= rc_sq || r2 == 0.0)
                        continue;
                    const double sc = sc_row ? sc_row[jj] : 1.0;
                    // E = k qi qj / r, F = E / r^2 * d: one sqrt, one divide.
                    const double rinv = 1.0 / std::sqrt(r2);
                    const double prefactor = sc * qqi * q_ptr[j] * rinv;
                    const double fpair = prefactor * rinv * rinv;

                    const double fxp = dx * fpair;
                    const double fyp = dy * fpair;
                    const double fzp = dz * fpair;
                    fi_x += fxp;
                    fi_y += fyp;
                    fi_z += fzp;
                    if (do_forces) {
                        fxt[j] -= fxp;
                        fyt[j] -= fyp;
                        fzt[j] -= fzp;
                    }
                    if (do_virial) {
                        a_xx += dx * fxp;
                        a_yy += dy * fyp;
                        a_zz += dz * fzp;
                        a_xy += dx * fyp;
                        a_xz += dx * fzp;
                        a_yz += dy * fzp;
                    }
                    e_i += prefactor;
                }
                if (do_forces) {
                    fxt[i] += fi_x;
                    fyt[i] += fi_y;
                    fzt[i] += fi_z;
                }
                e_thr += e_i;
                t_xx += a_xx;
                t_yy += a_yy;
                t_zz += a_zz;
                t_xy += a_xy;
                t_xz += a_xz;
                t_yz += a_yz;
            }
            double* sums = chunks_.sums(c);
            sums[0] = e_thr;
            sums[1] = t_xx;
            sums[2] = t_yy;
            sums[3] = t_zz;
            sums[4] = t_xy;
            sums[5] = t_xz;
            sums[6] = t_yz;
        } // chunks (implicit barrier: every scatter has landed)

        if (!direct && do_forces)
            chunks_.reduce_forces(fx_ptr, fy_ptr, fz_ptr);
    }

    const double energy_total = chunks_.sum(0);
    const double v_xx = chunks_.sum(1), v_yy = chunks_.sum(2),
                 v_zz = chunks_.sum(3), v_xy = chunks_.sum(4),
                 v_xz = chunks_.sum(5), v_yz = chunks_.sum(6);
    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyPair) += energy_total;
    if (do_virial) {
        f.meta.get(kStress)[0] += v_xx;
        f.meta.get(kStress)[1] += v_yy;
        f.meta.get(kStress)[2] += v_zz;
        f.meta.get(kStress)[3] += v_xy;
        f.meta.get(kStress)[4] += v_xz;
        f.meta.get(kStress)[5] += v_yz;
    }
}

void PairCoulCut::destroy(const DestroyContext&) {
    chunks_.clear();
    special_.clear();
    filtered_neighbors_.clear();
    filtered_scale_.clear();
    filtered_numneigh_.clear();
    filtered_firstneigh_.clear();
    filtered_inum_ = 0;
    filtered_version_ = 0;
    filtered_source_ = nullptr;
    part_source_ = nullptr;
    any_special_ = false;
}

} // namespace cpu
} // namespace ATV
