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
#include "atomiverse/cpu/potentials/pair_loop.h"

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
    : PairPotential(cutoff), cutoff_sq_(cutoff * cutoff) {
    if (cutoff <= 0.0)
        throw std::invalid_argument("PairCoulCut: cutoff must be > 0");
}

PairCoulCut& PairCoulCut::set_coulomb_constant(double k) & {
    if (!(k > 0.0) || !std::isfinite(k))
        throw std::invalid_argument(
            "PairCoulCut: the Coulomb constant must be finite and > 0");
    coulomb_ = k;
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
    // LAMMPS `force->qqrd2e` of the run unless the model states its own.
    qqr2e_ = coulomb_ > 0.0 ? coulomb_ : ctx.runtime.units.qqrd2e();
    filtered_source_ = nullptr;
    loop_.reset();

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

namespace {

/// coul/cut: E = k q_i q_j / r inside the cutoff, scaled by the special
/// weight of the walked CSR slot. The row walk is an xsimd ket loop (as in
/// PairLJCut; NEON64 W=2, AVX2 W=4, AVX-512 W=8; W=1 degenerates to the
/// scalar loop): W partners at a time with the same arithmetic as the scalar
/// tail, out-of-cutoff (and coincident) lanes masked to zero, batch
/// accumulators reduced once per row.
struct CoulCutKernel {
    const StorageReal* px = nullptr;
    const StorageReal* py = nullptr;
    const StorageReal* pz = nullptr;
    const StorageReal* q = nullptr;
    const double* scales = nullptr; ///< per CSR slot; null: weight 1
    double rc_sq = 0, qqr2e = 1;

    struct Row {
        double qqi; ///< k · q_i
    };

    double cutoff_sq() const { return rc_sq; }
    Row row(int i) const { return {qqr2e * q[i]}; }

    md::PairTerm term(const Row& ri, int j, int entry, double r2) const {
        const double sc = scales ? scales[entry] : 1.0;
        // E = k qi qj / r, F = E / r^2 * d: one sqrt, one divide.
        const double rinv = 1.0 / std::sqrt(r2);
        const double prefactor = sc * ri.qqi * q[j] * rinv;
        return {prefactor * rinv * rinv, prefactor};
    }

    md::PairTerm operator()(const Row& ri, const md::PairSite& p,
                            const md::PairChunk&) const {
        return term(ri, p.j, p.entry, p.r2);
    }

    void walk(const md::PairRows& L, int ii_begin, int ii_end,
              md::ForceRows rows, md::PairAcc& acc, bool do_forces,
              bool do_virial) const {
        // Restrict-qualified locals: the scatter must not alias the
        // coordinates for the loads to stay hoisted.
        const StorageReal* __restrict x = px;
        const StorageReal* __restrict y = py;
        const StorageReal* __restrict z = pz;
        double* __restrict fx = rows.fx;
        double* __restrict fy = rows.fy;
        double* __restrict fz = rows.fz;
        using batch_t = xsimd::batch<double>;
        constexpr int W = static_cast<int>(batch_t::size);
        const batch_t v_rc_sq(rc_sq), v_one(1.0), v_zero(0.0);
        alignas(64) double xj_buf[16], yj_buf[16], zj_buf[16], qj_buf[16];
        alignas(64) double fxp_buf[16], fyp_buf[16], fzp_buf[16];

        for (int ii = ii_begin; ii < ii_end; ++ii) {
            const int i = L.ilist[ii];
            const Row ri = row(i);
            if (ri.qqi == 0.0)
                continue;
            const int jnum = L.numneigh[i];
            const int entry0 = L.firstneigh[i];
            const int* __restrict nb = L.neighbors + entry0;
            const double xi = x[i], yi = y[i], zi = z[i];
            const double qqi = ri.qqi;
            const double* __restrict sc_row =
                scales ? scales + entry0 : nullptr;
            int jj = 0;
            const batch_t v_xi(xi), v_yi(yi), v_zi(zi), v_qqi(qqi);
            batch_t acc_fx(0.0), acc_fy(0.0), acc_fz(0.0), acc_e(0.0);
            batch_t acc_xx(0.0), acc_yy(0.0), acc_zz(0.0);
            batch_t acc_xy(0.0), acc_xz(0.0), acc_yz(0.0);
            for (; jj + W <= jnum; jj += W) {
                for (int lane = 0; lane < W; ++lane) {
                    const int j = nb[jj + lane];
                    xj_buf[lane] = x[j];
                    yj_buf[lane] = y[j];
                    zj_buf[lane] = z[j];
                    qj_buf[lane] = q[j];
                }
                const batch_t v_dx = v_xi - batch_t::load_aligned(xj_buf);
                const batch_t v_dy = v_yi - batch_t::load_aligned(yj_buf);
                const batch_t v_dz = v_zi - batch_t::load_aligned(zj_buf);
                const batch_t v_r2 = v_dx * v_dx + v_dy * v_dy + v_dz * v_dz;
                const auto mask = (v_r2 < v_rc_sq) && (v_r2 != v_zero);
                const batch_t v_sc =
                    sc_row ? batch_t::load_unaligned(sc_row + jj) : v_one;
                const batch_t v_rinv =
                    v_one / xsimd::sqrt(xsimd::select(mask, v_r2, v_one));
                const batch_t v_pref = xsimd::select(
                    mask, v_sc * v_qqi * batch_t::load_aligned(qj_buf) * v_rinv,
                    v_zero);
                const batch_t v_fpair = v_pref * v_rinv * v_rinv;
                const batch_t v_fxp = v_dx * v_fpair;
                const batch_t v_fyp = v_dy * v_fpair;
                const batch_t v_fzp = v_dz * v_fpair;
                acc_fx = acc_fx + v_fxp;
                acc_fy = acc_fy + v_fyp;
                acc_fz = acc_fz + v_fzp;
                acc_e = acc_e + v_pref;
                // Unconditional batch virial: see PairLJCut's ket loop for why a
                // branch here is not worth its risk.
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
                        fx[j] -= fxp_buf[lane];
                        fy[j] -= fyp_buf[lane];
                        fz[j] -= fzp_buf[lane];
                    }
                }
            }
            double fi_x = xsimd::reduce_add(acc_fx);
            double fi_y = xsimd::reduce_add(acc_fy);
            double fi_z = xsimd::reduce_add(acc_fz);
            double e_i = xsimd::reduce_add(acc_e);
            double a_xx = 0, a_yy = 0, a_zz = 0, a_xy = 0, a_xz = 0, a_yz = 0;
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
                const double dx = xi - x[j];
                const double dy = yi - y[j];
                const double dz = zi - z[j];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc_sq || r2 == 0.0)
                    continue;
                const md::PairTerm t = term(ri, j, entry0 + jj, r2);
                const double fxp = dx * t.fpair;
                const double fyp = dy * t.fpair;
                const double fzp = dz * t.fpair;
                fi_x += fxp;
                fi_y += fyp;
                fi_z += fzp;
                if (do_forces) {
                    fx[j] -= fxp;
                    fy[j] -= fyp;
                    fz[j] -= fzp;
                }
                if (do_virial) {
                    a_xx += dx * fxp;
                    a_yy += dy * fyp;
                    a_zz += dz * fzp;
                    a_xy += dx * fyp;
                    a_xz += dx * fzp;
                    a_yz += dy * fzp;
                }
                e_i += t.energy;
            }
            if (do_forces) {
                fx[i] += fi_x;
                fy[i] += fi_y;
                fz[i] += fi_z;
            }
            acc.energy += e_i;
            acc.virial.v[0] += a_xx;
            acc.virial.v[1] += a_yy;
            acc.virial.v[2] += a_zz;
            acc.virial.v[3] += a_xy;
            acc.virial.v[4] += a_xz;
            acc.virial.v[5] += a_yz;
        }
    }
};

} // namespace

void PairCoulCut::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairCoulCut::launch: runtime.md.pair_list is null");
    }
    const auto& pairs = *ctx.runtime.md.pair_list;

    const bool use_filtered = any_special_;
    if (use_filtered
        && (filtered_source_ != &pairs || pairs.version != filtered_version_)) {
        rebuild_filtered_list(f, pairs);
        filtered_version_ = pairs.version;
        filtered_source_ = &pairs;
    }
    const md::PairRows rows = use_filtered
                                  ? md::PairRows{filtered_inum_,
                                                 pairs.ilist.data(),
                                                 filtered_numneigh_.data(),
                                                 filtered_firstneigh_.data(),
                                                 filtered_neighbors_.data(),
                                                 &filtered_neighbors_,
                                                 pairs.version}
                                  : md::PairRows::of(pairs);

    CoulCutKernel kernel;
    kernel.px = f.span(kAtom_x);
    kernel.py = f.span(kAtom_y);
    kernel.pz = f.span(kAtom_z);
    kernel.q = f.span(kAtom_charge);
    kernel.scales = use_filtered ? filtered_scale_.data() : nullptr;
    kernel.rc_sq = cutoff_sq_;
    kernel.qqr2e = qqr2e_;
    const md::PairTotals sums = loop_.run(ctx, rows, kernel);

    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));
}

void PairCoulCut::destroy(const DestroyContext&) {
    special_.clear();
    filtered_neighbors_.clear();
    filtered_scale_.clear();
    filtered_numneigh_.clear();
    filtered_firstneigh_.clear();
    filtered_inum_ = 0;
    filtered_version_ = 0;
    filtered_source_ = nullptr;
    loop_.reset();
    any_special_ = false;
}

} // namespace cpu
} // namespace ATV
