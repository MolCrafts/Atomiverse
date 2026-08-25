/// @file src/cpu/potentials/pair_lj_cut_coul_long.cpp
/// @brief Fused multi-type LJ + tabulated coul/long, one neighbor walk.
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

namespace {
inline void lorentz_berthelot(double eii, double sii, double ejj, double sjj,
                              double& eij, double& sij) {
    eij = std::sqrt(eii * ejj);
    sij = 0.5 * (sii + sjj);
}
} // namespace

PairLJCutCoulLong::PairLJCutCoulLong(std::vector<LJTypeCoeff> coeffs,
                                     double alpha, double cutoff)
    : Pair(cutoff), alpha_(alpha), cutoff_sq_(cutoff * cutoff) {
    if (coeffs.empty())
        throw std::invalid_argument("PairLJCutCoulLong: empty coeff table");
    if (!(alpha > 0.0))
        throw std::invalid_argument("PairLJCutCoulLong: alpha must be > 0");
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairLJCutCoulLong: cutoff must be > 0");

    int max_t = 0;
    for (const auto& c : coeffs) {
        if (c.type_i < 0 || c.type_j < 0)
            throw std::invalid_argument("PairLJCutCoulLong: negative type");
        if (!(c.sigma > 0.0) || !(c.epsilon >= 0.0))
            throw std::invalid_argument("PairLJCutCoulLong: bad epsilon/sigma");
        max_t = std::max(max_t, std::max(c.type_i, c.type_j));
    }
    n_types_ = max_t + 1;
    const std::size_t n2 =
        static_cast<std::size_t>(n_types_) * static_cast<std::size_t>(n_types_);
    eps_tab_.assign(n2, 0.0);
    sigma_tab_.assign(n2, 0.0);
    set_tab_.assign(n2, 0);
    for (const auto& c : coeffs) {
        auto put = [&](int a, int b) {
            const std::size_t ab = static_cast<std::size_t>(a) * n_types_
                                   + static_cast<std::size_t>(b);
            eps_tab_[ab] = c.epsilon;
            sigma_tab_[ab] = c.sigma;
            set_tab_[ab] = 1;
        };
        put(c.type_i, c.type_j);
        put(c.type_j, c.type_i);
    }
    set_special_bonds_amber();
}

PairLJCutCoulLong& PairLJCutCoulLong::set_special_bonds_amber() & {
    // LAMMPS `special_bonds amber`
    return set_special_bonds(0.0, 0.0, 0.5, 0.0, 0.0, 5.0 / 6.0);
}

PairLJCutCoulLong PairLJCutCoulLong::set_special_bonds_amber() && {
    set_special_bonds_amber();
    return std::move(*this);
}

PairLJCutCoulLong&
PairLJCutCoulLong::set_special_bonds(double lj12, double lj13, double lj14,
                                     double c12, double c13, double c14) & {
    lj12_ = lj12;
    lj13_ = lj13;
    lj14_ = lj14;
    c12_ = c12;
    c13_ = c13;
    c14_ = c14;
    any_special_ = (lj12 != 1.0 || lj13 != 1.0 || lj14 != 1.0 || c12 != 1.0
                    || c13 != 1.0 || c14 != 1.0);
    return *this;
}

PairLJCutCoulLong PairLJCutCoulLong::set_special_bonds(double lj12, double lj13,
                                                       double lj14, double c12,
                                                       double c13,
                                                       double c14) && {
    set_special_bonds(lj12, lj13, lj14, c12, c13, c14);
    return std::move(*this);
}

PairLJCutCoulLong& PairLJCutCoulLong::set_compute_stress(bool on) & {
    compute_stress_ = on;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_compute_stress(bool on) && {
    compute_stress_ = on;
    return std::move(*this);
}

PairLJCutCoulLong& PairLJCutCoulLong::set_energy_shift(bool on) & {
    energy_shift_ = on;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_energy_shift(bool on) && {
    energy_shift_ = on;
    return std::move(*this);
}

PairLJCutCoulLong& PairLJCutCoulLong::set_tail_correction(bool on) & {
    tail_ = on;
    tail_ready_ = false;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_tail_correction(bool on) && {
    set_tail_correction(on);
    return std::move(*this);
}

void PairLJCutCoulLong::finalize_type_table_() {
    if (n_types_ == 0)
        return;
    const double rc = static_cast<double>(pair_cutoff());
    const std::size_t n2 =
        static_cast<std::size_t>(n_types_) * static_cast<std::size_t>(n_types_);
    sig6_tab_.assign(n2, 0.0);
    shift_tab_.assign(n2, 0.0);
    lj1_tab_.assign(n2, 0.0);
    lj2_tab_.assign(n2, 0.0);
    lj3_tab_.assign(n2, 0.0);
    lj4_tab_.assign(n2, 0.0);
    for (int a = 0; a < n_types_; ++a) {
        for (int b = 0; b < n_types_; ++b) {
            const std::size_t ab = static_cast<std::size_t>(a) * n_types_
                                   + static_cast<std::size_t>(b);
            double e = eps_tab_[ab], s = sigma_tab_[ab];
            if (!set_tab_[ab]) {
                const std::size_t aa = static_cast<std::size_t>(a) * n_types_
                                       + static_cast<std::size_t>(a);
                const std::size_t bb = static_cast<std::size_t>(b) * n_types_
                                       + static_cast<std::size_t>(b);
                if (set_tab_[aa] && set_tab_[bb]) {
                    lorentz_berthelot(eps_tab_[aa], sigma_tab_[aa],
                                      eps_tab_[bb], sigma_tab_[bb], e, s);
                    eps_tab_[ab] = e;
                    sigma_tab_[ab] = s;
                } else {
                    e = 0.0;
                    s = 1.0;
                    eps_tab_[ab] = 0.0;
                    sigma_tab_[ab] = s;
                }
            }
            const double s2 = s * s;
            const double s6 = s2 * s2 * s2;
            const double s12 = s6 * s6;
            sig6_tab_[ab] = s6;
            // LAMMPS pair_lj_cut: lj1=48εσ¹², lj2=24εσ⁶, lj3=4εσ¹², lj4=4εσ⁶
            lj1_tab_[ab] = 48.0 * e * s12;
            lj2_tab_[ab] = 24.0 * e * s6;
            lj3_tab_[ab] = 4.0 * e * s12;
            lj4_tab_[ab] = 4.0 * e * s6;
            if (energy_shift_ && e > 0.0 && rc > 0.0) {
                const double rc2 = rc * rc;
                const double r6 = rc2 * rc2 * rc2;
                const double r6inv = 1.0 / r6;
                shift_tab_[ab] =
                    lj3_tab_[ab] * r6inv * r6inv - lj4_tab_[ab] * r6inv;
            } else {
                shift_tab_[ab] = 0.0;
            }
        }
    }
}

void PairLJCutCoulLong::build(const BuildContext& ctx) {
    n_threads_ = 1;
#ifdef _OPENMP
    n_threads_ = omp_get_max_threads();
#endif
    finalize_type_table_();
    erfc_.build(alpha_, static_cast<double>(pair_cutoff()));

    const int n = ctx.frame.capacity();
    if (n_threads_ > 1) {
        fx_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
        fy_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
        fz_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
    } else {
        fx_thr_.clear();
        fy_thr_.clear();
        fz_thr_.clear();
    }

    special_lj_.s12 = lj12_;
    special_lj_.s13 = lj13_;
    special_lj_.s14 = lj14_;
    special_coul_.s12 = c12_;
    special_coul_.s13 = c13_;
    special_coul_.s14 = c14_;
    special_lj_.clear();
    special_coul_.clear();
    scale_version_ = 0;
    if (!any_special_)
        return;
    const auto* dd = ctx.runtime.md.dd;
    const int n_owners =
        (dd && ctx.runtime.mpi_size > 1) ? dd->n_local : ctx.frame.nlocal();
    special_lj_.rebuild(ctx.frame, n_owners);
    special_coul_.rebuild(ctx.frame, n_owners);
}

void PairLJCutCoulLong::rebuild_filtered_list(const Frame& f,
                                              const md::PairList& pairs) {
    // Fill scale arrays parallel to pairs.neighbors[] — no compact neighbor
    // copy. Hot path skips pairs with both scales zero (Amber 1–2 / 1–3).
    //
    // Always rebuild special tables here: atom spatial sort (and migration)
    // reorders local rows while n_owners stays fixed, so a size-only cache
    // would keep stale owner→partner maps and corrupt exclusions.
    const int n = pairs.inum;
    special_lj_.s12 = lj12_;
    special_lj_.s13 = lj13_;
    special_lj_.s14 = lj14_;
    special_coul_.s12 = c12_;
    special_coul_.s13 = c13_;
    special_coul_.s14 = c14_;
    special_lj_.rebuild(f, n);
    special_coul_.rebuild(f, n);

    const int np = pairs.total_pairs;
    scale_lj_.assign(static_cast<std::size_t>(np), 1.0f);
    scale_coul_.assign(static_cast<std::size_t>(np), 1.0f);

    const int* __restrict numneigh = pairs.numneigh.data();
    const int* __restrict firstneigh = pairs.firstneigh.data();
    const int* __restrict neighbors = pairs.neighbors.data();
    const TagInt* __restrict tag = f.span(kAtom_tag);
    float* __restrict sl_out = scale_lj_.data();
    float* __restrict sc_out = scale_coul_.data();

    for (int i = 0; i < n; ++i) {
        const int off = firstneigh[i];
        const int jn = numneigh[i];
        const bool no_lj =
            special_lj_.by_owner[static_cast<std::size_t>(i)].empty();
        const bool no_c =
            special_coul_.by_owner[static_cast<std::size_t>(i)].empty();
        if (no_lj && no_c)
            continue; // already filled with 1.0f
        for (int jj = 0; jj < jn; ++jj) {
            const int j = neighbors[off + jj];
            const int idx = off + jj;
            if (!no_lj)
                sl_out[idx] =
                    static_cast<float>(special_lj_.scale_owner(i, tag[j]));
            if (!no_c)
                sc_out[idx] =
                    static_cast<float>(special_coul_.scale_owner(i, tag[j]));
        }
    }
}

void PairLJCutCoulLong::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairLJCutCoulLong: runtime.md.pair_list is null");
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

    if (n_threads_ > 1
        && static_cast<int>(fx_thr_.empty() ? 0 : fx_thr_[0].size())
               < n_force) {
        for (int t = 0; t < n_threads_; ++t) {
            fx_thr_[t].resize(n_force, 0.0);
            fy_thr_[t].resize(n_force, 0.0);
            fz_thr_[t].resize(n_force, 0.0);
        }
    }

    if (any_special_ && pairs.version != scale_version_) {
        rebuild_filtered_list(f, pairs);
        scale_version_ = pairs.version;
    }

    const int k_inum = pairs.inum;
    const int* __restrict k_ilist = pairs.ilist.data();
    const int* __restrict k_numneigh = pairs.numneigh.data();
    const int* __restrict k_firstneigh = pairs.firstneigh.data();
    const int* __restrict k_neighbors = pairs.neighbors.data();
    const float* __restrict k_sl = any_special_ ? scale_lj_.data() : nullptr;
    const float* __restrict k_sc = any_special_ ? scale_coul_.data() : nullptr;
    const bool k_has_shifts = pairs.has_shifts;
    const double* __restrict k_sx =
        k_has_shifts ? pairs.shift_x.data() : nullptr;
    const double* __restrict k_sy =
        k_has_shifts ? pairs.shift_y.data() : nullptr;
    const double* __restrict k_sz =
        k_has_shifts ? pairs.shift_z.data() : nullptr;

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    const StorageReal* __restrict q_ptr = f.span(kAtom_charge);
    const int* __restrict type_ptr = f.span(kAtom_type);
    AccReal* __restrict fx_ptr = f.span(kAtom_fx);
    AccReal* __restrict fy_ptr = f.span(kAtom_fy);
    AccReal* __restrict fz_ptr = f.span(kAtom_fz);

    const double rc_sq = cutoff_sq_;
    const double qqr2e = ATV::units::coulomb;
    const int ntypes = n_types_;
    const double* __restrict lj1 = lj1_tab_.data();
    const double* __restrict lj2 = lj2_tab_.data();
    const double* __restrict lj3 = lj3_tab_.data();
    const double* __restrict lj4 = lj4_tab_.data();
    const double* __restrict shift = shift_tab_.data();
    const double* __restrict e_tab = erfc_.e_tab.data();
    const double* __restrict f_tab = erfc_.f_tab.data();
    const double inv_dr2 = erfc_.inv_dr2;
    const int nbin = erfc_.nbin;
    const bool do_virial = compute_stress_;

    double energy_total = 0;
    double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;

// if(n_threads_>1): pure serial path skips OpenMP fork (OMP=1 fair vs LAMMPS).
#pragma omp parallel if (n_threads_ > 1) reduction(+ : energy_total)           \
    reduction(+ : v_xx) reduction(+ : v_yy) reduction(+ : v_zz)                \
    reduction(+ : v_xy) reduction(+ : v_xz) reduction(+ : v_yz)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const bool direct = (n_threads_ == 1);
        double* __restrict fxt;
        double* __restrict fyt;
        double* __restrict fzt;
        if (direct) {
            fxt = fx_ptr;
            fyt = fy_ptr;
            fzt = fz_ptr;
        } else {
            std::fill(fx_thr_[tid].begin(), fx_thr_[tid].begin() + n_force,
                      0.0);
            std::fill(fy_thr_[tid].begin(), fy_thr_[tid].begin() + n_force,
                      0.0);
            std::fill(fz_thr_[tid].begin(), fz_thr_[tid].begin() + n_force,
                      0.0);
            fxt = fx_thr_[tid].data();
            fyt = fy_thr_[tid].data();
            fzt = fz_thr_[tid].data();
        }

        const double lx = box.length(0), ly = box.length(1), lz = box.length(2);
        const double inv_lx = 1.0 / lx, inv_ly = 1.0 / ly, inv_lz = 1.0 / lz;

        // Per-pair min-image is NOT redundant under MPI: VelocityVerlet
        // wraps coordinates into the box every step, so between neighbor
        // rebuilds a wrapped owner (or the ghost copy forwarded from it)
        // sits a full box length from its listed partners. Raw dx would
        // silently drop those pairs until the next rebuild (slow thermal
        // runaway, worse at high rank counts). LAMMPS avoids this by
        // wrapping only at reneighbor; until ATV does the same, keep the
        // wrap here.
        const bool wrap = !k_has_shifts;

#pragma omp for schedule(static)
        for (int ii = 0; ii < k_inum; ++ii) {
            const int i = k_ilist[ii];
            const double xi = x_ptr[i], yi = y_ptr[i], zi = z_ptr[i];
            const double qi_scale = qqr2e * q_ptr[i]; // hoist coulomb*qi
            const int ti = type_ptr[i];
            const int jnum = k_numneigh[i];
            const int off = k_firstneigh[i];
            double fi_x = 0, fi_y = 0, fi_z = 0;
            const std::size_t ti_base =
                static_cast<std::size_t>(ti) * static_cast<std::size_t>(ntypes);

            // LAMMPS-shaped tight loop: early distance exit before any
            // table/scale load — ~45% of a 2 A-skin list is out of cutoff.
            for (int jj = 0; jj < jnum; ++jj) {
                const int j = k_neighbors[off + jj];
                double dx = xi - x_ptr[j];
                double dy = yi - y_ptr[j];
                double dz = zi - z_ptr[j];
                if (k_has_shifts) {
                    dx -= k_sx[off + jj];
                    dy -= k_sy[off + jj];
                    dz -= k_sz[off + jj];
                } else if (wrap) {
                    dx -= lx * std::nearbyint(dx * inv_lx);
                    dy -= ly * std::nearbyint(dy * inv_ly);
                    dz -= lz * std::nearbyint(dz * inv_lz);
                }
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc_sq || r2 == 0.0)
                    continue;

                double fpair = 0.0;
                double epair = 0.0;

                const double sl =
                    k_sl ? static_cast<double>(k_sl[off + jj]) : 1.0;
                if (sl != 0.0) {
                    const std::size_t ab =
                        ti_base + static_cast<std::size_t>(type_ptr[j]);
                    const double a1 = lj1[ab];
                    if (a1 != 0.0) {
                        const double r2inv = 1.0 / r2;
                        const double r6inv = r2inv * r2inv * r2inv;
                        fpair +=
                            sl * r2inv * (a1 * r6inv * r6inv - lj2[ab] * r6inv);
                        epair += sl
                                 * (lj3[ab] * r6inv * r6inv - lj4[ab] * r6inv
                                    - shift[ab]);
                    }
                }
                // LAMMPS special convention with a kspace partner: the erfc
                // part is UNscaled; scaled/excluded bonded pairs subtract
                // (1-f)*qq/r — the plain-Coulomb share the reciprocal sum
                // added for them.
                if (qi_scale != 0.0) {
                    const double qj = q_ptr[j];
                    if (qj != 0.0) {
                        double x = r2 * inv_dr2;
                        int bi = static_cast<int>(x);
                        if (bi >= nbin)
                            bi = nbin - 1;
                        const double t = x - static_cast<double>(bi);
                        const std::size_t ii2 = static_cast<std::size_t>(bi);
                        const double et =
                            e_tab[ii2] + t * (e_tab[ii2 + 1] - e_tab[ii2]);
                        const double ft =
                            f_tab[ii2] + t * (f_tab[ii2 + 1] - f_tab[ii2]);
                        const double qq = qi_scale * qj;
                        double ec = qq * et;
                        double fc = qq * ft;
                        const double sc =
                            k_sc ? static_cast<double>(k_sc[off + jj]) : 1.0;
                        if (sc != 1.0) {
                            const double r2i = 1.0 / r2;
                            const double comp =
                                (1.0 - sc) * qq * std::sqrt(r2i);
                            ec -= comp;
                            fc -= comp * r2i;
                        }
                        fpair += fc;
                        epair += ec;
                    }
                }
                if (fpair == 0.0 && epair == 0.0)
                    continue;

                const double fxp = dx * fpair;
                const double fyp = dy * fpair;
                const double fzp = dz * fpair;
                fi_x += fxp;
                fi_y += fyp;
                fi_z += fzp;
                fxt[j] -= fxp;
                fyt[j] -= fyp;
                fzt[j] -= fzp;
                energy_total += epair;
                if (do_virial) {
                    v_xx += dx * fxp;
                    v_yy += dy * fyp;
                    v_zz += dz * fzp;
                    v_xy += dx * fyp;
                    v_xz += dx * fzp;
                    v_yz += dy * fzp;
                }
            }
            fxt[i] += fi_x;
            fyt[i] += fi_y;
            fzt[i] += fi_z;
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
    if (do_virial) {
        f.meta.get(kStress)[0] += v_xx;
        f.meta.get(kStress)[1] += v_yy;
        f.meta.get(kStress)[2] += v_zz;
        f.meta.get(kStress)[3] += v_xy;
        f.meta.get(kStress)[4] += v_xz;
        f.meta.get(kStress)[5] += v_yz;
    }

    // ── LJ tail correction (LAMMPS pair_modify tail yes) ─────────────────
    // E_tail = (2π/V)·Σ_i Σ_j N_i N_j ∫_rc^∞ u_ij r² dr = etail_pref_/V,
    // P_tail = ptail_pref_/V² — a global constant over the fixed type
    // counts, so it lands on rank 0 only (the energy/virial reduction sums
    // ranks). Volume is re-read every step (barostats move the cell).
    if (tail_) {
        if (!tail_ready_) {
            std::vector<double> cnt(static_cast<std::size_t>(ntypes), 0.0);
#ifdef ATV_HAS_MPI
            const int n_owned = mpi ? dd->n_local : f.nlocal();
#else
            const int n_owned = f.nlocal();
#endif
            for (int i = 0; i < n_owned; ++i) {
                const int t = type_ptr[i];
                if (t >= 0 && t < ntypes)
                    cnt[static_cast<std::size_t>(t)] += 1.0;
            }
#ifdef ATV_HAS_MPI
            if (mpi)
                MPI_Allreduce(MPI_IN_PLACE, cnt.data(), ntypes, MPI_DOUBLE,
                              MPI_SUM, dd->cart_comm);
#endif
            const double rc = std::sqrt(rc_sq);
            const double rc3 = rc * rc * rc;
            const double rc6 = rc3 * rc3;
            const double rc9 = rc6 * rc3;
            const double c89 = 8.0 * ATV::units::pi / 9.0;
            etail_pref_ = 0.0;
            ptail_pref_ = 0.0;
            for (int a = 0; a < ntypes; ++a) {
                for (int b = 0; b < ntypes; ++b) {
                    const std::size_t ab = static_cast<std::size_t>(a) * ntypes
                                           + static_cast<std::size_t>(b);
                    const double e = eps_tab_[ab];
                    if (!(e > 0.0))
                        continue;
                    const double s6 = sig6_tab_[ab];
                    const double s12 = s6 * s6;
                    const double nn = cnt[static_cast<std::size_t>(a)]
                                      * cnt[static_cast<std::size_t>(b)];
                    etail_pref_ += nn * c89 * e * (s12 - 3.0 * s6 * rc6) / rc9;
                    ptail_pref_ +=
                        nn * c89 * e * (4.0 * s12 - 6.0 * s6 * rc6) / rc9;
                }
            }
            tail_ready_ = true;
        }
#ifdef ATV_HAS_MPI
        const bool first_rank = !mpi || ctx.runtime.mpi_rank == 0;
#else
        const bool first_rank = true;
#endif
        if (first_rank) {
            const double vol = f.box.volume();
            f.meta.get(kEnergy) += etail_pref_ / vol;
            f.meta.get(kEnergyPair) += etail_pref_ / vol;
            if (do_virial) {
                // Each diagonal += ptail_pref_/V so tr(W)/(3V) = P_tail.
                f.meta.get(kStress)[0] += ptail_pref_ / vol;
                f.meta.get(kStress)[1] += ptail_pref_ / vol;
                f.meta.get(kStress)[2] += ptail_pref_ / vol;
            }
        }
    }
}

void PairLJCutCoulLong::destroy(const DestroyContext&) {
    fx_thr_.clear();
    fy_thr_.clear();
    fz_thr_.clear();
    special_lj_.clear();
    special_coul_.clear();
    erfc_.clear();
    scale_lj_.clear();
    scale_coul_.clear();
    scale_version_ = 0;
    any_special_ = false;
}

std::size_t PairLJCutCoulLong::memory_bytes() const {
    std::size_t b = 0;
    for (const auto& v : fx_thr_)
        b += v.capacity() * sizeof(double);
    for (const auto& v : fy_thr_)
        b += v.capacity() * sizeof(double);
    for (const auto& v : fz_thr_)
        b += v.capacity() * sizeof(double);
    b += scale_lj_.capacity() * sizeof(float);
    b += scale_coul_.capacity() * sizeof(float);
    b += erfc_.e_tab.capacity() * sizeof(double);
    b += erfc_.f_tab.capacity() * sizeof(double);
    return b;
}

} // namespace cpu
} // namespace ATV
