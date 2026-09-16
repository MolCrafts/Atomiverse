/// @file src/cpu/potentials/pair_lj_cut_tip4p_cut.cpp
#include "atomiverse/cpu/potentials/pair_lj_cut_tip4p_cut.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/cpu/communication/domain.h"

namespace ATV {
namespace cpu {

PairLJCutTip4pCut::PairLJCutTip4pCut(int type_O, int type_H, double qdist,
                                     double r_oh, double theta_hoh,
                                     double epsilon_OO, double sigma_OO,
                                     double cut_lj, double cut_coul)
    : PairPotential(static_cast<AccReal>(std::max(
          cut_lj, (cut_coul > 0.0 ? cut_coul : cut_lj) + 2.0 * qdist))),
      type_O_(type_O), type_H_(type_H), qdist_(qdist), epsilon_(epsilon_OO),
      sigma_(sigma_OO), cut_lj_(cut_lj) {
    if (cut_lj <= 0.0)
        throw std::invalid_argument("PairLJCutTip4pCut: cut_lj must be > 0");
    if (qdist < 0.0)
        throw std::invalid_argument("PairLJCutTip4pCut: qdist must be ≥ 0");
    if (r_oh <= 0.0)
        throw std::invalid_argument("PairLJCutTip4pCut: r_oh must be > 0");
    if (sigma_OO <= 0.0)
        throw std::invalid_argument("PairLJCutTip4pCut: sigma_OO must be > 0");
    if (epsilon_OO < 0.0)
        throw std::invalid_argument(
            "PairLJCutTip4pCut: epsilon_OO must be ≥ 0");
    if (!(theta_hoh > 0.0) || !(theta_hoh < ATV::units::pi))
        throw std::invalid_argument(
            "PairLJCutTip4pCut: theta_hoh must be in (0, π)");

    cut_coul_ = (cut_coul > 0.0) ? cut_coul : cut_lj;
    if (cut_coul_ <= 0.0)
        throw std::invalid_argument("PairLJCutTip4pCut: cut_coul must be > 0");

    const double half = 0.5 * theta_hoh;
    const double c = std::cos(half);
    if (!(c > 0.0))
        throw std::invalid_argument(
            "PairLJCutTip4pCut: cos(θ/2) must be > 0 to form α");
    alpha_ = qdist_ / (c * r_oh);

    cut_lj_sq_ = cut_lj_ * cut_lj_;
    cut_coul_sq_ = cut_coul_ * cut_coul_;
    const double cut_plus = cut_coul_ + 2.0 * qdist_;
    cut_coul_plus_sq_ = cut_plus * cut_plus;
    neigh_cut_ = std::max(cut_lj_, cut_plus);
    // PairPotential::cutoff_ is the NeighborConsumer value.
    cutoff_ = static_cast<AccReal>(neigh_cut_);

    sigma6_ = std::pow(sigma_, 6.0);
    if (cut_lj_ > 0.0) {
        const double r2inv = 1.0 / cut_lj_sq_;
        const double r6inv = r2inv * r2inv * r2inv * sigma6_;
        v_shift_ = 4.0 * epsilon_ * (r6inv * r6inv - r6inv);
    }
}

PairLJCutTip4pCut& PairLJCutTip4pCut::set_special_bonds(double s12, double s13,
                                                        double s14) & {
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    return *this;
}

PairLJCutTip4pCut PairLJCutTip4pCut::set_special_bonds(double s12, double s13,
                                                       double s14) && {
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    return std::move(*this);
}

void PairLJCutTip4pCut::compute_newsite(double xO, double yO, double zO,
                                        double xH1, double yH1, double zH1,
                                        double xH2, double yH2, double zH2,
                                        double& xM, double& yM,
                                        double& zM) const {
    const double dx1 = xH1 - xO, dy1 = yH1 - yO, dz1 = zH1 - zO;
    const double dx2 = xH2 - xO, dy2 = yH2 - yO, dz2 = zH2 - zO;
    xM = xO + alpha_ * 0.5 * (dx1 + dx2);
    yM = yO + alpha_ * 0.5 * (dy1 + dy2);
    zM = zO + alpha_ * 0.5 * (dz1 + dz2);
}

void PairLJCutTip4pCut::build(const BuildContext& ctx) {
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
    if (any_special_) {
        const auto* dd = ctx.runtime.md.dd;
        const int n_owners =
            (dd && ctx.runtime.mpi_size > 1) ? dd->n_local : ctx.frame.nlocal();
        special_.rebuild(ctx.frame, n_owners);
    }
    n_sites_ = 0;
    filtered_version_ = 0;
}

void PairLJCutTip4pCut::ensure_sites_(const Frame& f) {
    const int n = f.n_total();
    const int* type = f.span(kAtom_type).data();
    const TagInt* tag = f.span(kAtom_tag).data();
    const auto& index = f.meta.get(kTagIndex);
    const StorageReal* x = f.span(kAtom_x).data();
    const StorageReal* y = f.span(kAtom_y).data();
    const StorageReal* z = f.span(kAtom_z).data();
    const auto& box = f.box;

    // Resolve H partners only when the atom count changes (topology assumed
    // fixed for the foothold). M-site coordinates are always refreshed.
    if (n_sites_ != n || h1_.empty()) {
        n_sites_ = n;
        h1_.assign(n, -1);
        h2_.assign(n, -1);
        is_O_.assign(n, 0);
        mx_.assign(n, 0.0);
        my_.assign(n, 0.0);
        mz_.assign(n, 0.0);

        for (int i = 0; i < n; ++i) {
            if (type[i] != type_O_)
                continue;
            is_O_[i] = 1;
            const TagInt t = tag[i];
            auto it1 = index.find(t + 1);
            auto it2 = index.find(t + 2);
            if (it1 == index.end() || it2 == index.end())
                throw std::runtime_error(
                    "PairLJCutTip4pCut: TIP4P hydrogen missing for oxygen tag "
                    "(need consecutive tags t, t+1, t+2)");
            const int iH1 = it1->second;
            const int iH2 = it2->second;
            if (type[iH1] != type_H_ || type[iH2] != type_H_)
                throw std::runtime_error("PairLJCutTip4pCut: TIP4P hydrogen "
                                         "has incorrect atom type");
            h1_[i] = iH1;
            h2_[i] = iH2;
        }
    }

    for (int i = 0; i < n; ++i) {
        if (!is_O_[i])
            continue;
        const int iH1 = h1_[i];
        const int iH2 = h2_[i];
        // Closest-image H relative to O (LAMMPS domain->closest_image).
        double dx1 = x[iH1] - x[i];
        double dy1 = y[iH1] - y[i];
        double dz1 = z[iH1] - z[i];
        box.min_image(dx1, dy1, dz1);
        double dx2 = x[iH2] - x[i];
        double dy2 = y[iH2] - y[i];
        double dz2 = z[iH2] - z[i];
        box.min_image(dx2, dy2, dz2);
        mx_[i] = x[i] + alpha_ * 0.5 * (dx1 + dx2);
        my_[i] = y[i] + alpha_ * 0.5 * (dy1 + dy2);
        mz_[i] = z[i] + alpha_ * 0.5 * (dz1 + dz2);
    }
}

void PairLJCutTip4pCut::rebuild_filtered_list_(const Frame& f,
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

    const int* numneigh = pairs.numneigh.data();
    const int* firstneigh = pairs.firstneigh.data();
    const int* neighbors = pairs.neighbors.data();
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
            ++kept;
        }
        filtered_numneigh_[i] = kept;
    }
}

void PairLJCutTip4pCut::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairLJCutTip4pCut::launch: runtime.md.pair_list is null");
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

    ensure_sites_(f);

    const bool use_filtered = any_special_;
    if (use_filtered && pairs.version != filtered_version_) {
        rebuild_filtered_list_(f, pairs);
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

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    const StorageReal* __restrict q_ptr = f.span(kAtom_charge);
    const int* __restrict type_ptr = f.span(kAtom_type);
    AccReal* __restrict fx_ptr = f.span(kAtom_fx);
    AccReal* __restrict fy_ptr = f.span(kAtom_fy);
    AccReal* __restrict fz_ptr = f.span(kAtom_fz);

    const double qqr2e = ATV::units::coulomb;
    const double eps4 = 4.0 * epsilon_;
    const double eps24 = 24.0 * epsilon_;
    const double sig6 = sigma6_;
    const double vshift = v_shift_;
    const double cut_lj_sq = cut_lj_sq_;
    const double cut_coul_sq = cut_coul_sq_;
    const double cut_coul_plus_sq = cut_coul_plus_sq_;
    const int type_O = type_O_;
    const double alpha = alpha_;

    double energy_total = 0;
    // Pair virial, Voigt xx yy zz xy xz yz [eV] — Σ Δr ⊗ f, LAMMPS sign.
    //
    // The Coulomb force is measured between M sites and then spread over O,
    // H1 and H2, so "which separation times which force" is a real question.
    // It has an exact answer: the redistribution weights are the barycentric
    // weights that define M,
    //     x_M = (1−α) x_O + (α/2)(x_H1 + x_H2),
    //     F_O = (1−α) f,  F_H1 = F_H2 = (α/2) f,
    // and they sum to one, so the group's contribution
    //     Σ_k x_k ⊗ F_k = [(1−α)x_O + (α/2)(x_H1+x_H2)] ⊗ f = x_M ⊗ f.
    // Both molecules give (x_M(i) − x_M(j)) ⊗ f, i.e. the M-site separation
    // times the M-site force — not the O–O separation. Using the real-atom
    // separation here would be wrong by α·(the intramolecular geometry), a few
    // per cent of the electrostatic stress, and invisible in the forces.
    md::Voigt6 vir{};

    // Serial path (foothold): TIP4P force partitioning writes H partners of O,
    // so a simple single-thread walk avoids OMP races on H force rows.
    // Multi-thread would need atomics or full per-thread buffers + careful
    // partner coverage; keep serial for correctness of the E9 foothold.
    (void)n_threads_;
    std::fill(fx_thr_[0].begin(), fx_thr_[0].begin() + n_force, 0.0);
    std::fill(fy_thr_[0].begin(), fy_thr_[0].begin() + n_force, 0.0);
    std::fill(fz_thr_[0].begin(), fz_thr_[0].begin() + n_force, 0.0);
    double* __restrict fxt = fx_thr_[0].data();
    double* __restrict fyt = fy_thr_[0].data();
    double* __restrict fzt = fz_thr_[0].data();

    const double lx = box.length(0), ly = box.length(1), lz = box.length(2);
    const double inv_lx = 1.0 / lx, inv_ly = 1.0 / ly, inv_lz = 1.0 / lz;

    for (int ii = 0; ii < k_inum; ++ii) {
        const int i = use_filtered ? ii : ilist[ii];
        const int itype = static_cast<int>(type_ptr[i]);
        const double xi = x_ptr[i], yi = y_ptr[i], zi = z_ptr[i];
        const double qi = q_ptr[i];

        // Charge site for i: M if oxygen, else real position.
        double x1x = xi, x1y = yi, x1z = zi;
        if (itype == type_O) {
            x1x = mx_[i];
            x1y = my_[i];
            x1z = mz_[i];
        }

        const int jnum = numneigh[i];
        const int off = firstneigh[i];
        for (int jj = 0; jj < jnum; ++jj) {
            const int j = neighbors[off + jj];
            const double sc = scales ? scales[off + jj] : 1.0;
            const int jtype = static_cast<int>(type_ptr[j]);

            // True-atom separation for LJ (and Coulomb reach test).
            double dx = xi - x_ptr[j];
            double dy = yi - y_ptr[j];
            double dz = zi - z_ptr[j];
            dx -= lx * std::nearbyint(dx * inv_lx);
            dy -= ly * std::nearbyint(dy * inv_ly);
            dz -= lz * std::nearbyint(dz * inv_lz);
            const double rsq = dx * dx + dy * dy + dz * dz;

            // LJ on true positions (O–O only for classic TIP4P).
            if (rsq < cut_lj_sq && itype == type_O && jtype == type_O
                && epsilon_ > 0.0) {
                const double r2inv = 1.0 / rsq;
                const double r6inv = r2inv * r2inv * r2inv * sig6;
                const double fpair =
                    sc * eps24 * r2inv * (2.0 * r6inv * r6inv - r6inv);
                fxt[i] += fpair * dx;
                fyt[i] += fpair * dy;
                fzt[i] += fpair * dz;
                fxt[j] -= fpair * dx;
                fyt[j] -= fpair * dy;
                fzt[j] -= fpair * dz;
                energy_total += sc * (eps4 * (r6inv * r6inv - r6inv) - vshift);
                // LJ acts on the real O positions: Δ = (dx,dy,dz) is already
                // r_i − r_j and F_i = +fpair·Δ.
                vir.v[0] += fpair * dx * dx;
                vir.v[1] += fpair * dy * dy;
                vir.v[2] += fpair * dz * dz;
                vir.v[3] += fpair * dx * dy;
                vir.v[4] += fpair * dx * dz;
                vir.v[5] += fpair * dy * dz;
            }

            // Coulomb with M-site adjustment when either atom is oxygen.
            if (rsq < cut_coul_plus_sq) {
                double x2x = x_ptr[j], x2y = y_ptr[j], x2z = z_ptr[j];
                if (jtype == type_O) {
                    x2x = mx_[j];
                    x2y = my_[j];
                    x2z = mz_[j];
                }
                double cdx = x1x - x2x;
                double cdy = x1y - x2y;
                double cdz = x1z - x2z;
                cdx -= lx * std::nearbyint(cdx * inv_lx);
                cdy -= ly * std::nearbyint(cdy * inv_ly);
                cdz -= lz * std::nearbyint(cdz * inv_lz);
                const double crsq = cdx * cdx + cdy * cdy + cdz * cdz;
                if (crsq >= cut_coul_sq || crsq == 0.0)
                    continue;
                const double qj = q_ptr[j];
                if (qi == 0.0 && qj == 0.0)
                    continue;

                const double r2inv = 1.0 / crsq;
                const double rinv = std::sqrt(r2inv);
                const double prefactor = sc * qqr2e * qi * qj * rinv;
                const double cforce = prefactor * r2inv; // * |r| scale via cdx
                energy_total += prefactor;
                // Charge-site pair: Δ = cd (site of i minus site of j) and the
                // force on that site is +cforce·cd, so the whole six-atom
                // redistribution below collapses to cforce·cd_a·cd_b (see the
                // barycentric identity at the top of this function).
                vir.v[0] += cforce * cdx * cdx;
                vir.v[1] += cforce * cdy * cdy;
                vir.v[2] += cforce * cdz * cdz;
                vir.v[3] += cforce * cdx * cdy;
                vir.v[4] += cforce * cdx * cdz;
                vir.v[5] += cforce * cdy * cdz;

                // Force on charge site of i, partitioned if oxygen.
                if (itype != type_O) {
                    fxt[i] += cdx * cforce;
                    fyt[i] += cdy * cforce;
                    fzt[i] += cdz * cforce;
                } else {
                    const double fdx = cdx * cforce;
                    const double fdy = cdy * cforce;
                    const double fdz = cdz * cforce;
                    const double fOx = fdx * (1.0 - alpha);
                    const double fOy = fdy * (1.0 - alpha);
                    const double fOz = fdz * (1.0 - alpha);
                    const double fHx = 0.5 * alpha * fdx;
                    const double fHy = 0.5 * alpha * fdy;
                    const double fHz = 0.5 * alpha * fdz;
                    fxt[i] += fOx;
                    fyt[i] += fOy;
                    fzt[i] += fOz;
                    const int iH1 = h1_[i], iH2 = h2_[i];
                    fxt[iH1] += fHx;
                    fyt[iH1] += fHy;
                    fzt[iH1] += fHz;
                    fxt[iH2] += fHx;
                    fyt[iH2] += fHy;
                    fzt[iH2] += fHz;
                }

                // Force on charge site of j (opposite).
                if (jtype != type_O) {
                    fxt[j] -= cdx * cforce;
                    fyt[j] -= cdy * cforce;
                    fzt[j] -= cdz * cforce;
                } else {
                    const double fdx = -cdx * cforce;
                    const double fdy = -cdy * cforce;
                    const double fdz = -cdz * cforce;
                    const double fOx = fdx * (1.0 - alpha);
                    const double fOy = fdy * (1.0 - alpha);
                    const double fOz = fdz * (1.0 - alpha);
                    const double fHx = 0.5 * alpha * fdx;
                    const double fHy = 0.5 * alpha * fdy;
                    const double fHz = 0.5 * alpha * fdz;
                    fxt[j] += fOx;
                    fyt[j] += fOy;
                    fzt[j] += fOz;
                    const int jH1 = h1_[j], jH2 = h2_[j];
                    fxt[jH1] += fHx;
                    fyt[jH1] += fHy;
                    fzt[jH1] += fHz;
                    fxt[jH2] += fHx;
                    fyt[jH2] += fHy;
                    fzt[jH2] += fHz;
                }
            }
        }
    }

    for (int i = 0; i < n_force; ++i) {
        fx_ptr[i] += fxt[i];
        fy_ptr[i] += fyt[i];
        fz_ptr[i] += fzt[i];
    }

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyPair) += energy_total;
    vir.flush_to(f.meta.get(kStress));
}

void PairLJCutTip4pCut::destroy(const DestroyContext&) {
    fx_thr_.clear();
    fy_thr_.clear();
    fz_thr_.clear();
    special_.clear();
    h1_.clear();
    h2_.clear();
    is_O_.clear();
    mx_.clear();
    my_.clear();
    mz_.clear();
    n_sites_ = 0;
    filtered_neighbors_.clear();
    filtered_scale_.clear();
    filtered_numneigh_.clear();
    filtered_firstneigh_.clear();
    filtered_inum_ = 0;
    filtered_version_ = 0;
    any_special_ = false;
}

} // namespace cpu
} // namespace ATV
