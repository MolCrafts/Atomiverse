/// @file src/cpu/potentials/coul_long.cpp
/// @brief Real-space screened Coulomb implementation (CPU).
#include "atomiverse/cpu/potentials/coul_long.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

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
    if (md::ChunkForces* arena = ctx.runtime.md.chunk_forces)
        arena->reserve(n_threads_, n);

    special_.s12 = s12_;
    special_.s13 = s13_;
    special_.s14 = s14_;
    any_special_ = special_.active();
    special_.clear();
    special_version_ = 0;
    if (!any_special_)
        return;
    const int n_owners = ctx.frame.nlocal();
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
    const TagInt* __restrict tag_ptr =
        any_special_ ? f.span(kAtom_tag).data() : nullptr;

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    const StorageReal* __restrict q_ptr = f.span(kAtom_charge);

    const double alpha = alpha_;
    const double rc_sq = cutoff_sq_;
    const double qqr2e = ATV::units::md_qqr2e;

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
        double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;
        double* __restrict fxt = tf.fx;
        double* __restrict fyt = tf.fy;
        double* __restrict fzt = tf.fz;

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
                dx = xi - x_ptr[j];
                dy = yi - y_ptr[j];
                dz = zi - z_ptr[j];
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
                if (ctx.forces_requested) {
                    fxt[j] -= fxp;
                    fyt[j] -= fyp;
                    fzt[j] -= fzp;
                }

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
            if (ctx.forces_requested) {
                fxt[i] += fi_x;
                fyt[i] += fi_y;
                fzt[i] += fi_z;
            }
        }
        forces.store(tid, energy_total,
                     md::Voigt6{{v_xx, v_yy, v_zz, v_xy, v_xz, v_yz}});
    }
    const double energy_total = forces.energy();
    const md::Voigt6 vir = forces.virial();

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyPair) += energy_total;
    if (ctx.stress_requested)
        vir.flush_to(f.meta.get(kStress));
}

void CoulLong::destroy(const DestroyContext&) {
    special_.clear();
    any_special_ = false;
    special_version_ = 0;
}

} // namespace cpu
} // namespace ATV
