/// @file src/cpu/potentials/eam.cpp
/// @brief cpu::EAM + EAMPotential — analytic embedded-atom many-body style.
///
/// Density/embedding communication algorithm ported from LAMMPS
/// `src/MANYBODY/pair_eam.cpp` and `src/OPENMP/pair_eam_omp.cpp`, checkout `a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b`.
/// Copyright (2003) Sandia Corporation. Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software. Distributed under the GNU General
/// Public License (GPL-2.0-only).

#include "atomiverse/cpu/potentials/eam.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/eam_data/eam_tables.h"

namespace ATV {
namespace cpu {

EAMPotential::EAMPotential(molrs::Element element, std::string_view name) {
    const int z = static_cast<int>(element);
    for (unsigned long k = 0; k < eam_data::kSuttonChenCount; ++k) {
        const eam_data::SuttonChenRow& row = eam_data::kSuttonChen[k];
        if (row.z == z && name == row.name) {
            n_ = row.n;
            m_ = row.m;
            eps_ = row.eps;
            c_ = row.c;
            a_ = row.a;
            return;
        }
    }
    // No fallback: an unsupported element or name is a programming error, not a
    // reason to silently substitute something plausible.
    throw std::out_of_range("EAMPotential: no parameterization '"
                            + std::string(name)
                            + "' for Z=" + std::to_string(z));
}

EAM::EAM(EAMPotential potential, double cutoff)
    : pot_(potential), cutoff_(cutoff) {
    if (!std::isfinite(cutoff) || cutoff <= 0.0)
        throw std::invalid_argument("EAM: cutoff must be finite and positive");
    // Sutton-Chen is a power law with no intrinsic range. Shift the pair term so
    // it reaches zero at the cutoff; without this a neighbour crossing the cutoff
    // steps the energy discontinuously and NVE stops conserving.
    phi_shift_ = pot_.phi(cutoff);
}

void EAM::build(const cpu::BuildContext& ctx) {
    const std::size_t n = static_cast<std::size_t>(ctx.frame.nlocal());
    rho_.assign(n, 0.0);
    dF_.assign(n, 0.0);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    forces_.build(threads, ctx.frame.capacity());
    rho_threads_.assign(threads > 1 ? threads : 0, std::vector<double>(n));
}

int EAM::pack_forward_comm(int n, const int* list, double* buf,
                           int /*pbc_flag*/, const int* /*pbc*/) {
    for (int k = 0; k < n; ++k)
        buf[k] = dF_[static_cast<std::size_t>(list[k])];
    return n;
}

void EAM::unpack_forward_comm(int n, int first, const double* buf) {
    std::copy_n(buf, n, dF_.begin() + first);
}

int EAM::pack_reverse_comm(int n, int first, double* buf) {
    std::copy_n(rho_.begin() + first, n, buf);
    return n;
}

void EAM::unpack_reverse_comm(int n, const int* list, const double* buf) {
    for (int k = 0; k < n; ++k)
        rho_[static_cast<std::size_t>(list[k])] += buf[k];
}

void EAM::launch(const cpu::LaunchContext& ctx) {
    cpu::Frame& f = ctx.frame;
    const cpu::md::PairList* pl = ctx.runtime.md.pair_list;
    if (pl == nullptr)
        throw std::runtime_error("EAM: no half neighbor list published");
    const bool distributed =
        cpu::multi_rank_dd(ctx.runtime)
        || (ctx.runtime.md.dd && ctx.runtime.md.comm && f.nghost() > 0);
    if (distributed && ctx.runtime.md.comm == nullptr)
        throw std::runtime_error(
            "EAM: spatial decomposition requires a communicator");
    const int n_active = cpu::force_count(ctx);
    const int n_owned = cpu::owned_count(ctx);
    rho_.resize(static_cast<std::size_t>(n_active));
    dF_.resize(static_cast<std::size_t>(n_active));
    for (auto& density : rho_threads_)
        density.resize(static_cast<std::size_t>(n_active));
    const double rc2 = cutoff_ * cutoff_;

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    forces_.bind(f);
    double e_total = 0.0;
    double v[6] = {0, 0, 0, 0, 0, 0};
#pragma omp parallel num_threads(forces_.n_threads())                          \
    reduction(+ : e_total, v[ : 6])
    {
        int tid = 0, team_size = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        team_size = omp_get_num_threads();
#endif
        const md::ThreadForce force = forces_.begin_thread(tid, n_active);
        double* __restrict density =
            rho_threads_.empty() ? rho_.data() : rho_threads_[tid].data();
        std::fill_n(density, n_active, 0.0);
        double* __restrict fx_ptr = force.fx;
        double* __restrict fy_ptr = force.fy;
        double* __restrict fz_ptr = force.fz;

        // LAMMPS PairEAM::compute, Newton on: each half-list pair contributes
        // density to both endpoints. Ghost contributions then return to owners.
#pragma omp for schedule(static)
        for (int ii = 0; ii < pl->inum; ++ii) {
            const int i = pl->ilist[ii];
            const int off = pl->firstneigh[i];
            const int nn = pl->numneigh[i];
            double rho_i = density[i];
            for (int jj = 0; jj < nn; ++jj) {
                const int j = pl->neighbors[off + jj];
                double dx = x_ptr[i] - x_ptr[j];
                double dy = y_ptr[i] - y_ptr[j];
                double dz = z_ptr[i] - z_ptr[j];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc2)
                    continue;
                const double contribution = pot_.f(std::sqrt(r2));
                rho_i += contribution;
                density[j] += contribution;
            }
            density[i] = rho_i;
        }
        // LAMMPS OPENMP: finish density reduction before rank communication;
        // MPI remains on the initiating thread.
        if (!rho_threads_.empty()) {
#pragma omp for schedule(static)
            for (int i = 0; i < n_active; ++i) {
                double value = 0.0;
                for (int thread = 0; thread < team_size; ++thread)
                    value += rho_threads_[thread][i];
                rho_[i] = value;
            }
        }
#pragma omp master
        {
            if (distributed)
                ctx.runtime.md.comm->reverse_comm(
                    static_cast<ReversePacker&>(*this));
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for (int i = 0; i < n_owned; ++i) {
            const double rho = rho_[static_cast<std::size_t>(i)];
            dF_[static_cast<std::size_t>(i)] = pot_.dF(rho);
            e_total += pot_.F(rho);
        }
#pragma omp master
        {
            if (distributed)
                ctx.runtime.md.comm->forward_comm(
                    static_cast<ForwardPacker&>(*this));
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for (int ii = 0; ii < pl->inum; ++ii) {
            const int i = pl->ilist[ii];
            const int off = pl->firstneigh[i];
            const int nn = pl->numneigh[i];
            for (int jj = 0; jj < nn; ++jj) {
                const int j = pl->neighbors[off + jj];
                double dx = x_ptr[i] - x_ptr[j];
                double dy = y_ptr[i] - y_ptr[j];
                double dz = z_ptr[i] - z_ptr[j];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc2)
                    continue;
                const double r = std::sqrt(r2);
                const double phi = pot_.phi(r);
                e_total += phi - phi_shift_;
                if (!ctx.forces_requested && !ctx.stress_requested)
                    continue;
                const double dE_dr = -(pot_.n() * phi
                                       + (dF_[static_cast<std::size_t>(i)]
                                          + dF_[static_cast<std::size_t>(j)])
                                             * pot_.m() * pot_.f(r))
                                     / r;
                const double fpair = -dE_dr / r;
                if (ctx.forces_requested) {
                    fx_ptr[i] += fpair * dx;
                    fy_ptr[i] += fpair * dy;
                    fz_ptr[i] += fpair * dz;
                    fx_ptr[j] -= fpair * dx;
                    fy_ptr[j] -= fpair * dy;
                    fz_ptr[j] -= fpair * dz;
                }
                if (ctx.stress_requested) {
                    v[0] += dx * fpair * dx;
                    v[1] += dy * fpair * dy;
                    v[2] += dz * fpair * dz;
                    v[3] += dx * fpair * dy;
                    v[4] += dx * fpair * dz;
                    v[5] += dy * fpair * dz;
                }
            }
        }
    } // thread-private Newton-on force accumulation
    if (ctx.forces_requested)
        forces_.reduce_into(n_active);
    f.meta.get(kEnergy) += e_total;
    f.meta.get(kEnergyPair) += e_total;
    if (ctx.stress_requested)
        for (int k = 0; k < 6; ++k)
            f.meta.get(kStress)[k] += v[k];
}

void EAM::destroy(const cpu::DestroyContext&) {
    rho_.clear();
    dF_.clear();
    rho_threads_.clear();
    forces_.clear();
}

} // namespace cpu
} // namespace ATV
