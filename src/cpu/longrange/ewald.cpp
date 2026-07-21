/// @file src/cpu/longrange/ewald.cpp
/// @brief Reciprocal-space Ewald summation — Frame/MPI wrapper over EwaldSum.
#include "atomiverse/cpu/longrange/ewald.h"

#include <algorithm>
#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>

#include "atomiverse/cpu/comm/domain.h"
#endif

namespace ATV {
namespace cpu {

KSpaceEwald::KSpaceEwald(double alpha, double accuracy)
    : sum_(alpha, accuracy) {}

double KSpaceEwald::recommend_alpha(double cutoff, double accuracy) {
    return longrange::EwaldSum::recommend_alpha(cutoff, accuracy);
}

void KSpaceEwald::ensure_kvectors(const cpu::Frame& frame, bool initial_build) {
    const auto& box = frame.simbox;
    if (have_kvectors_ && box_identity_ == &box
        && box.revision() == cell_revision_)
        return;

    const bool table_grew = sum_.rebuild(box.inv_cell());
    const std::size_t nk = sum_.size();
    const bool capacity_grows =
        table_grew || sr_.capacity() < nk || si_.capacity() < nk;
    if (sr_.capacity() < nk)
        sr_.reserve(nk);
    if (si_.capacity() < nk)
        si_.reserve(nk);
    if (have_kvectors_ && !initial_build) {
        if (capacity_grows)
            note_capacity_rebuild();
        else
            note_cell_refresh();
    }

    sr_.resize(nk);
    si_.resize(nk);
    std::fill(sr_.begin(), sr_.end(), 0.0);
    std::fill(si_.begin(), si_.end(), 0.0);
    box_identity_ = &box;
    cell_revision_ = box.revision();
    have_kvectors_ = true;
}

void KSpaceEwald::build(const BuildContext& ctx) {
    note_build();
    ensure_kvectors(ctx.frame, true);
}

void KSpaceEwald::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    ensure_kvectors(f, false);
    const int ng = static_cast<int>(sum_.size());

    const auto* dd = ctx.runtime.md.dd;
    const bool mpi = dd && ctx.runtime.mpi_size > 1;
    const int n_local = mpi ? dd->n_local : f.nlocal();
    const bool first_rank = !mpi || ctx.runtime.mpi_rank == 0;

    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    const StorageReal* __restrict q = f.span(kAtom_charge);
    AccReal* __restrict sp_fx = f.span(kAtom_fx);
    AccReal* __restrict sp_fy = f.span(kAtom_fy);
    AccReal* __restrict sp_fz = f.span(kAtom_fz);

    // ── Partial structure factor S(k) over owned atoms ─────────────────
    sum_.structure_factor(x, y, z, q, n_local, sr_.data(), si_.data());

#ifdef ATV_HAS_MPI
    if (mpi) {
        MPI_Allreduce(MPI_IN_PLACE, sr_.data(), ng, MPI_DOUBLE, MPI_SUM,
                      dd->cart_comm);
        MPI_Allreduce(MPI_IN_PLACE, si_.data(), ng, MPI_DOUBLE, MPI_SUM,
                      dd->cart_comm);
    }
#endif

    const double vol = f.simbox.volume();

    // ── Reciprocal forces on owned atoms from global S(k) ──────────────
    sum_.add_forces(x, y, z, q, n_local, vol, sr_.data(), si_.data(), sp_fx,
                    sp_fy, sp_fz);

    // ── Self energy (partitions across ranks: owned atoms only) ────────
    double qsq_local = 0.0, q_local = 0.0;
    for (int i = 0; i < n_local; ++i) {
        qsq_local += q[i] * q[i];
        q_local += q[i];
    }
    const double e_self = sum_.self_energy(qsq_local);
    f.meta.get(kEnergy) += e_self;
    f.meta.get(kEnergyPair) += e_self;

    // ── Reciprocal + net-charge energy/stress: rank 0 only ─────────────
    if (first_rank) {
        double e_recip = 0;
        double stress[6];
        sum_.reciprocal_energy_virial(sr_.data(), si_.data(), vol, e_recip,
                                      stress);

        double q_total = q_local;
#ifdef ATV_HAS_MPI
        if (mpi)
            MPI_Allreduce(MPI_IN_PLACE, &q_total, 1, MPI_DOUBLE, MPI_SUM,
                          dd->cart_comm);
#endif
        const double e_net = sum_.net_charge_energy(q_total, vol);

        f.meta.get(kEnergy) += e_recip + e_net;
        f.meta.get(kEnergyPair) += e_recip + e_net;
        f.meta.get(kStress)[0] += stress[0];
        f.meta.get(kStress)[1] += stress[1];
        f.meta.get(kStress)[2] += stress[2];
        f.meta.get(kStress)[3] += stress[3];
        f.meta.get(kStress)[4] += stress[4];
        f.meta.get(kStress)[5] += stress[5];
    }
#ifdef ATV_HAS_MPI
    else if (mpi) {
        // Non-root ranks must still participate in the net-charge Allreduce
        // to match the collective issued on rank 0.
        double q_total = q_local;
        MPI_Allreduce(MPI_IN_PLACE, &q_total, 1, MPI_DOUBLE, MPI_SUM,
                      dd->cart_comm);
    }
#endif
}

void KSpaceEwald::destroy(const DestroyContext&) {
    sum_.clear();
    std::vector<double>().swap(sr_);
    std::vector<double>().swap(si_);
    box_identity_ = nullptr;
    cell_revision_ = 0;
    have_kvectors_ = false;
}

std::size_t KSpaceEwald::memory_bytes() const {
    return sum_.memory_bytes()
           + (sr_.capacity() + si_.capacity()) * sizeof(double);
}

} // namespace cpu
} // namespace ATV
