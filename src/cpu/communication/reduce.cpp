/// @file cpu/communication/reduce.cpp
/// @brief `Reduce` -- batched end-of-step MPI_Allreduce of the
///        per-rank scalars and stress tensor.
///
/// Bound at `end_of_step @ -50`, after `KineticEnergy @ -100` (so KE is
/// the locally-summed value) and before `Publish @ +10` (so any
/// async observer reading from the snapshot sees the globally-reduced
/// scalars). Single-rank: structural no-op.

#include "atomiverse/cpu/communication/reduce.h"

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace md {
namespace mpi {

#ifdef ATV_HAS_MPI

void Reduce::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_size <= 1)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;

    // Cadence gate (see Reduce doc): with every_ > 1, fire only on step 0, the
    // final step of the run, and every every_-th step. Gated on the
    // all-ranks-identical ctx.step / is_last_step so the collective is still
    // entered in lockstep — no rank-divergent skip, hence no deadlock.
    // every_ == 1 (default) reduces every step, byte-identical to before; the
    // larger cadence is opt-in via MD::set_scalar_reduce_every. Always firing
    // on the last step keeps the globally-reduced scalars valid for the common
    // "read Frame metadata after run()" pattern, so the cadence is safe whenever
    // no *per-step* consumer reads them.
    if (every_ > 1 && !ctx.runtime.md.is_last_step && ctx.step != 0
        && (ctx.step % every_) != 0)
        return;

    auto& f = ctx.frame;
    // Pack {energy, energy_pair, energy_bond, energy_angle, energy_dihedral,
    // energy_improper, energy_charging_work, kinetic, stress[6],
    // stress_kinetic[6]} into one 20-double allreduce. Per-term energy fields
    // reduce identically to `energy` (each is a per-rank partial sum of one
    // evaluator family). `n_local` is omitted -- it is authoritative on each
    // rank from `dd->n_local` and does not need to be summed. Both stress
    // fields reduce identically (each is a per-rank partial sum of Voigt
    // components).
    double sendbuf[20];
    sendbuf[0] = f.meta.get(kEnergy);
    sendbuf[1] = f.meta.get(kEnergyPair);
    sendbuf[2] = f.meta.get(kEnergyBond);
    sendbuf[3] = f.meta.get(kEnergyAngle);
    sendbuf[4] = f.meta.get(kEnergyDihedral);
    sendbuf[5] = f.meta.get(kEnergyImproper);
    sendbuf[6] = f.meta.get(kEnergyChargingWork);
    sendbuf[7] = f.meta.get(kKinetic);
    for (int k = 0; k < 6; ++k)
        sendbuf[8 + k] = f.meta.get(kStress)[k];
    for (int k = 0; k < 6; ++k)
        sendbuf[14 + k] = f.meta.get(kStressKinetic)[k];

    double recvbuf[20];
    MPI_Allreduce(sendbuf, recvbuf, 20, MPI_DOUBLE, MPI_SUM, dd->cart_comm);

    f.meta.get(kEnergy) = recvbuf[0];
    f.meta.get(kEnergyPair) = recvbuf[1];
    f.meta.get(kEnergyBond) = recvbuf[2];
    f.meta.get(kEnergyAngle) = recvbuf[3];
    f.meta.get(kEnergyDihedral) = recvbuf[4];
    f.meta.get(kEnergyImproper) = recvbuf[5];
    f.meta.get(kEnergyChargingWork) = recvbuf[6];
    f.meta.get(kKinetic) = recvbuf[7];
    for (int k = 0; k < 6; ++k)
        f.meta.get(kStress)[k] = recvbuf[8 + k];
    for (int k = 0; k < 6; ++k)
        f.meta.get(kStressKinetic)[k] = recvbuf[14 + k];
}

#else // !ATV_HAS_MPI

void Reduce::launch(const cpu::LaunchContext& /*ctx*/) {}

#endif // ATV_HAS_MPI

} // namespace mpi
} // namespace md
} // namespace cpu
} // namespace ATV
