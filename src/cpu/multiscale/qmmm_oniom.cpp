/// @file src/cpu/multiscale/qmmm_oniom.cpp
#include "atomiverse/cpu/multiscale/qmmm_oniom.h"

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {

QmmmOniom::QmmmOniom(std::vector<LJTypeCoeff> low_lj, double cutoff,
                     bool with_coulomb)
    : low_(std::move(low_lj), cutoff, with_coulomb) {}

void QmmmOniom::build(const BuildContext& ctx) {
    // High level is vacuum QM on the model (+ links); no embedding.
    sub_.build(ctx.frame, ctx.runtime.mpi_size, /*require_embedding=*/false);
}

void QmmmOniom::launch(const LaunchContext& ctx) {
    // E_high(model): QmSubsystem accumulates e_high into kEnergy / electronic.
    sub_.launch(ctx.frame, ctx.forces_requested, e_high_);

    // E_low(real) − E_low(model), with matching force signs.
    e_low_real_ = low_.launch(ctx.frame, /*rows=*/{}, ctx.forces_requested,
                              /*force_sign=*/+1.0);
    e_low_model_ = low_.launch(ctx.frame, sub_.qm_rows(), ctx.forces_requested,
                               /*force_sign=*/-1.0);

    ctx.frame.meta.get(kEnergy) += e_low_real_ - e_low_model_;
    // Classical low is not electronic — leave kEnergyElectronic = high only.
}

} // namespace cpu
} // namespace ATV
