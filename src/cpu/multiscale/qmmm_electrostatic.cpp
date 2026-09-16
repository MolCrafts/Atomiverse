/// @file src/cpu/multiscale/qmmm_electrostatic.cpp
#include "atomiverse/cpu/multiscale/qmmm_electrostatic.h"

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {

QmmmElectrostatic::QmmmElectrostatic(std::vector<LJTypeCoeff> cross_lj,
                                     double lj_cutoff, double embed_cutoff)
    : vdw_(std::move(cross_lj), lj_cutoff) {
    sub_.embed_cutoff() = embed_cutoff;
}

void QmmmElectrostatic::build(const BuildContext& ctx) {
    sub_.build(ctx.frame, ctx.runtime.mpi_size);
}

void QmmmElectrostatic::launch(const LaunchContext& ctx) {
    sub_.launch(ctx.frame, ctx.forces_requested, e_qm_);
    e_vdw_ = vdw_.launch(ctx.frame, sub_.qm_rows(), ctx.forces_requested);
    ctx.frame.meta.get(kEnergy) += e_vdw_;
    // VdW is classical cross term — not counted in kEnergyElectronic.
}

} // namespace cpu
} // namespace ATV
