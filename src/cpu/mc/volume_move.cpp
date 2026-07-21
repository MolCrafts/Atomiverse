/// @file src/cpu/mc/volume_move.cpp
/// @brief cpu::mc::VolumeMove implementation.

#include "atomiverse/cpu/mc/volume_move.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/mc/transaction.h"

namespace ATV {
namespace cpu {
namespace mc {

VolumeMove::VolumeMove(double dv_max) : dv_max_(dv_max) {
    if (!(dv_max > 0.0)) {
        throw std::invalid_argument("VolumeMove: dv_max must be > 0 Å³");
    }
}

void VolumeMove::launch(const cpu::LaunchContext& ctx) {
    if (ctx.step < 0)
        return; // energy-only bootstrap pass

    std::mt19937_64* rng = ctx.runtime.mc.rng;
    mc::Transaction* txn = ctx.runtime.mc.proposal;
    if (rng == nullptr || txn == nullptr) {
        throw std::runtime_error(
            "VolumeMove::launch: ctx.mc.rng / ctx.mc.proposal not published — "
            "VolumeMove must run under an MC preset");
    }

    const double v_old = ctx.frame.simbox.volume();
    std::uniform_real_distribution<double> dv(-dv_max_, dv_max_);
    const double v_new = v_old + dv(*rng);
    if (!(v_new > 0.0)) {
        // A non-positive trial volume has zero Boltzmann weight — record an
        // empty proposal the policy rejects outright.
        txn->begin(Transaction::Verb::None, ctx.frame);
        return;
    }

    const double s = std::cbrt(v_new / v_old);
    txn->begin(Transaction::Verb::Box, ctx.frame);
    txn->record_box(ctx.frame);
    // Hoisted once before the loop -- the 13.7x rule.
    StorageReal* __restrict sp_x = ctx.frame.span(kAtom_x);
    StorageReal* __restrict sp_y = ctx.frame.span(kAtom_y);
    StorageReal* __restrict sp_z = ctx.frame.span(kAtom_z);
    for (int i = 0; i < ctx.frame.nlocal(); ++i) {
        txn->record_touch(ctx.frame, i); // snapshot BEFORE rescaling the row
        sp_x[i] = static_cast<StorageReal>(static_cast<double>(sp_x[i]) * s);
        sp_y[i] = static_cast<StorageReal>(static_cast<double>(sp_y[i]) * s);
        sp_z[i] = static_cast<StorageReal>(static_cast<double>(sp_z[i]) * s);
    }
    ctx.frame.simbox.scale_isotropic(s);
}

} // namespace mc
} // namespace cpu
} // namespace ATV
