/// @file src/cpu/semiempirical/rm1_evaluator.cpp
/// @brief RM1 evaluator implementation (mirrors AM1Evaluator).

#include "atomiverse/cpu/semiempirical/rm1_evaluator.h"

#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"

namespace {

struct WrappedPositionsSnapshot {
    std::vector<ATV::StorageReal> x, y, z;
};

WrappedPositionsSnapshot unwrap_for_rm1(ATV::cpu::Frame& frame) {
    using namespace ATV;
    const int n = static_cast<int>(frame.nlocal());
    const StorageReal* __restrict px = frame.span(kAtom_x);
    const StorageReal* __restrict py = frame.span(kAtom_y);
    const StorageReal* __restrict pz = frame.span(kAtom_z);
    WrappedPositionsSnapshot snap{std::vector<ATV::StorageReal>(px, px + n),
                                  std::vector<ATV::StorageReal>(py, py + n),
                                  std::vector<ATV::StorageReal>(pz, pz + n)};

    if (n <= 1)
        return snap;
    if (frame.box.length(0) <= 0.0 || frame.box.length(1) <= 0.0
        || frame.box.length(2) <= 0.0)
        return snap;

    StorageReal* __restrict sp_x = frame.span(kAtom_x);
    StorageReal* __restrict sp_y = frame.span(kAtom_y);
    StorageReal* __restrict sp_z = frame.span(kAtom_z);
    const double x0 = sp_x[0];
    const double y0 = sp_y[0];
    const double z0 = sp_z[0];
    for (int i = 1; i < n; ++i) {
        double dx = sp_x[i] - x0;
        double dy = sp_y[i] - y0;
        double dz = sp_z[i] - z0;
        frame.box.min_image(dx, dy, dz);
        sp_x[i] = static_cast<StorageReal>(x0 + dx);
        sp_y[i] = static_cast<StorageReal>(y0 + dy);
        sp_z[i] = static_cast<StorageReal>(z0 + dz);
    }
    return snap;
}

void restore_wrapped(ATV::cpu::Frame& frame, WrappedPositionsSnapshot&& snap) {
    using namespace ATV;
    StorageReal* __restrict px = frame.span(kAtom_x);
    StorageReal* __restrict py = frame.span(kAtom_y);
    StorageReal* __restrict pz = frame.span(kAtom_z);
    const int n = static_cast<int>(snap.x.size());
    for (int i = 0; i < n; ++i) {
        px[i] = snap.x[static_cast<std::size_t>(i)];
        py[i] = snap.y[static_cast<std::size_t>(i)];
        pz[i] = snap.z[static_cast<std::size_t>(i)];
    }
}

} // namespace

namespace ATV {
namespace cpu {
namespace semiempirical {

void RM1Evaluator::build(const cpu::BuildContext& ctx) {
    solver_.build(ctx.frame);
}

void RM1Evaluator::launch(const cpu::LaunchContext& ctx) {
    auto snap = unwrap_for_rm1(ctx.frame);

    auto& result = solver_.run(ctx.frame);
    ctx.frame.meta.get(kEnergy) += result.total_energy_ev;
    ctx.frame.meta.get(kEnergyElectronic) += result.total_energy_ev;
    const int n = ctx.frame.nlocal();
    // Mulliken charges are a STATE channel, not an accumulator: this evaluator
    // is their sole producer and each launch republishes the whole set, so
    // last-writer-wins `=` is the contract (`+=` would sum charges across
    // steps). Only kEnergy / kEnergy* / kAtom_f* accumulate.
    StorageReal* __restrict sp_charge = ctx.frame.span(kAtom_charge);
    for (int i = 0; i < n; ++i) {
        sp_charge[i] = static_cast<StorageReal>(
            result.mulliken_charges[static_cast<std::size_t>(i)]);
    }

    if (compute_force_ && ctx.forces_requested) {
        solver_.compute_forces(ctx.frame);
        StorageReal* __restrict fx = ctx.frame.span(kAtom_fx);
        StorageReal* __restrict fy = ctx.frame.span(kAtom_fy);
        StorageReal* __restrict fz = ctx.frame.span(kAtom_fz);
        const auto& forces = solver_.last_result().forces_ev_a;
        for (int i = 0; i < n; ++i) {
            const std::size_t base = static_cast<std::size_t>(3 * i);
            fx[i] += static_cast<StorageReal>(forces[base + 0]);
            fy[i] += static_cast<StorageReal>(forces[base + 1]);
            fz[i] += static_cast<StorageReal>(forces[base + 2]);
        }
    }

    restore_wrapped(ctx.frame, std::move(snap));
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
