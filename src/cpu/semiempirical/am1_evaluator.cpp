/// @file src/cpu/semiempirical/am1_evaluator.cpp
/// @brief AM1 evaluator implementation.

#include "atomiverse/cpu/semiempirical/am1_evaluator.h"

#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"

namespace {

/// @brief Snapshot of wrapped positions for in-place restore after molecular unwrap.
///
/// Same contract as the Gaussian SCF evaluator: copy element-wise back into
/// existing columns so other components' span pointers stay valid.
struct WrappedPositionsSnapshot {
    std::vector<ATV::StorageReal> x, y, z;
};

/// @brief Anchor-on-atom-0 minimum-image unwrap for an isolated fragment.
///
/// AM1 is molecular (no PBC lattice sums). MD wrapping can place partners on
/// opposite faces of the box; SCF without unwrap then sees unphysical
/// distances and may fail to converge.
WrappedPositionsSnapshot unwrap_for_am1(ATV::cpu::Frame& frame) {
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

void AM1Evaluator::build(const cpu::BuildContext& ctx) {
    solver_.build(ctx.frame);
}

void AM1Evaluator::launch(const cpu::LaunchContext& ctx) {
    auto snap = unwrap_for_am1(ctx.frame);
    // Writeback lives once on AM1 (QmEngine::evaluate path); MD needs
    // accumulate so the shared helper takes that flag.
    solver_.publish(ctx.frame, compute_force_ && ctx.forces_requested,
                    /*accumulate=*/true);
    restore_wrapped(ctx.frame, std::move(snap));
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
