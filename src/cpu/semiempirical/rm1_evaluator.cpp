/// @file src/cpu/semiempirical/rm1_evaluator.cpp
/// @brief RM1 evaluator implementation (mirrors AM1Evaluator).

#include "atomiverse/cpu/semiempirical/rm1_evaluator.h"

#include <utility>
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
        const double original_dx = dx, original_dy = dy, original_dz = dz;
        frame.box.min_image(dx, dy, dz);
        // Preserve the input exactly when no periodic image changed.
        if (dx != original_dx)
            sp_x[i] = static_cast<StorageReal>(x0 + dx);
        if (dy != original_dy)
            sp_y[i] = static_cast<StorageReal>(y0 + dy);
        if (dz != original_dz)
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
    solver_.publish(ctx.frame, compute_force_ && ctx.forces_requested,
                    /*accumulate=*/true);
    restore_wrapped(ctx.frame, std::move(snap));
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
