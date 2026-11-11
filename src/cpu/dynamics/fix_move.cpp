/// @file cpu/dynamics/fix_move.cpp
#include "atomiverse/cpu/dynamics/fix_move.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"

#include <stdexcept>

namespace ATV {
namespace cpu {
namespace md {

void FixMove::build(const BuildContext& ctx) {
    if (multi_rank_dd(ctx.runtime))
        throw std::runtime_error(
            "FixMove: spatial decomposition is unsupported; "
            "trajectory origins are not migrated");
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(vx)
        || !std::isfinite(vy) || !std::isfinite(vz))
        throw std::invalid_argument(
            "FixMove: velocity and timestep must be finite; dt > 0");
    origins_.clear();
    origin_set_ = false;
    const auto type = ctx.frame.span(kAtom_type);
    const auto tags = ctx.frame.span(kAtom_tag);
    const auto x = ctx.frame.span(kAtom_x);
    const auto y = ctx.frame.span(kAtom_y);
    const auto z = ctx.frame.span(kAtom_z);
    for (int i = 0; i < owned_count(ctx); ++i) {
        if (static_cast<int>(type[i]) != type_id)
            continue;
        if (!origins_.emplace(tags[i], std::array<double, 3>{x[i], y[i], z[i]})
                 .second)
            throw std::invalid_argument(
                "FixMove: selected atoms must have unique global tags");
    }
    if (origins_.empty())
        throw std::runtime_error("FixMove: no atoms of requested type");
    origin_set_ = true;
}

void FixMove::launch(const LaunchContext& ctx) {
    if (!origin_set_)
        return;
    if (multi_rank_dd(ctx.runtime))
        throw std::runtime_error(
            "FixMove: spatial decomposition is unsupported; "
            "trajectory origins are not migrated");
    const int n = owned_count(ctx);
    const auto tags = ctx.frame.span(kAtom_tag);
    const auto type = ctx.frame.span(kAtom_type);
    // Resolve current rows by tag on every call: sorting cannot reassign an
    // existing origin to a different atom. New selected identities need build.
    for (int i = 0; i < n; ++i)
        if (static_cast<int>(type[i]) == type_id && !origins_.contains(tags[i]))
            throw std::runtime_error(
                "FixMove: selected atom has no trajectory origin; rebuild "
                "after changing atom identities");
    auto x = ctx.frame.span(kAtom_x);
    auto y = ctx.frame.span(kAtom_y);
    auto z = ctx.frame.span(kAtom_z);
    auto vx_a = ctx.frame.span(kAtom_vx);
    auto vy_a = ctx.frame.span(kAtom_vy);
    auto vz_a = ctx.frame.span(kAtom_vz);
    auto fx = ctx.frame.span(kAtom_fx);
    auto fy = ctx.frame.span(kAtom_fy);
    auto fz = ctx.frame.span(kAtom_fz);

    // post_force: zero force so integrator does not fight the constraint.
    if (ctx.at(ATV::md::Phase::post_force)) {
        if (ctx.forces_requested)
            for (int i = 0; i < n; ++i)
                if (static_cast<int>(type[i]) == type_id)
                    fx[i] = fy[i] = fz[i] = 0;
        return;
    }
    if (!ctx.at(ATV::md::Phase::initial_integrate))
        return;

    const double t = static_cast<double>(ctx.step) * dt;
    bool first = true;
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(type[i]) != type_id)
            continue;
        const auto& origin = origins_.at(tags[i]);
        x[i] = static_cast<StorageReal>(origin[0] + vx * t);
        y[i] = static_cast<StorageReal>(origin[1] + vy * t);
        z[i] = static_cast<StorageReal>(origin[2] + vz * t);
        vx_a[i] = static_cast<StorageReal>(vx);
        vy_a[i] = static_cast<StorageReal>(vy);
        vz_a[i] = static_cast<StorageReal>(vz);
        if (first) {
            last_z_ = static_cast<double>(z[i]);
            first = false;
        }
    }
}

void FixMove::destroy(const DestroyContext&) {
    origins_.clear();
    origin_set_ = false;
}

} // namespace md
} // namespace cpu
} // namespace ATV
