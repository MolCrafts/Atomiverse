/// @file cpu/dynamics/fix_freeze.cpp
/// @brief Immobile atom subset (electrode / wall freeze).

#include "atomiverse/cpu/dynamics/fix_freeze.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <array>

namespace ATV {
namespace cpu {
namespace md {

FixFreeze::FixFreeze(int type_id) : type_id_(type_id) {}

int FixFreeze::n_constraints(const cpu::Context& /*ctx*/,
                             const cpu::Frame& frame) const {
    const int n = frame.nlocal();
    const auto type = frame.span(kAtom_type);
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(type[i]) == type_id_)
            ++count;
    }
    return 3 * count;
}

void FixFreeze::resolve_rows_(const cpu::Frame& f, int n_owned) {
    rows_.clear();
    const auto type = f.span(kAtom_type);
    rows_.reserve(static_cast<std::size_t>(n_owned));
    for (int i = 0; i < n_owned; ++i) {
        if (static_cast<int>(type[i]) == type_id_)
            rows_.push_back(i);
    }
}

void FixFreeze::snapshot_positions_(const cpu::Frame& f) {
    x0_by_tag_.clear();
    const auto tag = f.span(kAtom_tag);
    const auto x = f.span(kAtom_x);
    const auto y = f.span(kAtom_y);
    const auto z = f.span(kAtom_z);
    for (int i : rows_) {
        x0_by_tag_[tag[i]] = {x[i], y[i], z[i]};
    }
    n_frozen_global_ = static_cast<int>(x0_by_tag_.size());
}

void FixFreeze::zero_force_(cpu::Frame& f) const {
    auto fx = f.span(kAtom_fx);
    auto fy = f.span(kAtom_fy);
    auto fz = f.span(kAtom_fz);
    for (int i : rows_) {
        fx[i] = AccReal(0);
        fy[i] = AccReal(0);
        fz[i] = AccReal(0);
    }
}

void FixFreeze::zero_velocity_(cpu::Frame& f) const {
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    for (int i : rows_) {
        vx[i] = StorageReal(0);
        vy[i] = StorageReal(0);
        vz[i] = StorageReal(0);
    }
}

void FixFreeze::restore_positions_(cpu::Frame& f) const {
    const auto tag = f.span(kAtom_tag);
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    for (int i : rows_) {
        const auto it = x0_by_tag_.find(tag[i]);
        if (it == x0_by_tag_.end())
            continue;
        x[i] = it->second[0];
        y[i] = it->second[1];
        z[i] = it->second[2];
    }
}

void FixFreeze::build(const cpu::BuildContext& ctx) {
    const int n = owned_count(ctx);
    resolve_rows_(ctx.frame, n);
    snapshot_positions_(ctx.frame);
    zero_velocity_(ctx.frame);
    zero_force_(ctx.frame);
}

void FixFreeze::launch(const cpu::LaunchContext& ctx) {
    const int n = owned_count(ctx);
    resolve_rows_(ctx.frame, n);

    if (ctx.at(ATV::md::Phase::post_force)) {
        zero_force_(ctx.frame);
        return;
    }
    if (ctx.at(ATV::md::Phase::post_integrate)) {
        restore_positions_(ctx.frame);
        zero_velocity_(ctx.frame);
        return;
    }
    if (ctx.at(ATV::md::Phase::final_integrate)) {
        zero_velocity_(ctx.frame);
        return;
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
