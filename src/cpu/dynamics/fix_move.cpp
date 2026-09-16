/// @file cpu/dynamics/fix_move.cpp
#include "atomiverse/cpu/dynamics/fix_move.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"

#include <stdexcept>

namespace ATV {
namespace cpu {
namespace md {

void FixMove::build(const BuildContext& ctx) {
    atoms_.clear();
    x0_.clear();
    y0_.clear();
    z0_.clear();
    origin_set_ = false;
    auto type = ctx.frame.span(kAtom_type);
    auto x = ctx.frame.span(kAtom_x);
    auto y = ctx.frame.span(kAtom_y);
    auto z = ctx.frame.span(kAtom_z);
    const int n = ctx.frame.nlocal();
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(type[i]) != type_id)
            continue;
        atoms_.push_back(i);
        x0_.push_back(static_cast<double>(x[i]));
        y0_.push_back(static_cast<double>(y[i]));
        z0_.push_back(static_cast<double>(z[i]));
    }
    if (atoms_.empty())
        throw std::runtime_error("FixMove: no atoms of requested type");
    origin_set_ = true;
    step0_ = 0;
}

void FixMove::launch(const LaunchContext& ctx) {
    if (!origin_set_)
        return;
    auto x = ctx.frame.span(kAtom_x);
    auto y = ctx.frame.span(kAtom_y);
    auto z = ctx.frame.span(kAtom_z);
    auto vx_a = ctx.frame.span(kAtom_vx);
    auto vy_a = ctx.frame.span(kAtom_vy);
    auto vz_a = ctx.frame.span(kAtom_vz);
    auto fx = ctx.frame.span(kAtom_fx);
    auto fy = ctx.frame.span(kAtom_fy);
    auto fz = ctx.frame.span(kAtom_fz);

    // post_force: zero force so integrator does not fight the constraint
    if (ctx.at(ATV::md::Phase::post_force)) {
        for (int i : atoms_) {
            fx[i] = fy[i] = fz[i] = 0;
        }
        return;
    }

    // initial_integrate: impose r = r0 + v * (step - step0) * dt
    // LaunchContext step is the MD step index.
    const int step = ctx.step;
    if (step0_ == 0 && step > 0)
        step0_ = 0; // keep origin at build geometry
    const double t = static_cast<double>(step) * dt;
    for (std::size_t k = 0; k < atoms_.size(); ++k) {
        const int i = atoms_[k];
        x[i] = static_cast<StorageReal>(x0_[k] + vx * t);
        y[i] = static_cast<StorageReal>(y0_[k] + vy * t);
        z[i] = static_cast<StorageReal>(z0_[k] + vz * t);
        vx_a[i] = static_cast<StorageReal>(vx);
        vy_a[i] = static_cast<StorageReal>(vy);
        vz_a[i] = static_cast<StorageReal>(vz);
        last_z_ = static_cast<double>(z[i]);
    }
}

void FixMove::destroy(const DestroyContext&) {
    atoms_.clear();
    origin_set_ = false;
}

} // namespace md
} // namespace cpu
} // namespace ATV
