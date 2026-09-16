/// @file src/cpu/observe/classical_polarization.cpp
/// @brief Classical polarization density from Frame charges.

#include "atomiverse/cpu/observe/classical_polarization.h"

#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/owned_range.h"

namespace ATV {
namespace cpu {
namespace md {

void ClassicalPolarization::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!f.box.is_orthorhombic()) {
        throw std::runtime_error(
            "ClassicalPolarization: triclinic boxes are not supported in this "
            "foothold (need orthorhombic image unwrap)");
    }
    const int n = owned_count(ctx);
    const double lx = f.box.length(0);
    const double ly = f.box.length(1);
    const double lz = f.box.length(2);
    const double V = f.box.volume();
    if (!(V > 0.0)) {
        throw std::runtime_error(
            "ClassicalPolarization: box volume must be positive");
    }

    const StorageReal* __restrict q = f.span(kAtom_charge);
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    const auto image = f.span(kAtom_image);

    double mx = 0.0, my = 0.0, mz = 0.0;
#pragma omp parallel for reduction(+ : mx, my, mz) schedule(static)
    for (int i = 0; i < n; ++i) {
        const double qi = static_cast<double>(q[i]);
        const double xu =
            static_cast<double>(x[i]) + static_cast<double>(image(i, 0)) * lx;
        const double yu =
            static_cast<double>(y[i]) + static_cast<double>(image(i, 1)) * ly;
        const double zu =
            static_cast<double>(z[i]) + static_cast<double>(image(i, 2)) * lz;
        mx += qi * xu;
        my += qi * yu;
        mz += qi * zu;
    }

    mu_ = {mx, my, mz};
    P_ = {mx / V, my / V, mz / V};
}

} // namespace md
} // namespace cpu
} // namespace ATV
