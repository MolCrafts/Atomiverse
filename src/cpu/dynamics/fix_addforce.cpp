/// @file src/cpu/dynamics/fix_addforce.cpp
#include "atomiverse/cpu/dynamics/fix_addforce.h"

#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/owned_range.h"

namespace ATV {
namespace cpu {
namespace md {

void FixAddForce::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    const double fx_add = Fx;
    const double fy_add = Fy;
    const double fz_add = Fz;
    const int tid = type_id;

    // Unwrapped energy needs orthorhombic image · L (LAMMPS domain->unmap).
    if (!f.box.is_orthorhombic()) {
        throw std::runtime_error(
            "FixAddForce: triclinic boxes are not supported in this foothold "
            "(need orthorhombic image unwrap)");
    }
    const double lx = f.box.length(0);
    const double ly = f.box.length(1);
    const double lz = f.box.length(2);

    const auto type = f.span(kAtom_type);
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    const auto image = f.span(kAtom_image);
    AccReal* __restrict fx = f.span(kAtom_fx);
    AccReal* __restrict fy = f.span(kAtom_fy);
    AccReal* __restrict fz = f.span(kAtom_fz);

    double u = 0.0;
    const bool want_force = ctx.forces_requested;

#pragma omp parallel for reduction(+ : u) schedule(static)
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(type[i]) != tid)
            continue;

        if (want_force) {
            fx[i] += fx_add;
            fy[i] += fy_add;
            fz[i] += fz_add;
        }
        // r^u = r + image * L  (ortho); image column is Fixed3 int8.
        const double xu =
            static_cast<double>(x[i]) + static_cast<double>(image(i, 0)) * lx;
        const double yu =
            static_cast<double>(y[i]) + static_cast<double>(image(i, 1)) * ly;
        const double zu =
            static_cast<double>(z[i]) + static_cast<double>(image(i, 2)) * lz;
        // U += -F · r^u  (LAMMPS foriginal[0] constant path)
        u -= fx_add * xu + fy_add * yu + fz_add * zu;
    }

    last_u_ = u;
    f.meta.get(kEnergy) += u;
}

} // namespace md
} // namespace cpu
} // namespace ATV
