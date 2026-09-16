/// @file src/cpu/dynamics/fix_efield.cpp
#include "atomiverse/cpu/dynamics/fix_efield.h"

#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/owned_range.h"

namespace ATV {
namespace cpu {
namespace md {

void FixEfield::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // Metal units: F [eV/Å] = q [e] * E [V/Å] with qe2f = 1.
    const double ex = Ex;
    const double ey = Ey;
    const double ez = Ez;

    // Unwrapped energy needs orthorhombic image · L (LAMMPS domain->unmap).
    if (!f.box.is_orthorhombic()) {
        throw std::runtime_error(
            "FixEfield: triclinic boxes are not supported in this foothold "
            "(need orthorhombic image unwrap)");
    }
    const double lx = f.box.length(0);
    const double ly = f.box.length(1);
    const double lz = f.box.length(2);

    const StorageReal* __restrict q = f.span(kAtom_charge);
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
        const double qi = static_cast<double>(q[i]);
        const double fxi = qi * ex;
        const double fyi = qi * ey;
        const double fzi = qi * ez;
        if (want_force) {
            fx[i] += fxi;
            fy[i] += fyi;
            fz[i] += fzi;
        }
        // r^u = r + image * L  (ortho); image column is Fixed3 int8.
        const double xu =
            static_cast<double>(x[i]) + static_cast<double>(image(i, 0)) * lx;
        const double yu =
            static_cast<double>(y[i]) + static_cast<double>(image(i, 1)) * ly;
        const double zu =
            static_cast<double>(z[i]) + static_cast<double>(image(i, 2)) * lz;
        // U += -F · r^u  (LAMMPS fsum[0])
        u -= fxi * xu + fyi * yu + fzi * zu;
    }

    last_u_ = u;
    f.meta.get(kEnergy) += u;
}

} // namespace md
} // namespace cpu
} // namespace ATV
