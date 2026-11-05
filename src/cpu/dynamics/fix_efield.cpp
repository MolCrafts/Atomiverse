/// @file src/cpu/dynamics/fix_efield.cpp
/// @brief Constant-field path of LAMMPS `src/fix_efield.cpp`
///        (`FixEfield::post_force`, charge term), checkout SHA e8c4c278a8.
///        LAMMPS: Copyright (2003) Sandia Corporation, GNU GPL.
#include "atomiverse/cpu/dynamics/fix_efield.h"

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

    // Unwrapped coordinates r^u = r + H n with H's columns the lattice
    // vectors (LAMMPS Domain::unmap, orthorhombic and triclinic alike).
    const auto& h = f.box.cell();
    const double h00 = h(0, 0), h01 = h(0, 1), h02 = h(0, 2);
    const double h10 = h(1, 0), h11 = h(1, 1), h12 = h(1, 2);
    const double h20 = h(2, 0), h21 = h(2, 1), h22 = h(2, 2);

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
        // r^u = r + H n; the image column is Fixed3 int8.
        const double n0 = static_cast<double>(image(i, 0));
        const double n1 = static_cast<double>(image(i, 1));
        const double n2 = static_cast<double>(image(i, 2));
        const double xu =
            static_cast<double>(x[i]) + h00 * n0 + h01 * n1 + h02 * n2;
        const double yu =
            static_cast<double>(y[i]) + h10 * n0 + h11 * n1 + h12 * n2;
        const double zu =
            static_cast<double>(z[i]) + h20 * n0 + h21 * n1 + h22 * n2;
        // U += -F · r^u  (LAMMPS fsum[0])
        u -= fxi * xu + fyi * yu + fzi * zu;
    }

    last_u_ = u;
    f.meta.get(kEnergy) += u;
}

} // namespace md
} // namespace cpu
} // namespace ATV
