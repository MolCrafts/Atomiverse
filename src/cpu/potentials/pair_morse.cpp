/// @file cpu/potentials/pair_morse.cpp
/// @brief PairMorse — live "one new pair style = two files" demonstration.

#include "atomiverse/cpu/potentials/pair_morse.h"
#include "atomiverse/core/mat3.h"

#include <cmath>

namespace ATV {
namespace cpu {

PairMorse::PairMorse(double D, double alpha, double r0, double cutoff)
    : Pair(cutoff), D_(D), alpha_(alpha), r0_(r0) {
    cutoff_sq_ = cutoff * cutoff;
    // Shift the potential so V(cutoff) = 0 — no jump at the cutoff,
    // matching LAMMPS `pair_modify shift yes`.
    const double xc = std::exp(-alpha * (cutoff - r0));
    v_shift_ = D * (1.0 - xc) * (1.0 - xc);
}

void PairMorse::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = static_cast<int>(f.nlocal());

    double energy = 0.0;
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    AccReal* __restrict fx = f.span(kAtom_fx);
    AccReal* __restrict fy = f.span(kAtom_fy);
    AccReal* __restrict fz = f.span(kAtom_fz);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            f.simbox.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cutoff_sq_ || r2 == 0.0)
                continue;

            const double r = std::sqrt(r2);
            const double xexp = std::exp(-alpha_ * (r - r0_));
            const double one_minus_x = 1.0 - xexp;
            const double v = D_ * one_minus_x * one_minus_x - v_shift_;
            // dV/dr = 2 D α (1 − x) x
            const double dvdr = 2.0 * D_ * alpha_ * one_minus_x * xexp;
            const double fpair = -dvdr / r; // F_ij = fpair · r_ij

            fx[i] -= fpair * dx;
            fy[i] -= fpair * dy;
            fz[i] -= fpair * dz;
            fx[j] += fpair * dx;
            fy[j] += fpair * dy;
            fz[j] += fpair * dz;
            energy += v;
        }
    }
    f.meta.get(kEnergy) += energy;
}

} // namespace cpu
} // namespace ATV
