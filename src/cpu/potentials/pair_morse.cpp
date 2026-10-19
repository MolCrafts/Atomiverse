/// @file cpu/potentials/pair_morse.cpp
/// @brief PairMorse — live "one new pair style = two files" demonstration.

#include "atomiverse/cpu/potentials/pair_morse.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

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
    // Pair virial, Voigt xx yy zz xy xz yz [eV] — Σ Δr ⊗ f, LAMMPS sign.
    md::Voigt6 vir{};
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    AccReal* __restrict fx = f.span(kAtom_fx);
    AccReal* __restrict fy = f.span(kAtom_fy);
    AccReal* __restrict fz = f.span(kAtom_fz);

    const auto* pair_list = ctx.runtime.md.pair_list;
    auto accumulate_pair = [&](int i, int j, int entry = -1) {
        double dx = x[j] - x[i];
        double dy = y[j] - y[i];
        double dz = z[j] - z[i];
        if (pair_list) {
            // The list image is attached to j; this kernel uses r_j-r_i.
            dx = -dx;
            dy = -dy;
            dz = -dz;
            dx = -dx;
            dy = -dy;
            dz = -dz;
        } else
            f.box.min_image(dx, dy, dz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= cutoff_sq_ || r2 == 0.0)
            return;

        const double r = std::sqrt(r2);
        const double xexp = std::exp(-alpha_ * (r - r0_));
        const double one_minus_x = 1.0 - xexp;
        const double v = D_ * one_minus_x * one_minus_x - v_shift_;
        // dV/dr = 2 D α (1 − x) x
        const double dvdr = 2.0 * D_ * alpha_ * one_minus_x * xexp;
        const double fpair = -dvdr / r; // F_ij = fpair · r_ij

        if (ctx.forces_requested) {
            fx[i] -= fpair * dx;
            fy[i] -= fpair * dy;
            fz[i] -= fpair * dz;
            fx[j] += fpair * dx;
            fy[j] += fpair * dy;
            fz[j] += fpair * dz;
        }
        energy += v;
        // F_i = fpair·(r_i − r_j) ⇒ Δ_a f_b = fpair·d_a·d_b. Unconditional
        // (see the reassociation note in pair_lj_cut.cpp).
        vir.v[0] += fpair * dx * dx;
        vir.v[1] += fpair * dy * dy;
        vir.v[2] += fpair * dz * dz;
        vir.v[3] += fpair * dx * dy;
        vir.v[4] += fpair * dx * dz;
        vir.v[5] += fpair * dy * dz;
    };
    if (const auto* pairs = ctx.runtime.md.pair_list) {
        pairs->for_each_pair(accumulate_pair);
    } else {
        if (multi_rank_dd(ctx.runtime))
            throw std::runtime_error("PairMorse: spatial decomposition "
                                     "requires a half neighbor list");
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                accumulate_pair(i, j);
    }
    f.meta.get(kEnergy) += energy;
    if (ctx.stress_requested)
        vir.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV
