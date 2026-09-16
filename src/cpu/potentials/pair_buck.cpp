/// @file cpu/potentials/pair_buck.cpp
#include "atomiverse/cpu/potentials/pair_buck.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

PairBuck::PairBuck(double A, double rho, double C, double cutoff)
    : Pair(cutoff), A_(A), rho_(rho), C_(C) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairBuck: cutoff must be > 0");
    if (!(rho > 0.0))
        throw std::invalid_argument("PairBuck: rho must be > 0");
    if (!(A >= 0.0))
        throw std::invalid_argument("PairBuck: A must be ≥ 0");
    if (!(C >= 0.0))
        throw std::invalid_argument("PairBuck: C must be ≥ 0");
    cutoff_sq_ = cutoff * cutoff;
    // U(rc) = A exp(-rc/rho) - C/rc^6  (shift so U_shifted(rc)=0)
    const double rc6 = cutoff_sq_ * cutoff_sq_ * cutoff_sq_;
    v_shift_ = A * std::exp(-cutoff / rho) - C / rc6;
}

void PairBuck::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = static_cast<int>(f.nlocal());

    double energy = 0.0;
    // Pair virial, Voigt xx yy zz xy xz yz [eV] — Σ Δr ⊗ f, positive under
    // repulsion (LAMMPS sign, matching PairLJCut / PairBorn). Buckingham is an
    // oxide / halide short-range repulsion: without this a constant-pressure
    // run integrates a pressure missing its entire repulsive wall.
    md::Voigt6 vir{};
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    AccReal* __restrict fx = f.span(kAtom_fx);
    AccReal* __restrict fy = f.span(kAtom_fy);
    AccReal* __restrict fz = f.span(kAtom_fz);

    const double inv_rho = 1.0 / rho_;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            f.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cutoff_sq_ || r2 == 0.0)
                continue;

            const double r = std::sqrt(r2);
            const double r6 = r2 * r2 * r2;
            const double expterm = std::exp(-r * inv_rho);
            const double v = A_ * expterm - C_ / r6 - v_shift_;
            // dU/dr (unshifted) = -(A/rho) exp(-r/rho) + 6 C / r^7
            const double dUdr = -A_ * inv_rho * expterm + 6.0 * C_ / (r6 * r);
            // F on i from j: F_i = -∇_i U; r = |r_j - r_i|, ∇_i r = -r̂
            // → F_i = dUdr * r̂ with r̂ = (r_j-r_i)/r → F_i = (dUdr/r) * (r_j-r_i)
            // Convention in this codebase (Yukawa/Morse): fpair = -dUdr/r and
            // F_i -= fpair * (r_j-r_i)  ⇔  F_i = (dUdr/r) * (r_j-r_i).
            const double fpair = -dUdr / r;

            fx[i] -= fpair * dx;
            fy[i] -= fpair * dy;
            fz[i] -= fpair * dz;
            fx[j] += fpair * dx;
            fy[j] += fpair * dy;
            fz[j] += fpair * dz;
            energy += v;
            // Δ = r_i − r_j = −(dx,dy,dz) and F_i = fpair·Δ, so every
            // component is fpair·Δ_a·Δ_b = fpair·d_a·d_b — the sign of the
            // separation the loop happened to measure drops out.
            // Unconditional by design: a loop-invariant `if (do_virial)` here
            // is what let GCC 14.3 reassociate the reduction away in
            // PairLJCut's SIMD path (see pair_lj_cut.cpp).
            vir.v[0] += fpair * dx * dx;
            vir.v[1] += fpair * dy * dy;
            vir.v[2] += fpair * dz * dz;
            vir.v[3] += fpair * dx * dy;
            vir.v[4] += fpair * dx * dz;
            vir.v[5] += fpair * dy * dz;
        }
    }
    f.meta.get(kEnergy) += energy;
    vir.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV
