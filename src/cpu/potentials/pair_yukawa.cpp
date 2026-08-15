/// @file cpu/potentials/pair_yukawa.cpp
#include "atomiverse/cpu/potentials/pair_yukawa.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

PairYukawa::PairYukawa(double A, double kappa, double cutoff)
    : Pair(cutoff), A_(A), kappa_(kappa) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairYukawa: cutoff must be > 0");
    if (!(kappa >= 0.0))
        throw std::invalid_argument("PairYukawa: kappa must be ≥ 0");
    cutoff_sq_ = cutoff * cutoff;
    const double urc = A * std::exp(-kappa * cutoff) / cutoff;
    v_shift_ = urc;
}

void PairYukawa::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = static_cast<int>(f.nlocal());

    double energy = 0.0;
    // Pair virial, Voigt xx yy zz xy xz yz [eV] — Σ Δr ⊗ f, LAMMPS sign. The
    // screened Coulomb is the whole interaction here (hard cutoff, no
    // reciprocal partner), so this is its complete stress contribution.
    md::Voigt6 vir{};
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
            f.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cutoff_sq_ || r2 == 0.0)
                continue;

            const double r = std::sqrt(r2);
            const double e = std::exp(-kappa_ * r);
            const double inv_r = 1.0 / r;
            const double v = A_ * e * inv_r - v_shift_;
            // The shift is a constant, so it drops out of the derivative:
            // U = A e^{−κr}/r ⇒ dU/dr = −A e^{−κr}(κ/r + 1/r²).
            const double dUdr = -A_ * e * (kappa_ * inv_r + inv_r * inv_r);
            const double fpair =
                -dUdr / r; // F on i from j: fpair * r_ij with r_ij = r_j-r_i

            fx[i] -= fpair * dx;
            fy[i] -= fpair * dy;
            fz[i] -= fpair * dz;
            fx[j] += fpair * dx;
            fy[j] += fpair * dy;
            fz[j] += fpair * dz;
            energy += v;
            // F_i = fpair·(r_i − r_j) ⇒ Δ_a f_b = fpair·d_a·d_b. Unconditional
            // (see the reassociation note in pair_lj_cut.cpp).
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
