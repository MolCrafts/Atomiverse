/// @file cpu/potentials/pair_wca.cpp
#include "atomiverse/cpu/potentials/pair_wca.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

PairWCA::PairWCA(double epsilon, double sigma)
    : Pair(std::pow(2.0, 1.0 / 6.0) * sigma), epsilon_(epsilon), sigma_(sigma) {
    if (!(epsilon > 0.0) || !(sigma > 0.0))
        throw std::invalid_argument("PairWCA: epsilon and sigma must be > 0");
    const double rc = pair_cutoff();
    cutoff_sq_ = rc * rc;
    // At rc = 2^{1/6} σ: (σ/rc)^6 = 1/2, (σ/rc)^12 = 1/4 → 4ε(1/4−1/2)= −ε
    v_shift_ = -epsilon_;
}

void PairWCA::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = static_cast<int>(f.nlocal());
    const double sig2 = sigma_ * sigma_;

    double energy = 0.0;
    // Pair virial, Voigt xx yy zz xy xz yz [eV] — Σ Δr ⊗ f, LAMMPS sign. WCA
    // is purely repulsive, so the trace it contributes is strictly positive.
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

            const double r2inv = 1.0 / r2;
            const double s2 = sig2 * r2inv;
            const double s6 = s2 * s2 * s2;
            const double s12 = s6 * s6;
            const double v = 4.0 * epsilon_ * (s12 - s6) - v_shift_;
            // F magnitude: (24ε/r)(2 s12 − s6); fpair for F_i = −fpair * r_ij
            const double fpair = 24.0 * epsilon_ * r2inv * (2.0 * s12 - s6);

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
