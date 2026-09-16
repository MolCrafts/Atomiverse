/// @file src/cpu/potentials/pair_gaussian_coul.cpp
#include "atomiverse/cpu/potentials/pair_gaussian_coul.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/gaussian_charge.h"

namespace ATV {
namespace cpu {

PairGaussianCoul::PairGaussianCoul(double cutoff,
                                   std::vector<double> eta_by_type,
                                   bool include_self)
    : Pair(cutoff), eta_by_type_(std::move(eta_by_type)),
      include_self_(include_self) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairGaussianCoul: cutoff must be > 0 [Å]");
    cutoff_sq_ = cutoff * cutoff;
}

void PairGaussianCoul::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = static_cast<int>(f.nlocal());

    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    const StorageReal* __restrict q = f.span(kAtom_charge);
    const LocalInt* __restrict type = f.span(kAtom_type);
    AccReal* __restrict fx = f.span(kAtom_fx);
    AccReal* __restrict fy = f.span(kAtom_fy);
    AccReal* __restrict fz = f.span(kAtom_fz);

    const double K = ATV::units::coulomb;
    double energy = 0.0;
    // Pair virial, Voigt xx yy zz xy xz yz [eV] — Σ Δr ⊗ f, LAMMPS sign.
    //
    // This style is *not* a `LongRangeParticipant`: it is a self-contained
    // real-space Coulomb with a hard cutoff (Reed / MetalWalls form), so no
    // reciprocal solver carries any part of this energy and there is nothing
    // to double count — the cutoff pair sum is the whole stress contribution.
    // The Gaussian self-energy is deliberately absent from it: it depends only
    // on η and q, is independent of every coordinate and of the cell, and so
    // contributes nothing to dE/dV.
    md::Voigt6 vir{};

    auto accumulate = [&](int i, int j, double dx, double dy, double dz) {
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= cutoff_sq_ || r2 == 0.0)
            return;
        const double r = std::sqrt(r2);
        const double qi = static_cast<double>(q[i]);
        const double qj = static_cast<double>(q[j]);
        const double ei = eta_of(static_cast<int>(type[i]));
        const double ej = eta_of(static_cast<int>(type[j]));
        const double G = GaussianChargeKernel::greens(r, ei, ej);
        const double dG = GaussianChargeKernel::dgreens_dr(r, ei, ej);
        const double pref = K * qi * qj;
        energy += pref * G;
        const double fpair = pref * dG / r;
        fx[i] += fpair * dx;
        fy[i] += fpair * dy;
        fz[i] += fpair * dz;
        fx[j] -= fpair * dx;
        fy[j] -= fpair * dy;
        fz[j] -= fpair * dz;
        // Here the force on i is +fpair·(r_j − r_i), i.e. F_i = −fpair·Δ with
        // Δ = r_i − r_j, so the pair term is Δ_a F_b = −fpair·d_a·d_b — the
        // opposite sign to the styles that write `f[i] -= fpair·d`. Both give
        // a positive trace under repulsion; check the kernel, not the habit.
        const double c = -fpair;
        vir.v[0] += c * dx * dx;
        vir.v[1] += c * dy * dy;
        vir.v[2] += c * dz * dz;
        vir.v[3] += c * dx * dy;
        vir.v[4] += c * dx * dz;
        vir.v[5] += c * dy * dz;
    };

    if (include_self_) {
        for (int i = 0; i < n; ++i) {
            const double qi = static_cast<double>(q[i]);
            const double ei = eta_of(static_cast<int>(type[i]));
            if (GaussianChargeKernel::is_gaussian(ei))
                energy += GaussianChargeKernel::self_energy(K, qi, ei);
        }
    }

    if (ctx.runtime.md.pair_list) {
        const auto& pairs = *ctx.runtime.md.pair_list;
        for (int ii = 0; ii < pairs.inum; ++ii) {
            const int i = pairs.ilist[static_cast<std::size_t>(ii)];
            const int jnum = pairs.numneigh[static_cast<std::size_t>(i)];
            const int off = pairs.firstneigh[static_cast<std::size_t>(i)];
            for (int jj = 0; jj < jnum; ++jj) {
                const int j =
                    pairs.neighbors[static_cast<std::size_t>(off + jj)];
                if (j <= i)
                    continue; // half-list safety for full lists
                double dx = static_cast<double>(x[j] - x[i]);
                double dy = static_cast<double>(y[j] - y[i]);
                double dz = static_cast<double>(z[j] - z[i]);
                f.box.min_image(dx, dy, dz);
                accumulate(i, j, dx, dy, dz);
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dx = static_cast<double>(x[j] - x[i]);
                double dy = static_cast<double>(y[j] - y[i]);
                double dz = static_cast<double>(z[j] - z[i]);
                f.box.min_image(dx, dy, dz);
                accumulate(i, j, dx, dy, dz);
            }
        }
    }

    f.meta.get(kEnergy) += energy;
    vir.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV
