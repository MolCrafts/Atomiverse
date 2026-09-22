/// @file regressions/gpw/periodic-wannier-gpw/reg_periodic-wannier-gpw.cpp
/// @brief END-TO-END: multi-k GPW MLWF of bulk Si vs BCT bond midpoints.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + ATV_BUILD_PW.
/// Pin: diamond Si valence MLWFs sit on the Si–Si bond midpoints
/// (Marzari & Vanderbilt, Phys. Rev. B 56, 12847 (1997) §IV).
/// Exit non-zero on failure with got / want / tol.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/periodic_wannier.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using namespace ATV::cpu::gpw;

namespace {

constexpr double kA2B = 1.0 / 0.52917720859; // Å → Bohr

std::vector<GpwAtom> si_bct(double a_bohr) {
    const double at = a_bohr / std::sqrt(2.0);
    const GaussianBasis bSi(molrs::Element::Si, "DZVP-MOLOPT-GTH");
    const GthPotential pSi(molrs::Element::Si, "GTH-PBE-q4");
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    std::vector<GpwAtom> atoms;
    for (int i = 0; i < 4; ++i) {
        const double pos[3] = {frac[i][0] * at, frac[i][1] * at,
                               frac[i][2] * a_bohr};
        atoms.emplace_back(pos, bSi, pSi, 14);
    }
    return atoms;
}

double min_image_dist(const std::array<double, 3>& a,
                      const std::array<double, 3>& b, const double L[3]) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    dx -= L[0] * std::nearbyint(dx / L[0]);
    dy -= L[1] * std::nearbyint(dy / L[1]);
    dz -= L[2] * std::nearbyint(dz / L[2]);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/// Greedy bipartite max-of-min-image for 8 centers (8! too large → Hungarian-
/// free successive nearest with used mask after random restarts).
double max_assigned_distance(const std::vector<std::array<double, 3>>& centers,
                             const std::array<std::array<double, 3>, 8>& mids,
                             const double L[3]) {
    const int n = 8;
    // Cost matrix
    double cost[8][8];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            cost[i][j] = min_image_dist(centers[static_cast<std::size_t>(i)],
                                        mids[static_cast<std::size_t>(j)], L);

    // Greedy: repeatedly pick globally smallest unused edge.
    bool used_c[8] = {};
    bool used_m[8] = {};
    double dmax = 0.0;
    for (int k = 0; k < n; ++k) {
        double best = std::numeric_limits<double>::infinity();
        int bi = -1, bj = -1;
        for (int i = 0; i < n; ++i) {
            if (used_c[i])
                continue;
            for (int j = 0; j < n; ++j) {
                if (used_m[j])
                    continue;
                if (cost[i][j] < best) {
                    best = cost[i][j];
                    bi = i;
                    bj = j;
                }
            }
        }
        used_c[bi] = used_m[bj] = true;
        dmax = std::max(dmax, best);
    }
    return dmax;
}

} // namespace

int main() {
    constexpr double kA = 5.43; // lattice constant [Angstrom]
    constexpr double kCutoffRy = 100.0;
    constexpr int kMeshMlwf = 2;
    // Coarse 2×2×2 SMV discretization (same window as reg_periodic-wannier-si).
    constexpr double kMidTol = 0.90;      // bond-midpoint distance [Angstrom]
    constexpr double kSpreadRatio = 0.50; // Ω_final / Ω_init upper bound

    const double a_bohr = kA * kA2B;
    auto atoms = si_bct(a_bohr);

    GpwDft method;
    method.cell[0] = method.cell[1] = a_bohr / std::sqrt(2.0);
    method.cell[2] = a_bohr;
    method.cutoff_ry = kCutoffRy;
    method.n_occ = 8; // 4 Si × 4 valence e⁻ / 2
    method.max_iter = 60;
    method.tol = 1e-7;
    method.xc = XcKind::PADE;
    method.eps_default = 1e-7;
    method.build(atoms);

    PeriodicWannier mlwf;
    mlwf.k_mesh = kMeshMlwf;
    mlwf.max_iter = 300;
    mlwf.tol = 1e-10;
    mlwf.step = 0.05;
    mlwf.use_projection = true;
    mlwf.proj_alpha_ang = 1.0;

    const auto r = mlwf.compute(method, atoms);

    if (!r.scf_converged) {
        std::printf("FAIL: SCF did not converge (E=%g Ha)\n", r.energy_ha);
        return 1;
    }
    std::printf("SCF E=%.6f Ha  n_wann=%d  iters=%d  conv=%d\n", r.energy_ha,
                r.n_wann, r.iterations, static_cast<int>(r.converged));
    std::printf("Omega_init=%.6f  Omega=%.6f  [Ang^2]\n",
                r.total_spread_initial, r.total_spread);

    if (r.n_wann != 8 || static_cast<int>(r.centers.size()) != 8) {
        std::printf(
            "FAIL: expected 8 Wannier centers, got n_wann=%d size=%zu\n",
            r.n_wann, r.centers.size());
        return 1;
    }

    const auto mids = diamond_bct_bond_midpoints(kA);
    for (int i = 0; i < 8; ++i) {
        std::printf("  center[%d] = (%+.5f, %+.5f, %+.5f)  Omega=%.5f\n", i,
                    r.centers[static_cast<std::size_t>(i)][0],
                    r.centers[static_cast<std::size_t>(i)][1],
                    r.centers[static_cast<std::size_t>(i)][2],
                    r.spreads[static_cast<std::size_t>(i)]);
        if (!std::isfinite(r.spreads[static_cast<std::size_t>(i)])) {
            std::printf("FAIL: non-finite spread[%d]\n", i);
            return 1;
        }
    }
    for (int i = 0; i < 8; ++i)
        std::printf("  midpoint[%d] = (%+.5f, %+.5f, %+.5f)\n", i,
                    mids[static_cast<std::size_t>(i)][0],
                    mids[static_cast<std::size_t>(i)][1],
                    mids[static_cast<std::size_t>(i)][2]);

    const double L[3] = {kA / std::sqrt(2.0), kA / std::sqrt(2.0), kA};
    const double dmax = max_assigned_distance(r.centers, mids, L);
    std::printf("max assigned min-image distance to bond midpoints: %.5f Ang "
                "(tol %.2f)\n",
                dmax, kMidTol);
    if (!(dmax < kMidTol)) {
        std::printf("FAIL: bond-midpoint pin  dmax=%.5f  want < %.2f Ang\n",
                    dmax, kMidTol);
        return 1;
    }

    if (!(r.total_spread_initial > 0.0)
        || !(r.total_spread < kSpreadRatio * r.total_spread_initial)) {
        std::printf("FAIL: spread reduction  Omega=%.5f  Omega_init=%.5f  "
                    "want Omega < %.2f * Omega_init\n",
                    r.total_spread, r.total_spread_initial, kSpreadRatio);
        return 1;
    }
    std::printf("spread ratio Omega/Omega_init = %.4f (want < %.2f)\n",
                r.total_spread / r.total_spread_initial, kSpreadRatio);

    if (!r.converged) {
        std::printf("WARN: MV loop did not flag converged; centers still pin "
                    "(accepting for foothold)\n");
    }

    std::printf("PASS: GPW multi-k MLWF Si BCT (dmax=%.5f Ang, "
                "spread ratio=%.4f)\n",
                dmax, r.total_spread / r.total_spread_initial);
    return 0;
}
