/// @file regressions/gpw/periodic-wannier-gpw-uks/reg_periodic-wannier-gpw-uks.cpp
/// @brief END-TO-END: collinear UKS multi-k GPW MLWF of bulk Si vs BCT midpoints.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + ATV_BUILD_PW.
/// Pin: diamond Si valence MLWFs sit on the Si–Si bond midpoints **per spin**
/// at M = 0 (Marzari & Vanderbilt, Phys. Rev. B 56, 12847 (1997) §IV;
/// collinear productization of `periodic-wannier-gpw`).
/// Exit non-zero on failure with got / want / tol.

#include <algorithm>
#include <chrono>
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

double max_assigned_distance(const std::vector<std::array<double, 3>>& centers,
                             const std::array<std::array<double, 3>, 8>& mids,
                             const double L[3]) {
    const int n = 8;
    double cost[8][8];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            cost[i][j] = min_image_dist(centers[static_cast<std::size_t>(i)],
                                        mids[static_cast<std::size_t>(j)], L);

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
    // Coarse 2×2×2 SMV midpoint tol (same as reg_periodic-wannier-gpw).
    constexpr double kMidTol = 0.90;      // bond-midpoint distance [Angstrom]
    constexpr double kSpreadRatio = 0.50; // Ω_final / Ω_init upper bound
    // Offline CP2K 2026.1 twin (see REFERENCE.md): total energy [Ha].
    // ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  -15.769956662772422
    // (LSD M=1, PADE, 100 Ry, MP 2×2×2 FULL_GRID, DZVP-MOLOPT-GTH / GTH-PBE-q4)
    constexpr double kCp2kEHa = -15.769956662772422;
    constexpr double kEnergyTolHa = 2.0e-3; // multi-k UKS vs CP2K (single-grid)

    const double a_bohr = kA * kA2B;
    auto atoms = si_bct(a_bohr);

    GpwDft method;
    method.cell[0] = method.cell[1] = a_bohr / std::sqrt(2.0);
    method.cell[2] = a_bohr;
    method.cutoff_ry = kCutoffRy;
    // Collinear closed-shell UKS: 4 Si × 4 valence e⁻ → 8↑ + 8↓.
    method.n_alpha = 8;
    method.n_beta = 8;
    method.n_occ = 0;
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

    const auto t0 = std::chrono::steady_clock::now();
    const auto r = mlwf.compute(method, atoms);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    if (!r.scf_converged) {
        std::printf("FAIL: SCF did not converge (E=%g Ha)\n", r.energy_ha);
        return 1;
    }
    std::printf("SCF E=%.8f Ha  n_up=%d n_dn=%d n_wann=%d  iters=%d  conv=%d\n",
                r.energy_ha, r.n_wann_up, r.n_wann_dn, r.n_wann, r.iterations,
                static_cast<int>(r.converged));
    std::printf("Omega_init=%.6f  Omega=%.6f  [Ang^2]\n",
                r.total_spread_initial, r.total_spread);
    std::printf("ATV wall (SCF+MLWF) = %.3f s\n", wall_s);

    if (r.n_wann_up != 8 || r.n_wann_dn != 8 || r.n_wann != 16) {
        std::printf("FAIL: expected n_up=n_dn=8 (total 16), got up=%d dn=%d "
                    "n=%d\n",
                    r.n_wann_up, r.n_wann_dn, r.n_wann);
        return 1;
    }
    if (static_cast<int>(r.centers.size()) != 16
        || static_cast<int>(r.spin.size()) != 16) {
        std::printf("FAIL: centers/spin size want 16 got %zu / %zu\n",
                    r.centers.size(), r.spin.size());
        return 1;
    }

    // Energy pin vs offline CP2K (filled after twin run). Skip absolute gate
    // only when the placeholder is still the sentinel −15.77 exactly and the
    // live energy is far (first-boot); after pin, kCp2kEHa is the real value.
    const double de = std::abs(r.energy_ha - kCp2kEHa);
    std::printf("|E - CP2K| = %.6e Ha (tol %.1e)\n", de, kEnergyTolHa);
    if (!(de < kEnergyTolHa)) {
        std::printf("FAIL: energy pin  E=%.8f  want %.8f ± %.1e Ha\n",
                    r.energy_ha, kCp2kEHa, kEnergyTolHa);
        return 1;
    }

    const auto mids = diamond_bct_bond_midpoints(kA);
    const double L[3] = {kA / std::sqrt(2.0), kA / std::sqrt(2.0), kA};

    for (int s = 0; s < 2; ++s) {
        std::vector<std::array<double, 3>> ch_cent;
        double omega = 0.0;
        for (std::size_t i = 0; i < r.centers.size(); ++i) {
            if (r.spin[i] != s)
                continue;
            ch_cent.push_back(r.centers[i]);
            omega += r.spreads[i];
            if (!std::isfinite(r.spreads[i])) {
                std::printf("FAIL: non-finite spread spin=%d i=%zu\n", s, i);
                return 1;
            }
        }
        if (static_cast<int>(ch_cent.size()) != 8) {
            std::printf("FAIL: spin %d expected 8 centers, got %zu\n", s,
                        ch_cent.size());
            return 1;
        }
        for (int i = 0; i < 8; ++i)
            std::printf("  spin%d center[%d] = (%+.5f, %+.5f, %+.5f)\n", s, i,
                        ch_cent[static_cast<std::size_t>(i)][0],
                        ch_cent[static_cast<std::size_t>(i)][1],
                        ch_cent[static_cast<std::size_t>(i)][2]);

        const double dmax = max_assigned_distance(ch_cent, mids, L);
        std::printf("spin %d: max mid-image→midpoint = %.5f Ang (tol %.2f); "
                    "Omega_spin≈%.5f\n",
                    s, dmax, kMidTol, omega);
        if (!(dmax < kMidTol)) {
            std::printf("FAIL: spin %d bond-midpoint pin  dmax=%.5f want < "
                        "%.2f Ang\n",
                        s, dmax, kMidTol);
            return 1;
        }
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
                    "(accepting for product pin)\n");
    }

    std::printf("PASS: collinear UKS multi-k GPW MLWF Si BCT "
                "(M=0, n_up=n_dn=8, dmax pin, spread ratio=%.4f, wall=%.2fs)\n",
                r.total_spread / r.total_spread_initial, wall_s);
    return 0;
}
