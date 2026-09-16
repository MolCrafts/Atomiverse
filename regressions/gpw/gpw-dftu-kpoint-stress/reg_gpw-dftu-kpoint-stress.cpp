/// @file regressions/gpw/gpw-dftu-kpoint-stress/reg_gpw-dftu-kpoint-stress.cpp
/// @brief END-TO-END: multi-k UKS GPW + Dudarev DFT+U analytic cell stress.
///
/// High-spin H₂ (Nα=2, Nβ=0), PADE LSD, DZV-GTH-PADE, MP 1×1×2, PLUS_U on H
/// L=0 U−J=4 eV. Geometry matches `gpw-kpoint-uks-stress` (12 Bohr, bond 1.0).
///
/// Gates:
///   A1 SCF + e_dftu payload
///   A2 dual-gate analytic fixed-{P_k^σ} vs multi-SCF strain-FD (decisive)
///   A3 off-diagonal ~0
///   A4 magnitude floor
///   A5 soft absolute energy pin (ATV first-green)
///   A6 U activation |E(U)−E(0)|
///   A7 Γ-limit: k=Γ complex + U stress ≡ real Γ + U stress
///
/// Offline CP2K twin under tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/
/// (unphysical Mulliken on this toy — see REFERENCE.md). ctest never spawns
/// CP2K. Public GpwDft members only.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

std::vector<ATV::cpu::gpw::GpwAtom> h2_hs(double L) {
    using namespace ATV::cpu::gpw;
    const GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
    // Compressed bond along x (same as gpw-kpoint-uks-stress).
    const double r0[3] = {0.5 * L - 0.5, 0.5 * L, 0.5 * L};
    const double r1[3] = {0.5 * L + 0.5, 0.5 * L, 0.5 * L};
    return {GpwAtom(r0, bH, pH, 1), GpwAtom(r1, bH, pH, 1)};
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const double L = 12.0;
    const auto atoms = h2_hs(L);
    constexpr double kU_ev = 4.0;

    auto make_k = [&]() {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = L;
        m.cutoff_ry = 100.0;
        m.n_alpha = 2;
        m.n_beta = 0;
        m.max_iter = 120;
        m.tol = 1e-7;
        m.mix = 0.35;
        m.xc = XcKind::PADE;
        m.local_pp_on_grid = true;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.k_time_reversal = true;
        m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, kU_ev});
        return m;
    };

    // A1: SCF + DFT+U payload
    GpwDft energy = make_k();
    energy.build(atoms);
    const auto& er = energy.run(atoms);
    check(er.converged, "A1 SCF converged");
    check(std::isfinite(er.e_total) && std::isfinite(er.e_dftu),
          "A1 finite e_total / e_dftu");
    check(std::fabs(er.e_dftu) > 1e-8, "A1 e_dftu non-trivial");
    check(er.n_kpts >= 1 && !er.C_occ_k.empty(), "A1 C_occ_k present");
    check(static_cast<int>(er.C_occ_k_beta.size()) == er.n_kpts,
          "A1 C_occ_k_beta present");
    std::printf("E(k-UKS+U 1x1x2, 100 Ry) = %.12f Ha  e_dftu=%.8f  iters=%d  "
                "nk=%d\n",
                er.e_total, er.e_dftu, er.iters, er.n_kpts);

    // A2/A3: dual-gate analytic vs multi-SCF FD
    const double h = 2e-3;
    const auto san = gpw_stress(atoms, make_k(), h);
    const auto sfd = gpw_stress_fd(atoms, make_k(), h);

    double max_diag_err = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int a = 0; a < 3; ++a) {
        const double an = san[static_cast<std::size_t>(a)];
        const double fd = sfd[static_cast<std::size_t>(a)];
        const double err = std::abs(an - fd);
        max_diag_err = std::max(max_diag_err, err);
        std::printf("  σ_%s  analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[a], an, fd, err);
        check(std::isfinite(an) && std::isfinite(fd), "A2 finite stress");
    }
    for (int k = 3; k < 6; ++k)
        check(std::abs(san[static_cast<std::size_t>(k)]) <= 1e-8,
              "A3 off-diagonal ~ 0");

    // With the variational ½(V̂S+SV̂) +U scatter (2026-07-31) the fixed point
    // is stationary and the dual gate is machine-tight (measured 3e-9
    // Ha/Bohr³; the old bare-V̂ residual was ~9e-4). Gate with margin.
    constexpr double kAnFdTol = 1.0e-4;
    std::printf("max|σ_an-σ_fd|=%.3e Ha/Bohr3 (tol %.1e)\n", max_diag_err,
                kAnFdTol);
    check(max_diag_err <= kAnFdTol, "A2 analytic vs FD max diag");

    // A4: magnitude floor
    const double max_diag =
        std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
    std::printf("max|sigma_diag|=%.3e Ha/Bohr3\n", max_diag);
    check(max_diag > 1e-8, "A4 stress magnitude floor");

    // A5: soft absolute energy pin (ATV first-green; freeze after measured run).
    // Sibling U=0 multi-k UKS: E ≈ −0.528389 Ha (`gpw-kpoint-uks-stress`).
    // re-pinned 2026-07-31: variational +U scatter (pre-fix -0.546250995860)
    constexpr double kEPin = -0.734377348082; // Ha
    constexpr double kETol = 1.0e-3;          // Ha
    const double dE = std::abs(er.e_total - kEPin);
    std::printf("A5: |E - pin|=%.3e Ha (pin %.6f tol %.1e)\n", dE, kEPin,
                kETol);
    check(dE <= kETol, "A5 soft absolute e_tot");
    // A6: U activation
    {
        GpwDft m0 = make_k();
        m0.plus_u.clear();
        m0.build(atoms);
        const auto r0 = m0.run(atoms);
        check(r0.converged, "A6 U=0 converged");
        const double dU = std::fabs(er.e_total - r0.e_total);
        std::printf("A6: |E(U=4)-E(U=0)|=%.6e Ha\n", dU);
        check(dU >= 1e-4, "A6 U activation floor");
    }

    // A7: Γ-limit identity — complex k=Γ + U stress ≡ real Γ + U stress
    {
        auto make_g = [&]() {
            GpwDft m;
            m.cell[0] = m.cell[1] = m.cell[2] = L;
            m.cutoff_ry = 100.0;
            m.n_alpha = 2;
            m.n_beta = 0;
            m.max_iter = 120;
            m.tol = 1e-7;
            m.mix = 0.35;
            m.xc = XcKind::PADE;
            m.local_pp_on_grid = true;
            m.nonperiodic = false;
            m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, kU_ev});
            return m;
        };
        GpwDft kg = make_g();
        kg.k_points = {{0.0, 0.0, 0.0}};
        kg.k_point_weights = {1.0};
        const auto s_kG = gpw_stress(atoms, kg, h);
        const auto s_G = gpw_stress(atoms, make_g(), h);
        double max_id = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double d = std::abs(s_kG[static_cast<std::size_t>(a)]
                                      - s_G[static_cast<std::size_t>(a)]);
            max_id = std::max(max_id, d);
            std::printf("  A7 axis %d  kΓ=%.6e  Γ=%.6e  |d|=%.3e\n", a,
                        s_kG[static_cast<std::size_t>(a)],
                        s_G[static_cast<std::size_t>(a)], d);
        }
        std::printf("A7 max|σ_kΓ − σ_Γ|=%.3e\n", max_id);
        check(max_id <= 5e-4, "A7 Γ-limit k-path ≡ real Γ DFT+U stress");
    }

    const auto t1 = clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    std::printf("wall=%.1f s\n", wall_s);
    std::printf("PIN_CANDIDATE E=%.12f e_dftu=%.8f max_dσ=%.6e\n", er.e_total,
                er.e_dftu, max_diag_err);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf(
        "PASS: gpw-dftu-kpoint-stress  E=%.10f  e_dftu=%.6f  max|dσ|=%.3e  "
        "max|σ|=%.3e\n",
        er.e_total, er.e_dftu, max_diag_err, max_diag);
    return 0;
}
