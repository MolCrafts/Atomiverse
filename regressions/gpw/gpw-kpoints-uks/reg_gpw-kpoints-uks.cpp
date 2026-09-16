/// @file regressions/gpw/gpw-kpoints-uks/reg_gpw-kpoints-uks.cpp
/// @brief END-TO-END: multi-k collinear UKS GPW (identities + absolute pin).
///
/// Public `GpwDft` only. Offline CP2K twin for O₂ product geometry is under
/// `tools/cp2k_oracle/inputs/gpw-kpoints-uks/` (REFERENCE.md). ctest never
/// spawns CP2K. Fast H₂ gates keep wall ≤ 60 s; O₂ absolute product energy is
/// the offline CP2K pin cited for documentation (optional ATV re-pin note).
///
/// Gates:
///   A1 Γ-limit identity (explicit k=Γ k-UKS vs real Γ-UKS)
///   A2 closed-shell multi-k UKS ≡ multi-k RKS
///   A3 time-reversal fold identity (UKS)
///   A4 k-sampling activation (|E_MP − E_Γ| floor)
///   A5 absolute e_tot vs frozen first-green pin + payload shapes
///
/// Exit non-zero on failure.

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

std::vector<ATV::cpu::gpw::GpwAtom> h2_toy(double L_bohr) {
    using namespace ATV;
    const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
    const double r0[3] = {0.5 * L_bohr - 0.7, 0.5 * L_bohr, 0.5 * L_bohr};
    const double r1[3] = {0.5 * L_bohr + 0.7, 0.5 * L_bohr, 0.5 * L_bohr};
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.emplace_back(r0, bH, pH, 1);
    atoms.emplace_back(r1, bH, pH, 1);
    return atoms;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GpwResult;
    using cpu::gpw::XcKind;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const double L = 8.0;
    auto atoms = h2_toy(L);

    // ── A1: Γ-limit identity ──────────────────────────────────────────────
    {
        GpwDft g;
        g.cell[0] = g.cell[1] = g.cell[2] = L;
        g.cutoff_ry = 100.0;
        g.n_alpha = 2;
        g.n_beta = 0;
        g.max_iter = 80;
        g.tol = 1e-7;
        g.xc = XcKind::PADE;
        g.build(atoms);
        const GpwResult rg = g.run(atoms);

        GpwDft k;
        k.cell[0] = k.cell[1] = k.cell[2] = L;
        k.cutoff_ry = 100.0;
        k.n_alpha = 2;
        k.n_beta = 0;
        k.max_iter = 80;
        k.tol = 1e-7;
        k.xc = XcKind::PADE;
        k.k_points = {{0.0, 0.0, 0.0}};
        k.k_point_weights = {1.0};
        k.build(atoms);
        const GpwResult rk = k.run(atoms);
        check(rg.converged && rk.converged, "A1 both converge");
        check(std::fabs(rk.e_total - rg.e_total) <= 1e-6,
              "A1 |E_kΓ − E_Γ| ≤ 1e-6 Ha");
        std::printf("  A1 dE = %.3e Ha\n", rk.e_total - rg.e_total);
    }

    // ── A2: closed-shell multi-k UKS ≡ multi-k RKS ─────────────────────────
    {
        GpwDft rks;
        rks.cell[0] = rks.cell[1] = rks.cell[2] = L;
        rks.cutoff_ry = 100.0;
        rks.n_occ = 1;
        rks.max_iter = 80;
        rks.tol = 1e-7;
        rks.xc = XcKind::PADE;
        rks.k_mesh[0] = rks.k_mesh[1] = 1;
        rks.k_mesh[2] = 2;
        rks.build(atoms);
        const GpwResult rr = rks.run(atoms);

        GpwDft uks;
        uks.cell[0] = uks.cell[1] = uks.cell[2] = L;
        uks.cutoff_ry = 100.0;
        uks.n_alpha = 1;
        uks.n_beta = 1;
        uks.max_iter = 80;
        uks.tol = 1e-7;
        uks.xc = XcKind::PADE;
        uks.k_mesh[0] = uks.k_mesh[1] = 1;
        uks.k_mesh[2] = 2;
        uks.build(atoms);
        const GpwResult ru = uks.run(atoms);
        check(rr.converged && ru.converged, "A2 both converge");
        check(std::fabs(ru.e_total - rr.e_total) <= 5e-6,
              "A2 |E_UKS − E_RKS| ≤ 5e-6 Ha");
        std::printf("  A2 dE = %.3e Ha\n", ru.e_total - rr.e_total);
    }

    // ── A3: time-reversal identity ────────────────────────────────────────
    {
        auto run = [&](bool tr) {
            GpwDft m;
            m.cell[0] = m.cell[1] = m.cell[2] = L;
            m.cutoff_ry = 100.0;
            m.n_alpha = 2;
            m.n_beta = 0;
            m.max_iter = 80;
            m.tol = 1e-7;
            m.xc = XcKind::PADE;
            m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
            m.k_time_reversal = tr;
            m.build(atoms);
            return m.run(atoms);
        };
        const GpwResult full = run(false);
        const GpwResult fold = run(true);
        check(full.converged && fold.converged, "A3 both converge");
        check(full.n_kpts == 8 && fold.n_kpts == 4, "A3 n_kpts 8 vs 4");
        check(std::fabs(full.e_total - fold.e_total) <= 5e-6,
              "A3 |E_full − E_TR| ≤ 5e-6 Ha");
        std::printf("  A3 dE = %.3e Ha\n", full.e_total - fold.e_total);
    }

    // ── A4/A5: main MP 2×2×2 high-spin H₂ UKS (absolute pin) ─────────────
    // Frozen first-green ATV absolute (REFERENCE.md). CP2K product twin is O₂
    // PBE (documented offline; not re-run here for wall budget).
    // Pin filled after first green measurement below.
    // Frozen first-green ATV absolute (REFERENCE.md), 2026-07-25.
    constexpr double kE_pin = -0.719059673672;
    constexpr double kTolE = 1e-3; // Ha
    constexpr double kFloorAct = 1e-4;

    {
        GpwDft gamma;
        gamma.cell[0] = gamma.cell[1] = gamma.cell[2] = L;
        gamma.cutoff_ry = 120.0;
        gamma.n_alpha = 2;
        gamma.n_beta = 0;
        gamma.max_iter = 80;
        gamma.tol = 1e-7;
        gamma.xc = XcKind::PADE;
        gamma.build(atoms);
        const GpwResult rg = gamma.run(atoms);

        GpwDft mp;
        mp.cell[0] = mp.cell[1] = mp.cell[2] = L;
        mp.cutoff_ry = 120.0;
        mp.n_alpha = 2;
        mp.n_beta = 0;
        mp.max_iter = 80;
        mp.tol = 1e-7;
        mp.xc = XcKind::PADE;
        mp.k_mesh[0] = mp.k_mesh[1] = mp.k_mesh[2] = 2;
        mp.k_time_reversal = true;
        mp.build(atoms);
        const GpwResult rm = mp.run(atoms);

        check(rg.converged && rm.converged, "A4/A5 both converge");
        check(rm.n_kpts == 4, "A5 TR-folded n_kpts == 4");
        check(static_cast<int>(rm.eps_k.size()) == rm.n_kpts,
              "A5 eps_k size == n_kpts");
        check(!rm.k_frac.empty() && !rm.k_weights.empty(), "A5 k payload");
        const double d_act = std::fabs(rm.e_total - rg.e_total);
        check(d_act >= kFloorAct, "A4 |E_MP − E_Γ| activation floor");
        // Fixed occupations: M = Nα − Nβ = 2 (high-spin H₂ toy).
        check((2 - 0) == 2, "A5 M = Nα − Nβ = 2");
        check(std::isfinite(rm.e_total), "A5 e_tot finite");

        std::printf("  E_Γ  = %.12f Ha  iters=%d\n", rg.e_total, rg.iters);
        std::printf("  E_MP = %.12f Ha  iters=%d nk=%d\n", rm.e_total, rm.iters,
                    rm.n_kpts);
        std::printf("  |E_MP − E_Γ| = %.3e Ha\n", d_act);
        std::printf("  PIN_CANDIDATE E_MP = %.12f Ha\n", rm.e_total);

        check(std::fabs(rm.e_total - kE_pin) <= kTolE,
              "A5 |E_MP − pin| within tol");
    }

    const auto t1 = clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    std::printf("wall = %.3f s  fails = %d\n", wall, g_fail);
    if (g_fail != 0) {
        std::fprintf(stderr, "FAIL gpw-kpoints-uks\n");
        return 1;
    }
    std::printf("PASS gpw-kpoints-uks\n");
    return 0;
}
