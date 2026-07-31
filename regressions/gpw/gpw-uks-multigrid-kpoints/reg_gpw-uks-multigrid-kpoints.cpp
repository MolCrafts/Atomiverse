/// @file regressions/gpw/gpw-uks-multigrid-kpoints/reg_gpw-uks-multigrid-kpoints.cpp
/// @brief Multi-k collinear UKS GPW with CP2K-style REL_CUTOFF multigrid
///        product collocation (NGRIDS 4) — internal exactness gates plus the
///        offline CP2K 2026.1 pin from REFERENCE.md.
///
/// Out-of-library consumer: public GPW headers only, flat main(), hard-coded
/// numbers, no external process. Gates (spec gpw-uks-multigrid-kpoints):
///   M1 Γ-limit: explicit k=(0,0,0) multigrid k-UKS vs Γ-UKS multigrid
///   M2 time-reversal: full 2×2×2 (8 k) vs TR-folded (4 k) multigrid UKS
///   M3 multigrid vs single-grid at main cutoff
///   M4 convergence + finiteness + dual-spin k payload shapes
///   M5 absolute e_tot vs offline CP2K NGRIDS 4 + KPOINTS pin
///
/// System: closed-shell H₂ on the UKS path (Nα=Nβ=1), DZVP-MOLOPT-GTH /
/// GTH-PBE-q1 / PBE — dual-spin product stacks, CP2K-convergent twin.

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
    const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-GTH");
    const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    const double r0[3] = {0.5 * L_bohr - 0.7, 0.5 * L_bohr, 0.5 * L_bohr};
    const double r1[3] = {0.5 * L_bohr + 0.7, 0.5 * L_bohr, 0.5 * L_bohr};
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.emplace_back(r0, bH, pH, 1);
    atoms.emplace_back(r1, bH, pH, 1);
    return atoms;
}

// Closed-shell UKS (dual-spin stacks with n_alpha = n_beta = 1).
ATV::cpu::gpw::GpwDft h2_method(double L, double cutoff_ry, int ngrids,
                                double rel_cutoff) {
    ATV::cpu::gpw::GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = L;
    m.cutoff_ry = cutoff_ry;
    m.ngrids = ngrids;
    m.rel_cutoff = rel_cutoff;
    m.n_alpha = 1;
    m.n_beta = 1;
    m.max_iter = 80;
    m.tol = 1e-8;
    m.xc = ATV::cpu::gpw::XcKind::PBE;
    m.eps_default = 1e-8;
    return m;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GpwDft;
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();
    auto mark = [&](const char* what) {
        std::printf("  [wall] %s at %.1f s\n", what,
                    std::chrono::duration<double>(clock::now() - t_start)
                        .count());
    };

    const double L = 8.0;
    const auto atoms = h2_toy(L);
    // Main pin ladder matches the CP2K twin (200 Ry / REL 40 / NGRIDS 4).
    // Identity gates run the same machinery on a cheaper 80 Ry / REL 20 ladder
    // so the ≤ 60 s budget goes to the absolute pin (each 200 Ry mg SCF is
    // ~18 s; full 8k TR twin at 200 Ry alone is ~36 s).
    constexpr double kCut = 200.0;
    constexpr int kNgrids = 4;
    constexpr double kRel = 40.0;
    constexpr double kCutId = 80.0;
    constexpr double kRelId = 20.0;

    // ── M1: Γ-limit identity on the cheap ladder ───────────────────────────
    {
        GpwDft g = h2_method(L, kCutId, kNgrids, kRelId);
        g.build(atoms);
        const auto& rg = g.run(atoms);
        check(rg.converged, "M1 Γ multigrid UKS converged");

        GpwDft k = h2_method(L, kCutId, kNgrids, kRelId);
        k.k_points = {{0.0, 0.0, 0.0}};
        k.k_point_weights = {1.0};
        k.build(atoms);
        const auto& rk = k.run(atoms);
        check(rk.converged, "M1 k=Γ multigrid UKS converged");
        const double dE = std::fabs(rk.e_total - rg.e_total);
        std::printf("M1: E_Γ=%.12f  E_kΓ=%.12f  |Δ|=%.3e Ha (80 Ry)\n",
                    rg.e_total, rk.e_total, dE);
        check(dE <= 1e-6, "M1 |E_kΓ − E_Γ| ≤ 1e-6 Ha");
        mark("M1 done");
    }

    // ── M2: time-reversal identity on the cheap ladder ─────────────────────
    {
        auto run = [&](bool tr) {
            GpwDft m = h2_method(L, kCutId, kNgrids, kRelId);
            m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
            m.k_time_reversal = tr;
            m.build(atoms);
            return m.run(atoms);
        };
        const auto full = run(false);
        const auto fold = run(true);
        check(full.converged && fold.converged, "M2 both TR legs converged");
        check(full.n_kpts == 8 && fold.n_kpts == 4, "M2 n_kpts 8 vs 4");
        const double dE = std::fabs(full.e_total - fold.e_total);
        std::printf("M2: E_full=%.12f  E_TR=%.12f  |Δ|=%.3e Ha (80 Ry)\n",
                    full.e_total, fold.e_total, dE);
        check(dE <= 5e-6, "M2 |E_full − E_TR| ≤ 5e-6 Ha");
        mark("M2 done");
    }

    // ── M3–M5: main MP 2×2×2 TR multigrid (one SCF) + single-grid twin ────
    {
        GpwDft mp = h2_method(L, kCut, kNgrids, kRel);
        mp.k_mesh[0] = mp.k_mesh[1] = mp.k_mesh[2] = 2;
        mp.k_time_reversal = true;
        mp.build(atoms);
        const auto t0 = clock::now();
        const auto& rm = mp.run(atoms);
        const double wall =
            std::chrono::duration<double>(clock::now() - t0).count();
        std::printf("MAIN: wall %.1f s (REFERENCE perf twin: CP2K NGRIDS 4)\n",
                    wall);
        check(rm.converged, "M4 MP multigrid UKS converged");
        check(std::isfinite(rm.e_total), "M4 finite e_total");
        check(rm.k_frac.size() == 4 && rm.k_weights.size() == 4
                  && rm.eps_k.size() == 4 && rm.eps_k_beta.size() == 4,
              "M4 dual-spin k payload shapes (4 TR k-points)");
        std::printf("MAIN: E(MP 2x2x2 TR, 200 Ry, NGRIDS 4) = %.12f Ha "
                    "(iters=%d)\n",
                    rm.e_total, rm.iters);

        GpwDft sg = h2_method(L, kCut, 1, 0.0);
        sg.k_mesh[0] = sg.k_mesh[1] = sg.k_mesh[2] = 2;
        sg.k_time_reversal = true;
        sg.build(atoms);
        const auto& rs = sg.run(atoms);
        check(rs.converged, "M3 single-grid SCF converged");
        const double dgrid = std::fabs(rm.e_total - rs.e_total);
        std::printf("M3: E(mg)=%.12f  E(sg)=%.12f  |Δ|=%.3e Ha\n", rm.e_total,
                    rs.e_total, dgrid);
        check(dgrid <= 1e-3, "M3 multigrid vs single-grid |Δ| ≤ 1e-3 Ha");

        // M5: offline CP2K 2026.1 pin (REFERENCE.md).
        constexpr double kEPin = -1.166228717524314; // Ha (CP2K 2026.1)
        constexpr double kTolAbs = 1e-3;              // Ha
        const double dpin = std::fabs(rm.e_total - kEPin);
        std::printf("M5: |E − pin| = %.3e Ha (pin %.12f)\n", dpin, kEPin);
        check(dpin <= kTolAbs, "M5 absolute e_tot within 1e-3 Ha of CP2K pin");
        mark("M3-M5 done");
    }

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-uks-multigrid-kpoints — multi-k collinear UKS with "
                "REL_CUTOFF multigrid product collocation (Γ-limit, TR, "
                "grid residual, CP2K pin)\n");
    return 0;
}
