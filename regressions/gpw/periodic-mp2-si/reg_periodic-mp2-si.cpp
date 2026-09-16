/// @file regressions/gpw/periodic-mp2-si/reg_periodic-mp2-si.cpp
/// @brief END-TO-END: Γ-point canonical GPW MP2 on periodic Si₂
///        (DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE reference). See REFERENCE.md.
///
/// Public API only (`GpwDft::run` + `GpwDft::mp2`). Requires ATV_BUILD_GPW
/// (+ HF + PW). Exit non-zero on mismatch. <= 60 s, single process. No live
/// CP2K — offline-pin contract per tools/cp2k_oracle/README.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/mp2.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV::cpu::gpw;
    constexpr double kA2B = 1.8897259885789;
    constexpr double a_bohr = 5.43 * kA2B;

    // ── Hard gates ───────────────────────────────────────────────────────
    // First-green frozen band on the correlation energy [Ha] (setup
    // integrity; replace with the offline CP2K twin re-pin — REFERENCE.md).
    constexpr double kEcorrRef = -0.2764036524; // Ha (first green)
    constexpr double kEcorrTol = 1.0e-3;        // Ha
    constexpr double kIdTol = 1.0e-12;          // exact identities

    // Si diamond fragment: 2 atoms in a cubic a = 5.43 Å cell, periodic Γ.
    const double pos0[3] = {0.0, 0.0, 0.0};
    const double pos1[3] = {a_bohr / 4.0, a_bohr / 4.0, a_bohr / 4.0};
    const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
    const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
    std::vector<GpwAtom> atoms;
    atoms.emplace_back(pos0, bas, pot, 14);
    atoms.emplace_back(pos1, bas, pot, 14);

    GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = a_bohr;
    // 30 Ry + level shift 0.3 Ha: this artificial Γ-only sc-Si₂ crystal is a
    // small-gap system (~0.015 Ha KS gap) — the shift is a convergence aid
    // only (converged eps are reported unshifted); 30 Ry keeps the e2e case
    // inside the ≤ 60 s budget. The CP2K twin pins the same deck
    // (REFERENCE.md).
    m.cutoff_ry = 30.0;
    m.n_occ = 4; // 2 Si × zion 4 → 8 valence e⁻
    m.max_iter = 80;
    m.tol = 1e-6;
    m.mix = 0.3;
    m.level_shift = 0.3;
    m.xc = XcKind::PBE;
    m.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const GpwResult& scf = m.run(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    if (!scf.converged || !std::isfinite(scf.e_total)) {
        std::fprintf(stderr,
                     "FAIL: periodic Γ GPW PBE SCF did not converge "
                     "(iters=%d E=%.8f)\n",
                     scf.iters, scf.e_total);
        return 1;
    }

    // Truncated virtual space first (monotonicity witness), then full MP2 —
    // both on the same converged reference.
    m.mp2_n_virt = 8;
    const double e_corr_trunc = m.mp2(atoms).e_corr;
    m.mp2_n_virt = 0;
    const Mp2Result r = m.mp2(atoms); // value copy: next mp2() would overwrite
    const auto t2 = std::chrono::steady_clock::now();
    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();
    const double wall_mp2 = std::chrono::duration<double>(t2 - t1).count();

    if (!std::isfinite(r.e_corr) || !(r.e_corr < 0.0)) {
        std::fprintf(stderr, "FAIL: E_corr(MP2)=%.10f Ha not finite/negative\n",
                     r.e_corr);
        return 1;
    }
    // Every virtual added can only lower the MP2 sum (each term <= 0).
    if (!(e_corr_trunc < 0.0) || !(r.e_corr <= e_corr_trunc)) {
        std::fprintf(stderr,
                     "FAIL: virtual-space monotonicity violated: "
                     "E_corr(all=%d)=%.10f Ha  E_corr(nv=8)=%.10f Ha\n",
                     r.n_virt, r.e_corr, e_corr_trunc);
        return 1;
    }
    if (std::abs(r.e_corr_os + r.e_corr_ss - r.e_corr) > kIdTol) {
        std::fprintf(stderr,
                     "FAIL: OS+SS decomposition broken: os=%.12f ss=%.12f "
                     "corr=%.12f Ha\n",
                     r.e_corr_os, r.e_corr_ss, r.e_corr);
        return 1;
    }
    if (std::abs(r.e_total - (r.e_scf + r.e_corr)) > kIdTol) {
        std::fprintf(stderr,
                     "FAIL: e_total != e_scf + e_corr (%.12f vs %.12f)\n",
                     r.e_total, r.e_scf + r.e_corr);
        return 1;
    }
    if (std::abs(r.e_corr - kEcorrRef) > kEcorrTol) {
        std::fprintf(stderr,
                     "FAIL: E_corr got=%.10f want=%.10f +/- %.1e Ha "
                     "(offline pin band; see REFERENCE.md)\n",
                     r.e_corr, kEcorrRef, kEcorrTol);
        return 1;
    }

    std::printf(
        "PASS: Γ GPW MP2 Si2 (DZVP-MOLOPT-GTH, GTH-PBE-q4, PBE ref, 30 Ry, "
        "periodic)\n"
        "  E_SCF   = %.10f Ha (iters=%d, wall=%.1f s)\n"
        "  E_corr  = %.10f Ha (OS=%.10f SS=%.10f; n_occ=%d n_virt=%d, "
        "wall=%.1f s)\n"
        "  E_total = %.10f Ha   E_corr(nv=8)=%.10f Ha (monotone)\n"
        "  [internal gates + first-green band; re-pin per REFERENCE.md]\n",
        r.e_scf, scf.iters, wall_scf, r.e_corr, r.e_corr_os, r.e_corr_ss,
        r.n_occ, r.n_virt, wall_mp2, r.e_total, e_corr_trunc);
    return 0;
}
