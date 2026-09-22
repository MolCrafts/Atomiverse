/// @file regressions/gpw/periodic-urpa-si/reg_periodic-urpa-si.cpp
/// @brief END-TO-END: Γ-point unrestricted GPW dRPA (URPA@UKS) on periodic
///        Si₂ with a UKS PBE reference (n_α = n_β = 4). See REFERENCE.md.
///
/// Public API only (`GpwDft::run` + `GpwDft::direct_rpa`). Requires
/// ATV_BUILD_GPW (+ HF + PW). Exit non-zero on mismatch. <= 60 s, single
/// process. No live CP2K — offline-pin contract per tools/cp2k_oracle/README.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/rpa.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV::cpu::gpw;
    constexpr double kA2B = 1.8897259885789;
    constexpr double a_bohr = 5.43 * kA2B;

    // Dual-gate vs closed-shell dRPA first-green on the same deck
    // (UKS nα=nβ → URPA ≡ RKS dRPA when orbitals match; setup-integrity band).
    // Offline CP2K re-pin when available — REFERENCE.md.
    constexpr double kEcorrRef =
        -0.2658539776;                   // Ha (RKS dRPA first-green twin)
    constexpr double kEcorrTol = 2.0e-3; // Ha (UKS numerical drift)
    constexpr double kRiCanonTol = 1.0e-8;
    constexpr double kIdTol = 1.0e-12;

    const double pos0[3] = {0.0, 0.0, 0.0};
    const double pos1[3] = {a_bohr / 4.0, a_bohr / 4.0, a_bohr / 4.0};
    const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
    const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
    std::vector<GpwAtom> atoms;
    atoms.emplace_back(pos0, bas, pot, 14);
    atoms.emplace_back(pos1, bas, pot, 14);

    GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = a_bohr;
    // Same 30 Ry + level shift 0.3 Ha deck as periodic-rpa-si (small-gap
    // artificial sc-Si₂). UKS closed-shell occupations (8 valence e⁻).
    m.cutoff_ry = 30.0;
    m.n_alpha = 4;
    m.n_beta = 4;
    m.max_iter = 80;
    m.tol = 1e-6;
    m.mix = 0.3;
    m.level_shift = 0.3;
    m.xc = XcKind::PBE;
    m.mp2_ri = true; // product RI-GPW
    m.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const GpwResult& scf = m.run(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    if (!scf.converged || !std::isfinite(scf.e_total)) {
        std::fprintf(stderr,
                     "FAIL: periodic Γ UKS-PBE SCF did not converge "
                     "(iters=%d E=%.8f)\n",
                     scf.iters, scf.e_total);
        return 1;
    }
    if (scf.C_beta.size() == 0 || scf.eps_beta.empty()) {
        std::fprintf(stderr, "FAIL: UKS result missing β orbitals\n");
        return 1;
    }

    m.mp2_n_virt = 0;
    m.mp2_ri = true;
    const RpaResult r_ri = m.direct_rpa(atoms);
    const auto t2 = std::chrono::steady_clock::now();

    m.mp2_ri = false;
    const RpaResult r_can = m.direct_rpa(atoms);
    const auto t3 = std::chrono::steady_clock::now();

    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();
    const double wall_ri = std::chrono::duration<double>(t2 - t1).count();
    const double wall_can = std::chrono::duration<double>(t3 - t2).count();

    if (!r_ri.ri) {
        std::fprintf(stderr, "FAIL: RpaResult.ri not set on product path\n");
        return 1;
    }
    if (r_can.ri) {
        std::fprintf(stderr,
                     "FAIL: RpaResult.ri set on Poisson foothold path\n");
        return 1;
    }
    if (!std::isfinite(r_ri.e_corr) || !(r_ri.e_corr < 0.0)) {
        std::fprintf(stderr,
                     "FAIL: E_corr(URPA)=%.10f Ha not finite/negative\n",
                     r_ri.e_corr);
        return 1;
    }
    if (std::abs(r_ri.e_total - (r_ri.e_scf + r_ri.e_corr)) > kIdTol) {
        std::fprintf(stderr,
                     "FAIL: e_total != e_scf + e_corr (%.12f vs %.12f)\n",
                     r_ri.e_total, r_ri.e_scf + r_ri.e_corr);
        return 1;
    }
    if (r_ri.n_occ != 8) {
        std::fprintf(stderr, "FAIL: n_occ=%d want 8 (nα+nβ)\n", r_ri.n_occ);
        return 1;
    }
    // Composite OV: for nα=nβ=4 and equal virt counts, n_Ω = 2·nα·nv_α.
    // With all virtuals, n_mo ≈ 26 → n_va=n_vb=22 → n_Ω = 2·4·22 = 176.
    if (r_ri.omega_ha.empty()) {
        std::fprintf(stderr, "FAIL: empty omega spectrum\n");
        return 1;
    }
    for (double w : r_ri.omega_ha) {
        if (!(w > 0.0) || !std::isfinite(w)) {
            std::fprintf(stderr, "FAIL: non-positive dRPA root Ω=%.6e\n", w);
            return 1;
        }
    }
    const double d_ri_can = std::abs(r_ri.e_corr - r_can.e_corr);
    if (d_ri_can > kRiCanonTol) {
        std::fprintf(stderr,
                     "FAIL: RI vs canonical |dE_corr|=%.3e Ha > %.1e "
                     "(RI=%.12f can=%.12f)\n",
                     d_ri_can, kRiCanonTol, r_ri.e_corr, r_can.e_corr);
        return 1;
    }
    if (std::abs(r_ri.e_corr - kEcorrRef) > kEcorrTol) {
        std::fprintf(
            stderr,
            "FAIL: E_corr got=%.10f want=%.10f +/- %.1e Ha "
            "(RKS dRPA dual-gate / offline pin band; see REFERENCE.md)\n",
            r_ri.e_corr, kEcorrRef, kEcorrTol);
        return 1;
    }

    std::printf(
        "PASS: Γ GPW URPA Si2 (UKS-PBE, DZVP-MOLOPT-GTH, GTH-PBE-q4, 30 Ry, "
        "periodic, RI)\n"
        "  E_SCF   = %.10f Ha (iters=%d, <S2>=%.6f, wall=%.1f s)\n"
        "  E_corr  = %.10f Ha (Tr A=%.6f; n_occ=%d n_virt=%d, "
        "RI wall=%.1f s)\n"
        "  E_total = %.10f Ha   |RI−can|=%.3e Ha (can wall=%.1f s)\n"
        "  Ω_min   = %.6f Ha    Ω_max=%.6f Ha  (n_Ω=%zu)\n"
        "  [URPA@UKS dual-gate vs RKS dRPA first-green; re-pin per "
        "REFERENCE.md]\n",
        r_ri.e_scf, scf.iters, scf.s2, wall_scf, r_ri.e_corr, r_ri.tr_A,
        r_ri.n_occ, r_ri.n_virt, wall_ri, r_ri.e_total, d_ri_can, wall_can,
        r_ri.omega_ha.front(), r_ri.omega_ha.back(), r_ri.omega_ha.size());
    return 0;
}
