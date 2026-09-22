/// @file regressions/gpw/gpw-vdw-uks-kpoints/reg_gpw-vdw-uks-kpoints.cpp
/// @brief Multi-k collinear UKS GPW PBE + nonlocal vdW-DF (DRSLL).
///
/// Out-of-library consumer. Self-consistent V_nl of the total density rides
/// both spin potentials on the multi-k UKS runner (`build_uks_k_energy` +
/// DIIS twin); post-SCF E_c^nl via `fold_dispersion_k_uks`. Gates
/// (REFERENCE.md):
///   G1 Γ-limit: explicit k=(0,0,0) multi-k UKS vs shipped Γ UKS;
///   G2 CP2K: MP 2×1×1 multigrid UKS vs offline CP2K 2026.1 LSD + DRSLL;
///   G3 closed-shell multi-k UKS ≡ multi-k RKS (both vdW).
/// No live oracle in ctest.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859;

// Same water dimer as gpw-vdw-df-ice / gpw-vdw-kpoints [Å].
const double kAng[6][3] = {
    {0.000000, 0.000000, 0.000000},  // O
    {0.758602, 0.000000, 0.504284},  // H
    {0.758602, 0.000000, -0.504284}, // H
    {3.000000, 0.000000, 0.000000},  // O
    {3.758602, 0.000000, 0.504284},  // H
    {3.758602, 0.000000, -0.504284}, // H
};
const int kZ[6] = {8, 1, 1, 8, 1, 1};

std::vector<ATV::cpu::gpw::GpwAtom> make_atoms() {
    using namespace ATV::cpu::gpw;
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    std::vector<GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double pos[3] = {kAng[i][0] * kA2B, kAng[i][1] * kA2B,
                               kAng[i][2] * kA2B};
        if (kZ[i] == 8)
            atoms.emplace_back(pos, bO, pO, 8);
        else
            atoms.emplace_back(pos, bH, pH, 1);
    }
    return atoms;
}

// Closed-shell UKS (nα = nβ = 8) water dimer base; n_occ unused when UKS.
ATV::cpu::gpw::GpwDft make_uks_base(int ngrids, double rel_cutoff) {
    ATV::cpu::gpw::GpwDft gpw;
    const double L = 12.0 * kA2B;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.ngrids = ngrids;
    gpw.rel_cutoff = rel_cutoff;
    gpw.n_occ = 0;
    gpw.n_alpha = 8;
    gpw.n_beta = 8;
    gpw.max_iter = 60;
    gpw.tol = 1e-7;
    gpw.xc = ATV::cpu::gpw::XcKind::PBE;
    gpw.nonperiodic = false;
    gpw.vdw_df = true;
    return gpw;
}

ATV::cpu::gpw::GpwDft make_rks_base(int ngrids, double rel_cutoff) {
    ATV::cpu::gpw::GpwDft gpw;
    const double L = 12.0 * kA2B;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.ngrids = ngrids;
    gpw.rel_cutoff = rel_cutoff;
    gpw.n_occ = 8;
    gpw.max_iter = 60;
    gpw.tol = 1e-7;
    gpw.xc = ATV::cpu::gpw::XcKind::PBE;
    gpw.nonperiodic = false;
    gpw.vdw_df = true;
    return gpw;
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;
    using Clock = std::chrono::steady_clock;

    // Offline CP2K 2026.1 (cp2k.psmp rev git:5e54ba2): water dimer LSD MULT 1
    // + NON_LOCAL DRSLL + MP 2×1×1 NGRIDS 4 / REL_CUTOFF 40 — REFERENCE.md.
    constexpr double kEDispCp2k = 0.15193559059130;   // Ha
    constexpr double kETotCp2k = -34.186333514352576; // Ha
    constexpr double kEDispTol = 1.0e-4;              // Ha
    // Same Γ-dimer twin-family systematic as gpw-vdw-kpoints / multigrid.
    constexpr double kETotTol = 1.0e-2;         // Ha
    constexpr double kGammaLimitTol = 1.0e-8;   // Ha
    constexpr double kIdentityTolEv = 3.0e-4;   // eV
    constexpr double kIdentityDispTol = 1.0e-5; // Ha

    const auto atoms = make_atoms();
    const auto t0 = Clock::now();

    // ── G1: Γ-limit identity (single-grid multi-k UKS vs Γ UKS) ───────────
    double e_tot_gamma = 0.0, e_disp_gamma = 0.0;
    {
        GpwDft gamma = make_uks_base(/*ngrids=*/1, /*rel_cutoff=*/0.0);
        gamma.build(atoms);
        const GpwResult& rg = gamma.run(atoms);
        if (!rg.converged) {
            std::fprintf(stderr, "FAIL: G1 Γ UKS SCF did not converge (%d)\n",
                         rg.iters);
            return 1;
        }
        e_tot_gamma = rg.e_total;
        e_disp_gamma = rg.e_disp;

        GpwDft kgamma = make_uks_base(/*ngrids=*/1, /*rel_cutoff=*/0.0);
        kgamma.k_points = {{0.0, 0.0, 0.0}};
        kgamma.k_point_weights = {1.0};
        kgamma.build(atoms);
        const GpwResult& rk = kgamma.run(atoms);
        if (!rk.converged) {
            std::fprintf(stderr,
                         "FAIL: G1 multi-k UKS Γ-limit SCF did not converge "
                         "(%d)\n",
                         rk.iters);
            return 1;
        }

        const double d_tot = std::abs(rk.e_total - e_tot_gamma);
        const double d_disp = std::abs(rk.e_disp - e_disp_gamma);
        std::printf("G1: E_tot  Γ-UKS=%.14f  kΓ-UKS=%.14f  |Δ|=%.3e Ha\n",
                    e_tot_gamma, rk.e_total, d_tot);
        std::printf("G1: E_disp Γ-UKS=%.14f  kΓ-UKS=%.14f  |Δ|=%.3e Ha\n",
                    e_disp_gamma, rk.e_disp, d_disp);
        if (d_tot > kGammaLimitTol || d_disp > kGammaLimitTol) {
            std::fprintf(stderr,
                         "FAIL: G1 Γ-limit |ΔE_tot|=%.3e |ΔE_disp|=%.3e "
                         "> %.1e Ha\n",
                         d_tot, d_disp, kGammaLimitTol);
            return 1;
        }
        if (!(std::isfinite(rk.e_disp) && std::abs(rk.e_disp) > 1e-6)) {
            std::fprintf(stderr,
                         "FAIL: G1 expected non-trivial E_disp, got %.6e\n",
                         rk.e_disp);
            return 1;
        }
    }

    // ── G2: CP2K MP 2×1×1 + multigrid multi-k UKS parity ──────────────────
    double wall_g2 = 0.0;
    {
        GpwDft mp = make_uks_base(/*ngrids=*/4, /*rel_cutoff=*/40.0);
        mp.k_mesh[0] = 2;
        mp.k_mesh[1] = 1;
        mp.k_mesh[2] = 1;
        // Match CP2K's full MP mesh (no TR fold) for a 1:1 twin.
        mp.k_time_reversal = false;
        mp.build(atoms);
        const auto t_mp0 = Clock::now();
        const GpwResult& rm = mp.run(atoms);
        wall_g2 = std::chrono::duration<double>(Clock::now() - t_mp0).count();
        if (!rm.converged) {
            std::fprintf(stderr,
                         "FAIL: G2 multi-k UKS multigrid SCF did not converge "
                         "(%d)\n",
                         rm.iters);
            return 1;
        }

        std::printf("G2: E_tot  ATV=%.14f  CP2K=%.14f  |Δ|=%.3e Ha\n",
                    rm.e_total, kETotCp2k, std::abs(rm.e_total - kETotCp2k));
        std::printf("G2: E_disp ATV=%.14f  CP2K=%.14f  |Δ|=%.3e Ha\n",
                    rm.e_disp, kEDispCp2k, std::abs(rm.e_disp - kEDispCp2k));
        std::printf("G2: wall %.1f s (CP2K twin wall 60.2 s OMP=8)\n", wall_g2);

        if (!std::isfinite(rm.e_disp)
            || std::abs(rm.e_disp - kEDispCp2k) > kEDispTol) {
            std::fprintf(stderr,
                         "FAIL: G2 E_disp got=%.14e want=%.14e ±%.1e Ha\n",
                         rm.e_disp, kEDispCp2k, kEDispTol);
            return 1;
        }
        if (std::abs(rm.e_total - kETotCp2k) > kETotTol) {
            std::fprintf(stderr,
                         "FAIL: G2 E_tot got=%.14f want=%.14f ±%.1e Ha\n",
                         rm.e_total, kETotCp2k, kETotTol);
            return 1;
        }
    }

    // ── G3: closed-shell multi-k UKS ≡ multi-k RKS (both vdW, same mesh) ─
    {
        GpwDft rks = make_rks_base(/*ngrids=*/1, /*rel_cutoff=*/0.0);
        rks.k_mesh[0] = 2;
        rks.k_mesh[1] = 1;
        rks.k_mesh[2] = 1;
        rks.k_time_reversal = false;
        rks.build(atoms);
        const GpwResult& rr = rks.run(atoms);

        GpwDft uks = make_uks_base(/*ngrids=*/1, /*rel_cutoff=*/0.0);
        uks.k_mesh[0] = 2;
        uks.k_mesh[1] = 1;
        uks.k_mesh[2] = 1;
        uks.k_time_reversal = false;
        uks.build(atoms);
        const GpwResult& ru = uks.run(atoms);

        if (!rr.converged || !ru.converged) {
            std::fprintf(stderr,
                         "FAIL: G3 multi-k RKS/UKS did not converge "
                         "(RKS=%d UKS=%d)\n",
                         rr.iters, ru.iters);
            return 1;
        }
        const double de_ev =
            std::abs(ru.e_total - rr.e_total) * ATV::units::kHartreeInEv;
        const double dd = std::abs(ru.e_disp - rr.e_disp);
        std::printf("G3: E_RKS=%.14f  E_UKS=%.14f  |Δ|=%.3e eV\n", rr.e_total,
                    ru.e_total, de_ev);
        std::printf("G3: D_RKS=%.14e  D_UKS=%.14e  |Δ|=%.3e Ha\n", rr.e_disp,
                    ru.e_disp, dd);
        if (de_ev > kIdentityTolEv || dd > kIdentityDispTol) {
            std::fprintf(stderr, "FAIL: G3 UKS≡RKS |ΔE|=%.3e eV |ΔD|=%.3e Ha\n",
                         de_ev, dd);
            return 1;
        }
        if (!(std::isfinite(ru.e_disp) && std::abs(ru.e_disp) > 1e-6)) {
            std::fprintf(stderr, "FAIL: G3 expected non-trivial E_disp\n");
            return 1;
        }
    }

    const double wall_all =
        std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("PASS gpw-vdw-uks-kpoints (G2 wall %.1f s, total %.1f s)\n",
                wall_g2, wall_all);
    return 0;
}
