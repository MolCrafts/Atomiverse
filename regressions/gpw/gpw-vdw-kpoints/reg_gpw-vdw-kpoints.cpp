/// @file regressions/gpw/gpw-vdw-kpoints/reg_gpw-vdw-kpoints.cpp
/// @brief Multi-k GPW PBE + vdW-DF (DRSLL) on the closed-shell RKS path.
///
/// Out-of-library consumer. Self-consistent nonlocal V_nl rides the multi-k
/// runner (`build_k_energy` + `fold_dispersion_k`), single-grid and
/// REL_CUTOFF multigrid alike. Gates (REFERENCE.md):
///   G1 Γ-limit: explicit k=(0,0,0) k-path vs shipped Γ runner (same settings);
///   G2 CP2K: MP 2×1×1 multigrid vs offline CP2K 2026.1 NON_LOCAL DRSLL pin.
/// No live oracle in ctest.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859;

// Same water dimer as gpw-vdw-df-ice / gpw-vdw-multigrid [Å].
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

ATV::cpu::gpw::GpwDft make_base(int ngrids, double rel_cutoff) {
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

    // Offline CP2K 2026.1 (cp2k.psmp rev 85b8a9b49c): water dimer, PBE +
    // NON_LOCAL DRSLL, CUTOFF 200 / NGRIDS 4 / REL_CUTOFF 40, KPOINTS
    // MONKHORST-PACK 2 1 1 — see REFERENCE.md.
    constexpr double kEDispCp2k = 0.15193559113836;      // Ha
    constexpr double kETotCp2k = -34.186333514352548;    // Ha
    constexpr double kEDispTol = 1.0e-4;                 // Ha
    // Same Γ-dimer twin-family systematic as gpw-vdw-multigrid (~8e-3 Ha).
    constexpr double kETotTol = 1.0e-2;                  // Ha
    // Γ-limit identity (k-path at Γ vs shipped Γ runner).
    constexpr double kGammaLimitTol = 1.0e-8;            // Ha

    const auto atoms = make_atoms();
    const auto t0 = Clock::now();

    // ── G1: Γ-limit identity (single-grid, cheap) ─────────────────────────
    // Explicit k_points = {(0,0,0)} forces the multi-k runner at effective Γ;
    // compare against the shipped Γ runner at identical settings.
    double e_tot_gamma = 0.0, e_disp_gamma = 0.0;
    {
        GpwDft gamma = make_base(/*ngrids=*/1, /*rel_cutoff=*/0.0);
        // k_mesh default {1,1,1} → Γ runner.
        gamma.build(atoms);
        const GpwResult& rg = gamma.run(atoms);
        if (!rg.converged) {
            std::fprintf(stderr, "FAIL: G1 Γ runner SCF did not converge (%d)\n",
                         rg.iters);
            return 1;
        }
        e_tot_gamma = rg.e_total;
        e_disp_gamma = rg.e_disp;

        GpwDft kgamma = make_base(/*ngrids=*/1, /*rel_cutoff=*/0.0);
        kgamma.k_points = {{0.0, 0.0, 0.0}};
        kgamma.k_point_weights = {1.0};
        kgamma.build(atoms);
        const GpwResult& rk = kgamma.run(atoms);
        if (!rk.converged) {
            std::fprintf(stderr,
                         "FAIL: G1 k-path Γ-limit SCF did not converge (%d)\n",
                         rk.iters);
            return 1;
        }

        const double d_tot = std::abs(rk.e_total - e_tot_gamma);
        const double d_disp = std::abs(rk.e_disp - e_disp_gamma);
        std::printf("G1: E_tot  Γ=%.14f  kΓ=%.14f  |Δ|=%.3e Ha\n", e_tot_gamma,
                    rk.e_total, d_tot);
        std::printf("G1: E_disp Γ=%.14f  kΓ=%.14f  |Δ|=%.3e Ha\n", e_disp_gamma,
                    rk.e_disp, d_disp);
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

    // ── G2: CP2K MP 2×1×1 + multigrid parity ──────────────────────────────
    {
        GpwDft mp = make_base(/*ngrids=*/4, /*rel_cutoff=*/40.0);
        mp.k_mesh[0] = 2;
        mp.k_mesh[1] = 1;
        mp.k_mesh[2] = 1;
        // Match CP2K's full MP mesh (no TR fold) for a 1:1 twin; with TR
        // the energy is machine-equivalent for this large cell but the
        // pin was taken without an explicit TR fold.
        mp.k_time_reversal = false;
        mp.build(atoms);
        const auto t_mp0 = Clock::now();
        const GpwResult& rm = mp.run(atoms);
        const double wall_mp =
            std::chrono::duration<double>(Clock::now() - t_mp0).count();
        if (!rm.converged) {
            std::fprintf(stderr, "FAIL: G2 MP multigrid SCF did not converge "
                                 "(%d)\n",
                         rm.iters);
            return 1;
        }

        std::printf("G2: E_tot  ATV=%.14f  CP2K=%.14f  |Δ|=%.3e Ha\n",
                    rm.e_total, kETotCp2k, std::abs(rm.e_total - kETotCp2k));
        std::printf("G2: E_disp ATV=%.14f  CP2K=%.14f  |Δ|=%.3e Ha\n",
                    rm.e_disp, kEDispCp2k, std::abs(rm.e_disp - kEDispCp2k));
        std::printf("G2: wall %.1f s (CP2K twin wall 18.4 s OMP=8)\n", wall_mp);

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

    const double wall_all =
        std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("PASS gpw-vdw-kpoints (total wall %.1f s)\n", wall_all);
    return 0;
}
