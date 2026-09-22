/// @file regressions/gpw/gpw-optb88-kpoints/reg_gpw-optb88-kpoints.cpp
/// @brief Multi-k GPW RKS optB88-vdW (OptB88+PW92 + DRSLL) offline pin.
///
/// Out-of-library consumer. Self-consistent nonlocal V_nl rides the multi-k
/// runner with XcKind::OptB88VdW. Gates (REFERENCE.md):
///   G1 Γ-limit: explicit k=(0,0,0) k-path vs shipped Γ runner (same settings);
///   G2 CP2K: MP 2×1×1 multigrid vs offline CP2K 2026.1 OptB88+DRSLL pin.
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

// Same water dimer as gpw-optb88-vdw / gpw-vdw-kpoints [Å].
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
    gpw.xc = ATV::cpu::gpw::XcKind::OptB88VdW;
    gpw.nonperiodic = false;
    gpw.vdw_df = true;
    return gpw;
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;
    using Clock = std::chrono::steady_clock;

    // Offline CP2K 2026.1 (cp2k.psmp rev git:5e54ba2, libxc 7.0.0): water
    // dimer, GGA_X_OPTB88_VDW + PW92 + NON_LOCAL DRSLL, CUTOFF 200 / NGRIDS 4 /
    // REL_CUTOFF 40, KPOINTS MONKHORST-PACK 2 1 1 — see REFERENCE.md.
    constexpr double kEDispCp2k = 0.15206184081319;   // Ha
    constexpr double kETotCp2k = -34.487914403936173; // Ha
    constexpr double kEDispTol = 1.0e-4;              // Ha
    // Same Γ-dimer twin-family systematic as gpw-vdw-kpoints / multigrid
    // (~1e-2 Ha class).
    constexpr double kETotTol = 1.0e-2; // Ha
    // Γ-limit identity (k-path at Γ vs shipped Γ runner).
    constexpr double kGammaLimitTol = 1.0e-8; // Ha

    const auto atoms = make_atoms();
    const auto t0 = Clock::now();

    // ── G1: Γ-limit identity (single-grid, cheap) ─────────────────────────
    {
        GpwDft gamma = make_base(/*ngrids=*/1, /*rel_cutoff=*/0.0);
        gamma.build(atoms);
        const GpwResult& rg = gamma.run(atoms);
        if (!rg.converged) {
            std::fprintf(stderr,
                         "FAIL: G1 Γ runner SCF did not converge (%d)\n",
                         rg.iters);
            return 1;
        }

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

        const double d_tot = std::abs(rk.e_total - rg.e_total);
        const double d_disp = std::abs(rk.e_disp - rg.e_disp);
        std::printf("G1: E_tot  Γ=%.14f  kΓ=%.14f  |Δ|=%.3e Ha\n", rg.e_total,
                    rk.e_total, d_tot);
        std::printf("G1: E_disp Γ=%.14f  kΓ=%.14f  |Δ|=%.3e Ha\n", rg.e_disp,
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
        // OptB88 activation: semi-local energy should differ from a pure
        // PBE+DRSLL reference on the same multi-k path (dual gate floor).
        GpwDft pbe = make_base(/*ngrids=*/1, /*rel_cutoff=*/0.0);
        pbe.xc = XcKind::PBE;
        pbe.k_points = {{0.0, 0.0, 0.0}};
        pbe.k_point_weights = {1.0};
        pbe.build(atoms);
        const GpwResult& rp = pbe.run(atoms);
        if (!rp.converged) {
            std::fprintf(stderr,
                         "FAIL: G1 PBE dual-gate SCF did not converge\n");
            return 1;
        }
        const double d_semi = std::abs(rk.e_total - rp.e_total);
        constexpr double kDualFloor = 1.0e-4; // Ha
        std::printf("G1: dual |E_optB88-E_PBE|=%.6e Ha (kΓ path)\n", d_semi);
        if (d_semi <= kDualFloor) {
            std::fprintf(stderr,
                         "FAIL: OptB88VdW vs PBE dual gate |dE|=%.3e Ha "
                         "(need > %.1e)\n",
                         d_semi, kDualFloor);
            return 1;
        }
    }

    // ── G2: CP2K MP 2×1×1 + multigrid parity ──────────────────────────────
    {
        GpwDft mp = make_base(/*ngrids=*/4, /*rel_cutoff=*/40.0);
        mp.k_mesh[0] = 2;
        mp.k_mesh[1] = 1;
        mp.k_mesh[2] = 1;
        // Match CP2K's full MP mesh (no TR fold) for a 1:1 twin.
        mp.k_time_reversal = false;
        mp.build(atoms);
        const auto t_mp0 = Clock::now();
        const GpwResult& rm = mp.run(atoms);
        const double wall_mp =
            std::chrono::duration<double>(Clock::now() - t_mp0).count();
        if (!rm.converged) {
            std::fprintf(stderr,
                         "FAIL: G2 MP multigrid SCF did not converge "
                         "(%d)\n",
                         rm.iters);
            return 1;
        }

        std::printf("G2: E_tot  ATV=%.14f  CP2K=%.14f  |Δ|=%.3e Ha\n",
                    rm.e_total, kETotCp2k, std::abs(rm.e_total - kETotCp2k));
        std::printf("G2: E_disp ATV=%.14f  CP2K=%.14f  |Δ|=%.3e Ha\n",
                    rm.e_disp, kEDispCp2k, std::abs(rm.e_disp - kEDispCp2k));
        std::printf("G2: wall %.1f s (CP2K twin wall 12.6 s OMP=8)\n", wall_mp);

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
    std::printf("PASS gpw-optb88-kpoints (total wall %.1f s)\n", wall_all);
    return 0;
}
