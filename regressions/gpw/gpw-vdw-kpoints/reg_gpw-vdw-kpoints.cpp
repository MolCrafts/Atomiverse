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

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

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

ATV::cpu::Frame make_frame() {
    using namespace ATV;
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = kZ[i];
        frame.span(kAtom_x)[i] = kAng[i][0];
        frame.span(kAtom_y)[i] = kAng[i][1];
        frame.span(kAtom_z)[i] = kAng[i][2];
    }
    frame.box = cpu::Box::cubic(12.0);
    return frame;
}

std::vector<ATV::cpu::gpw::Kind> water_kinds() {
    using namespace ATV::cpu::gpw;
    return {Kind{molrs::Element::O,
                 GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
                 GthPotential(molrs::Element::O, "GTH-PBE-q6")},
            Kind{molrs::Element::H,
                 GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
                 GthPotential(molrs::Element::H, "GTH-PBE-q1")}};
}

ATV::cpu::gpw::GpwDft make_base(int ngrids, double rel_cutoff) {
    ATV::cpu::gpw::GpwDft gpw(water_kinds(), 200.0, ATV::cpu::gpw::XcKind::PBE);
    gpw.ngrids = ngrids;
    gpw.rel_cutoff = rel_cutoff;
    gpw.n_occ = 8;
    gpw.max_iter = 60;
    gpw.tol = 1e-7;
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
    constexpr double kEDispCp2k = 0.15193559113836;   // Ha
    constexpr double kETotCp2k = -34.186333514352548; // Ha
    constexpr double kEDispTol = 1.0e-4;              // Ha
    // Same Γ-dimer twin-family systematic as gpw-vdw-multigrid (~8e-3 Ha).
    constexpr double kETotTol = 1.0e-2; // Ha
    // Γ-limit identity (k-path at Γ vs shipped Γ runner).
    constexpr double kGammaLimitTol = 1.0e-8; // Ha

    ATV::cpu::Frame frame = make_frame();
    const auto t0 = Clock::now();

    // ── G1: Γ-limit identity (single-grid, cheap) ─────────────────────────
    // Explicit k_points = {(0,0,0)} forces the multi-k runner at effective Γ;
    // compare against the shipped Γ runner at identical settings.
    double e_tot_gamma = 0.0, e_disp_gamma = 0.0;
    {
        GpwDft gamma = make_base(/*ngrids=*/1, /*rel_cutoff=*/0.0);
        // k_mesh default {1,1,1} → Γ runner.
        gamma.build(frame);
        const GpwResult& rg = gamma.run(frame);
        if (!rg.converged) {
            std::fprintf(stderr,
                         "FAIL: G1 Γ runner SCF did not converge (%d)\n",
                         rg.iters);
            return 1;
        }
        e_tot_gamma = rg.e_total;
        e_disp_gamma = rg.e_disp;

        GpwDft kgamma = make_base(/*ngrids=*/1, /*rel_cutoff=*/0.0);
        kgamma.k_points = {{0.0, 0.0, 0.0}};
        kgamma.k_point_weights = {1.0};
        kgamma.build(frame);
        const GpwResult& rk = kgamma.run(frame);
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
        mp.build(frame);
        const auto t_mp0 = Clock::now();
        const GpwResult& rm = mp.run(frame);
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
