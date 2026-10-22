/// @file regressions/gpw/gpw-multigrid-kpoints/reg_gpw-multigrid-kpoints.cpp
/// @brief Monkhorst-Pack k-point GPW RKS on diamond Si with CP2K-style
///        REL_CUTOFF multigrid product collocation (NGRIDS 4) — internal
///        exactness gates plus the offline CP2K 2026.1 pin from REFERENCE.md.
///
/// Out-of-library consumer: public GPW headers only, flat main(), hard-coded
/// numbers, no external process. Gates (spec gpw-multigrid-kpoints):
///   M2 TR-folded multigrid SCF converges on the cheap 40 Ry / REL 20 ladder
///      (all 4 levels populated)
///   M3 multigrid vs single-grid k discretization stays small (40 Ry twin)
///   M4 convergence + finiteness + k payload shapes (150 Ry main)
///   M5 absolute e_tot vs the offline CP2K NGRIDS 4 + KPOINTS pin (150 Ry)
/// The exactness identities live at unit scale in
/// tests/cpu/gpw/test_cpu_gpw_multigrid.cpp (GpwMultigridCells.*): Γ-limit
/// (uniform D-stack ≡ Γ product path), D/V_t phase plumbing vs the
/// independent χ^k machinery, per-cell adjointness, and time-reversal
/// D-stack folding. Full-SCF TR and Γ-limit twins were measured at bring-up
/// (REFERENCE.md: |Δ| ≤ 3e-14 Ha at 40/60/95 Ry) and are not re-run here —
/// each multigrid SCF costs ~15 s and the ≤ 60 s budget goes to the pin.

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
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_multigrid_kpoints {

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

// Diamond Si in the body-centred-tetragonal 4-atom cell (a_t = a/√2, c = a) —
// the orthorhombic-compatible setting of the fcc diamond lattice (the same
// geometry as regressions/gpw/gpw-kpoints-si and the CP2K twin input).
ATV::cpu::Frame si_bct(double a_ang) {
    const double at = a_ang / std::sqrt(2.0);
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    ATV::cpu::Frame frame(4);
    for (int i = 0; i < 4; ++i) {
        frame.span(ATV::kAtom_type)[i] = 14;
        frame.span(ATV::kAtom_x)[i] = frac[i][0] * at;
        frame.span(ATV::kAtom_y)[i] = frac[i][1] * at;
        frame.span(ATV::kAtom_z)[i] = frac[i][2] * a_ang;
    }
    frame.box = ATV::cpu::Box::orthorhombic(at, at, a_ang);
    return frame;
}

// One shared method shape; the main run is the CP2K twin &MGRID (CUTOFF 150,
// NGRIDS 4, REL_CUTOFF 60); identity gates run the same machinery on a
// cheaper 40 Ry / REL 20 ladder; single grid = ngrids 1 (rel 0).
ATV::cpu::gpw::GpwDft si_method(double cutoff_ry, int ngrids,
                                double rel_cutoff) {
    using namespace ATV::cpu::gpw;
    GpwDft m{{Kind{molrs::Element::Si,
                   GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
                   GthPotential(molrs::Element::Si, "GTH-PBE-q4")}},
             cutoff_ry,
             ATV::cpu::gpw::XcKind::PBE};
    m.ngrids = ngrids;
    m.rel_cutoff = rel_cutoff;
    m.n_occ = 8; // 4 atoms × 4 valence e⁻, closed shell
    m.max_iter = 60;
    m.tol = 1e-8;
    m.eps_default = 1e-7;
    return m;
}

} // namespace

int run() {
    using namespace ATV;
    using cpu::gpw::GpwDft;

    const auto t_start = std::chrono::steady_clock::now();
    auto mark = [&](const char* what) {
        std::printf("  [wall] %s at %.1f s\n", what,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_start)
                        .count());
    };

    auto frame = si_bct(5.43); // Si lattice constant 5.43 Å

    // ── M2: TR-folded multigrid SCF on the cheap ladder (40 Ry / REL 20) ───
    // (The 8k-vs-folded-4k full-SCF identity was measured machine-exact at
    // bring-up — REFERENCE.md; the folding mechanism itself is unit-pinned.)
    double e_mg_id = 0.0; // TR-folded multigrid energy, reused by M3
    {
        GpwDft tr = si_method(40.0, 4, 20.0);
        tr.k_mesh[0] = tr.k_mesh[1] = tr.k_mesh[2] = 2;
        tr.k_time_reversal = true; // 4 k, doubled weights
        tr.build(frame);
        const auto& rt = tr.run(frame);
        check(rt.converged, "M2 TR-mesh multigrid SCF converged");
        mark("M2 TR-mesh done");
        e_mg_id = rt.e_total;
        std::printf("M2: E(4k TR, 40 Ry mg) = %.9f Ha (iters=%d)\n", rt.e_total,
                    rt.iters);
    }

    // ── M3: multigrid vs single-grid discretization (40 Ry) ────────────────
    // Reuses the M2 TR-folded multigrid energy; one single-grid twin run.
    // (CP2K's own NGRIDS 4 ↔ 1 gap on this system is 1.5e-10 Ha — the
    // multigrid is a cost optimization, not a physics change; the ATV gap is
    // the REL_CUTOFF product-screen residual, mHa-class budget.)
    {
        GpwDft sg = si_method(40.0, 1, 0.0);
        sg.k_mesh[0] = sg.k_mesh[1] = sg.k_mesh[2] = 2;
        sg.build(frame);
        const auto& rs = sg.run(frame);
        check(rs.converged, "M3 single-grid SCF converged");
        mark("M3 single-grid done");
        const double dgrid = std::abs(e_mg_id - rs.e_total);
        std::printf("M3: E(mg 40) = %.10f Ha, E(sg 40) = %.10f Ha, "
                    "|Δ(mg-sg)| = %.4e Ha\n",
                    e_mg_id, rs.e_total, dgrid);
        check(dgrid <= 1e-3, "M3 multigrid vs single-grid |Δ| <= 1e-3 Ha");
    }

    // ── Main run: MP 2×2×2 TR at 150 Ry, NGRIDS 4 (M4-M5) ──────────────────
    {
        GpwDft mp = si_method(150.0, 4, 60.0);
        mp.k_mesh[0] = mp.k_mesh[1] = mp.k_mesh[2] = 2;
        mp.build(frame);
        const auto t0 = std::chrono::steady_clock::now();
        const auto& rm = mp.run(frame);
        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();
        std::printf("MAIN: wall %.1f s (REFERENCE perf twin: CP2K NGRIDS 4 "
                    "run)\n",
                    wall);
        check(rm.converged, "M4 MP 2x2x2 multigrid SCF converged");
        check(std::isfinite(rm.e_total), "M4 finite e_total");
        check(rm.k_frac.size() == 4 && rm.k_weights.size() == 4
                  && rm.eps_k.size() == 4,
              "M4 k payload shapes (4 TR k-points)");
        std::printf("MAIN: E(MP 2x2x2, 150 Ry, NGRIDS 4) = %.10f Ha "
                    "(iters=%d)\n",
                    rm.e_total, rm.iters);

        // M5: offline CP2K 2026.1 pin (REFERENCE.md): NGRIDS 4, REL_CUTOFF
        // 60, MONKHORST-PACK 2 2 2, CUTOFF 150 Ry.
        constexpr double kEPin = -15.802416412503845; // Ha (CP2K 2026.1)
        constexpr double kTolAbs = 1e-3;              // Ha
        const double dpin = std::abs(rm.e_total - kEPin);
        std::printf("M5: |E − pin| = %.3e Ha (pin %.10f)\n", dpin, kEPin);
        check(dpin <= kTolAbs, "M5 absolute e_tot within 1e-3 Ha of CP2K pin");
    }

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-multigrid-kpoints — multi-k GPW RKS with REL_CUTOFF "
                "multigrid product collocation (TR, grid residual, "
                "CP2K pin)\n");
    return 0;
}

} // namespace atv_reg_gpw_gpw_multigrid_kpoints

ATV_REGRESSION("gpw/rks", "gpw-multigrid-kpoints",
               atv_reg_gpw_gpw_multigrid_kpoints::run);
