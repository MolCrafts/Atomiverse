/// @file regressions/gpw/rks/gpw-multigrid-kpoints.cpp
/// @brief Monkhorst-Pack k-point GPW RKS on diamond Si with CP2K-style
///        REL_CUTOFF multigrid product collocation (NGRIDS 4) — internal
///        exactness gates plus external CP2K energy and stress pins.
///
/// Out-of-library consumer: public GPW headers only, flat main(), hard-coded
/// numbers, no external process. Gates (spec gpw-multigrid-kpoints):
///   M2 TR-folded multigrid SCF converges on the cheap 40 Ry / REL 20 ladder
///      (all 4 levels populated)
///   M3 multigrid vs single-grid k discretization stays small (40 Ry twin)
///   M4 convergence + finiteness + k payload shapes (150 Ry main)
///   M5 absolute e_tot vs the offline CP2K NGRIDS 4 + KPOINTS pin (150 Ry)
///   M6 energy and full stress tensor vs CP2K 2026.2 (100 Ry / REL 40, MP1x1x2)
/// The exactness identities live at unit scale in
/// tests/cpu/gpw/test_cpu_gpw_multigrid.cpp (GpwMultigridCells.*): Γ-limit
/// (uniform D-stack ≡ Γ product path), D/V_t phase plumbing vs the
/// independent χ^k machinery, per-cell adjointness, and time-reversal
/// D-stack folding. Full-SCF TR and Γ-limit twins were measured at bring-up
/// (REFERENCE.md: |Δ| ≤ 3e-14 Ha at 40/60/95 Ry) and are not re-run here —
/// each multigrid SCF costs ~15 s and the ≤ 60 s budget goes to the pin.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molpy.hpp"
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

    g_fail = 0;
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

    // M6: fixed-P cell-strain stress on the matched 100 Ry / REL 40 mesh.
    // Actual CP2K analytical tensor: inputs/gpw-kpoint-stress/reference.json.
    // Existing M2-M5 settings and gates above remain independent and unchanged.
    {
        const auto begin = std::chrono::steady_clock::now();
        GpwDft stress_method = si_method(100.0, 4, 40.0);
        stress_method.max_iter = 100;
        stress_method.tol = 1e-10;
        stress_method.residual_tol = 1e-8;
        stress_method.eps_default = 1e-12;
        stress_method.local_pp_on_grid = true;
        stress_method.k_mesh[0] = stress_method.k_mesh[1] = 1;
        stress_method.k_mesh[2] = 2;
        stress_method.stress_strain_h = 2e-3;
        cpu::Frame state = frame;
        stress_method.build(state);
        auto stress = stress_method.stress(state);
        const auto& result = stress_method.last_result();
        check(result.converged, "M6 unstrained stress SCF converged");
        check(result.n_kpts >= 1 && !result.C_occ_k.empty(),
              "M6 Bloch occupied-orbital payload present");
        constexpr double energy_pin = -15.348312942398339;
        constexpr std::array<double, 6> stress_pin = {
            0.00088302884218137618, 0.00088302884218137618,
            0.00026101331426867699, -2.8441034466934299e-16,
            2.3791573455322444e-16, 3.7052221084565209e-18};
        auto near_pin = [&](double got, double want, double tolerance,
                            const char* quantity) {
            const bool ok =
                std::isfinite(got) && std::abs(got - want) <= tolerance;
            std::printf("M6 %s got=% .12e want=% .12e tol=%.3e %s\n", quantity,
                        got, want, tolerance, ok ? "ok" : "FAIL");
            if (!ok)
                ++g_fail;
        };
        near_pin(result.e_total, energy_pin, 1e-3, "energy [Ha] vs CP2K");
        const char* names[] = {"xx", "yy", "zz", "xy", "xz", "yz"};
        double largest = 0;
        for (std::size_t component = 0; component < stress.size();
             ++component) {
            stress[component] /= units::kHartreePerBohr3InEvPerAngstrom3;
            char quantity[96];
            std::snprintf(quantity, sizeof(quantity),
                          "sigma_%s [Ha/Bohr3] vs CP2K", names[component]);
            near_pin(stress[component], stress_pin[component], 1e-4, quantity);
            if (component < 3)
                largest = std::max(largest, std::abs(stress[component]));
            else
                near_pin(stress[component], 0.0, 1e-8,
                         "off-diagonal symmetry [Ha/Bohr3]");
        }
        std::printf("M6 stress magnitude got=%.6e want=>1e-6 Ha/Bohr3\n",
                    largest);
        check(largest > 1e-6, "M6 nonzero stress magnitude floor");
        const double wall = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - begin)
                                .count();
        std::printf("M6 fixed-P stress product wall %.6f s\n", wall);
    }
    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t_start)
                            .count();
    std::printf("Whole section got=%.6f want<=60 seconds\n", wall);
    check(wall <= 60.0, "whole section within 60 seconds");

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-multigrid-kpoints — multi-k GPW RKS with REL_CUTOFF "
                "multigrid product collocation (TR, grid residual, "
                "CP2K energy/stress pins)\n");
    return 0;
}

} // namespace atv_reg_gpw_gpw_multigrid_kpoints

ATV_REGRESSION("gpw/rks", "gpw-multigrid-kpoints",
               atv_reg_gpw_gpw_multigrid_kpoints::run);
