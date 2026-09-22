/// @file regressions/gpw/gpw-d3bj-kpoint-forces/reg_gpw-d3bj-kpoint-forces.cpp
/// @brief END-TO-END: multi-k GPW PBE-D3(BJ) analytic ionic forces vs offline
///        CP2K Quickstep ENERGY_FORCE twins.
///
/// Capability: `assemble_gpw_forces_kpoints` folds geometry-only
/// \(F_{\mathrm{disp}} = -\nabla E_{\mathrm{disp}}^{\mathrm{D3(BJ)}}\) when
/// `d3bj` is set — same path as Γ `gpw-d3bj-forces`, now with a dedicated
/// multi-k offline pin (deps closed: Γ BJ forces + multi-k BJ stress).
///
/// Hard pins from two offline CP2K 2026.1 multi-k ENERGY_FORCE runs
/// (MONKHORST-PACK 1 1 2, D3BJ + plain):
///   1. `E_disp` — CP2K's printed "Dispersion energy" (k-mesh independent).
///   2. `F_disp` — CP2K F(D3BJ) − F(plain) per atom.
/// Decisive dual gate: product multi-k \(F(\mathrm{d3bj})-F(\mathrm{plain})\)
/// equals standalone D3(BJ). Fair wall vs the same CP2K D3BJ multi-k
/// ENERGY_FORCE twin (1 MPI × 8 OMP).
///
/// No live CP2K. See REFERENCE.md.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/dftd3_params.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

constexpr double kAng2Bohr = 1.8897259885789;

// ── Offline CP2K 2026.1 multi-k pins (see REFERENCE.md) ───────────────────
// Printed "Dispersion energy" on the D3BJ k-mesh twin (EPS_SCF 1e-8).
// Bit-identical to the Γ pin: D3 is geometry-only.
constexpr double kEDispCp2k = -0.00137627769481; // Ha
constexpr double kEDispTol = 1.0e-8;             // Ha

// CP2K FORCES| table for the multi-k D3(BJ) run [Ha/Bohr], order O H H O H H.
constexpr double kForceD3BjCp2k[6][3] = {
    {-8.18231441e-02, 4.01492980e-10, 3.72641618e-10},
    {4.08872726e-02, 1.13134904e-11, 1.31270241e-01},
    {4.08872726e-02, 1.06384740e-11, -1.31270241e-01},
    {-8.05124309e-02, 3.58188268e-10, 3.28458012e-10},
    {3.75636484e-02, 1.35480669e-11, 1.35137199e-01},
    {3.75636484e-02, 1.32671690e-11, -1.35137199e-01},
};
// CP2K FORCES| table for the plain multi-k twin [Ha/Bohr].
constexpr double kForcePlainCp2k[6][3] = {
    {-8.19227599e-02, 4.01489128e-10, 3.72641307e-10},
    {4.08225174e-02, 1.13134323e-11, 1.31264301e-01},
    {4.08225174e-02, 1.06385551e-11, -1.31264301e-01},
    {-8.04237542e-02, 3.58191250e-10, 3.28453339e-10},
    {3.76338733e-02, 1.35482542e-11, 1.35133744e-01},
    {3.76338733e-02, 1.32675516e-11, -1.35133744e-01},
};

// Difference of two independently converged CP2K SCFs (EPS_SCF 1e-8) — same
// floor class as Γ gpw-d3bj-forces (measured ~1e-6 vs max|F_disp| ~1e-4).
constexpr double kFDispTol = 5.0e-6; // Ha/Bohr
// Product multi-k wiring: same density both sides of the difference.
constexpr double kWiringTol = 1.0e-7; // Ha/Bohr
// Non-vacuity of the dispersion force signal.
constexpr double kFDispFloor = 1.0e-5; // Ha/Bohr
// Fair wall: CP2K internal timer on the multi-k D3BJ ENERGY_FORCE twin
// (1 MPI × 8 OMP, 2026-08-01, cp2k.psmp git:5e54ba2).
constexpr double kOracleWallS = 3.931;
constexpr double kWallBudget = 60.0; // s product path

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    constexpr double L = 12.0; // Å
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double xs[6] = {0.000000, 0.758602, 0.758602,
                          3.000000, 3.758602, 3.758602};
    const double ys[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double zs[6] = {0.000000, 0.504284, -0.504284,
                          0.000000, 0.504284, -0.504284};

    // ── (1) Standalone D3(BJ): energy + gradient vs the CP2K pins ─────────
    std::vector<int> Zv(6);
    std::vector<std::array<double, 3>> pos_bohr(6);
    for (int i = 0; i < 6; ++i) {
        Zv[static_cast<std::size_t>(i)] = z[i];
        pos_bohr[static_cast<std::size_t>(i)] = {
            xs[i] * kAng2Bohr, ys[i] * kAng2Bohr, zs[i] * kAng2Bohr};
    }
    const double cell_b[3] = {L * kAng2Bohr, L * kAng2Bohr, L * kAng2Bohr};
    // PBE-D3(BJ): s6=1, s8=0.7875, a1=0.4289, a2=4.4407 Bohr; CP2K CN form.
    const cpu::gpw::D3Result d3 = cpu::gpw::dftd3_pbe_bj(
        Zv, pos_bohr, cell_b, /*periodic=*/true, cpu::gpw::D3Params{},
        cpu::gpw::kD3Cp2kPairCutoffBohr, /*s6=*/1.0, /*s8=*/0.7875,
        /*a1=*/0.4289, /*a2=*/4.4407, /*cp2k_cn=*/true,
        /*calculate_c9=*/false);

    check(std::isfinite(d3.energy)
              && std::abs(d3.energy - kEDispCp2k) <= kEDispTol,
          "A |E_disp - CP2K pin|");

    double max_fdisp_err = 0.0;
    double max_fdisp = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double want = kForceD3BjCp2k[i][k] - kForcePlainCp2k[i][k];
            const double got = d3.forces[static_cast<std::size_t>(i)]
                                        [static_cast<std::size_t>(k)];
            max_fdisp_err = std::max(max_fdisp_err, std::abs(got - want));
            max_fdisp = std::max(max_fdisp, std::abs(want));
        }
    }
    std::printf("A max|F_disp - (F_d3bj-F_plain)_cp2k|=%.3e  signal=%.3e "
                "Ha/Bohr\n",
                max_fdisp_err, max_fdisp);
    check(max_fdisp > kFDispFloor, "A F_disp signal non-vacuous");
    check(max_fdisp_err <= kFDispTol, "A max|F_disp - CP2K multi-k FORCE Δ|");

    // ── (2) Product multi-k GPW with d3bj ─────────────────────────────────
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = xs[i];
        frame.span(kAtom_y)[i] = ys[i];
        frame.span(kAtom_z)[i] = zs[i];
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    }
    frame.box = cpu::Box::cubic(L);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::gpw::GPWEvaluator eval({O, H});
    eval.cutoff_ry = 120.0; // dispersion gates geometry-only; dual-gate budget
    eval.max_iter = 120;
    eval.tol = 1e-6;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = false;
    eval.k_mesh[0] = 1;
    eval.k_mesh[1] = 1;
    eval.k_mesh[2] = 2; // MP 1×1×2, TR-folded
    eval.d3bj = true;
    eval.compute_forces = true;
    eval.compute_stress = false;

    const auto t0 = clock::now();
    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);
    const auto t1 = clock::now();
    const double wall_product_s =
        std::chrono::duration<double>(t1 - t0).count();

    check(eval.last_result().converged, "B multi-k SCF converged");
    check(eval.last_result().n_kpts >= 1 && !eval.last_result().C_occ_k.empty(),
          "B multi-k C_occ_k payload present");
    const double e_disp = eval.last_result().e_disp;
    std::printf("B E_disp(multi-k product)=%.14e  |d pin|=%.2e  n_kpts=%d  "
                "iters=%d  wall=%.3f s\n",
                e_disp, std::abs(e_disp - kEDispCp2k),
                eval.last_result().n_kpts, eval.last_result().iters,
                wall_product_s);
    check(std::isfinite(e_disp) && std::abs(e_disp - kEDispCp2k) <= kEDispTol,
          "B product E_disp equals CP2K pin (k-mesh independent)");
    check(wall_product_s <= kWallBudget, "B product wall <= 60 s");

    // ── (3) Wiring: F(d3bj) − F(plain) on the multi-k assembly ────────────
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& kind = (z[i] == 8) ? O : H;
        const double p[3] = {xs[i] * kAng2Bohr, ys[i] * kAng2Bohr,
                             zs[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(p, kind.basis, kind.pot, z[i]));
    }

    auto make_cfg = [&](bool d3bj) {
        cpu::gpw::GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = L * kAng2Bohr;
        m.cutoff_ry = 120.0;
        m.n_occ = 8;
        m.max_iter = 120;
        m.tol = 1e-6;
        m.mix = 0.4;
        m.xc = cpu::gpw::XcKind::PBE;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.d3bj = d3bj;
        return m;
    };

    cpu::gpw::GpwDft method_bj = make_cfg(true);
    method_bj.build(atoms);
    const auto F_on =
        cpu::gpw::gpw_forces_from_scf(atoms, method_bj, eval.last_result());
    cpu::gpw::GpwDft method_plain = make_cfg(false);
    const auto F_plain =
        cpu::gpw::gpw_forces_from_scf(atoms, method_plain, eval.last_result());

    double max_wire = 0.0;
    double max_f_on = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double on =
                F_on[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
            const double pl = F_plain[static_cast<std::size_t>(i)]
                                     [static_cast<std::size_t>(k)];
            const double want = d3.forces[static_cast<std::size_t>(i)]
                                         [static_cast<std::size_t>(k)];
            max_wire = std::max(max_wire, std::abs((on - pl) - want));
            max_f_on = std::max(max_f_on, std::abs(on));
            check(std::isfinite(on) && std::isfinite(pl),
                  "C finite multi-k force component");
        }
    }
    std::printf("C multi-k wiring max|(F_d3bj-F_plain)-F_standalone|=%.3e  "
                "max|F_on|=%.3e Ha/Bohr\n",
                max_wire, max_f_on);
    check(max_wire <= kWiringTol, "C multi-k wiring vs standalone D3(BJ)");
    check(max_f_on > 1e-8, "C product multi-k forces nontrivial");

    const double ratio = wall_product_s / kOracleWallS;
    std::printf(
        "PASS: gpw-d3bj-kpoint-forces  |dE_disp|=%.2e  max|F_disp-cp2k|=%.3e"
        "  max|wiring|=%.3e Ha/Bohr  wall_prod=%.3f s  T_cp2k=%.3f s  "
        "ratio=%.3f  n_kpts=%d\n",
        std::abs(e_disp - kEDispCp2k), max_fdisp_err, max_wire, wall_product_s,
        kOracleWallS, ratio, eval.last_result().n_kpts);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    return 0;
}
