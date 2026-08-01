/// @file regressions/gpw/gpw-d4-kpoint-forces/reg_gpw-d4-kpoint-forces.cpp
/// @brief END-TO-END: multi-k GPW PBE-D4 (two-body BJ) analytic ionic forces.
///
/// Capability: `assemble_gpw_forces_kpoints` folds geometry-only
/// \(F_{\mathrm{disp}} = -\nabla E_2\) (s9=0) when `d4` is set and
/// `d4_atm` is off — same path as Γ `gpw-d4-forces`, now with a dedicated
/// multi-k offline pin. ATM multi-k is `gpw-d4-atm-kpoint-forces`.
///
/// Hard pins:
///   1. `E_disp` + standalone forces — offline dftd4 3.7.0 `--grad`
///      (geometry-only; bit-identical to the Γ two-body pin).
///   2. D4 force increment — offline CP2K 2026.1 multi-k ENERGY_FORCE
///      FORCE tables D4 on minus plain (libdftd4 rebuild).
/// Decisive dual gate: product multi-k \(F(\mathrm{d4})-F(\mathrm{plain})\)
/// equals standalone D4 (same multi-k density). Fair wall vs the same
/// CP2K D4 multi-k ENERGY_FORCE twin (1 MPI × 8 OMP).
///
/// No live CP2K. See REFERENCE.md.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/scf/dftd4.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

// Same Å→Bohr as Γ gpw-d4-forces / pin scripts (CODATA-ish constant).
constexpr double kAng2Bohr = 1.8897259885789;

// ── Offline dftd4 3.7.0 two-body (s9=0), periodic 12 Å water dimer ────────
// Geometry-only ⇒ independent of electronic k mesh; same as Γ gpw-d4-forces.
constexpr double kEDispRef = -0.00099458842745846774; // Ha
constexpr double kEDispTol = 1.0e-8;                  // Ha

// dftd4 3.7.0 --mbdscale 0 --grad, remapped O H H O H H. Force = −gradient.
constexpr double kGradPer[6][3] = {
    {-5.3738961271913915e-05, -1.5591535100726387e-24, -6.1987516039752966e-21},
    {-5.9550148958160894e-05, -3.0412121452621079e-25, -1.0810592129949165e-05},
    {-5.9550148958160921e-05, -5.7104654456581212e-25, 1.0810592129949182e-05},
    {1.3609823922940861e-04, 6.8060330704498828e-25, -8.4063699110321910e-20},
    {1.8370509979413634e-05, 5.7191630572539285e-25, -8.5497124425277334e-06},
    {1.8370509979413539e-05, 1.2193640572572847e-24, 8.5497124425278096e-06},
};

// ── Offline CP2K 2026.1 multi-k FORCE Δ (D4 − plain) [Ha/Bohr] ────────────
// FORCES| tables from ENERGY_FORCE + MP 1×1×2; libdftd4 rebuild.
// Atom order O H H O H H (same COORD as the .inp files).
constexpr double kFD4DeltaCp2k[6][3] = {
    {5.37430e-05, -6.050e-16, 1.1015e-14},
    {5.95481e-05, -3.376e-16, 1.08120e-05},
    {5.95481e-05, -3.570e-17, -1.08120e-05},
    {-1.36094e-04, 4.278e-15, -4.905e-15},
    {-1.83727e-05, 4.084e-16, 8.55000e-06},
    {-1.83727e-05, 5.760e-16, -8.55000e-06},
};

constexpr double kForcePinTol = 1.0e-7; // Ha/Bohr vs dftd4 −g
// CP2K multi-k Δ vs standalone D4: residual ~4e-9 under twin cutoffs.
constexpr double kCp2kDeltaTol = 5.0e-7; // Ha/Bohr
constexpr double kWiringTol = 1.0e-7;    // Ha/Bohr (SCF residual)
constexpr double kDispIncFloor = 1.0e-6; // Ha/Bohr — non-vacuity
// Fair wall: CP2K internal timer on the multi-k D4 ENERGY_FORCE twin
// (1 MPI × 8 OMP, 2026-08-01, private libdftd4 rebuild git:5e54ba2).
constexpr double kOracleWallS = 4.117;
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

    // ── (1) Standalone two-body D4 vs offline dftd4 pins ──────────────────
    std::vector<int> Zv(6);
    std::vector<std::array<double, 3>> pos_bohr(6);
    for (int i = 0; i < 6; ++i) {
        Zv[static_cast<std::size_t>(i)] = z[i];
        pos_bohr[static_cast<std::size_t>(i)] = {
            xs[i] * kAng2Bohr, ys[i] * kAng2Bohr, zs[i] * kAng2Bohr};
    }
    const double cell_b[3] = {L * kAng2Bohr, L * kAng2Bohr, L * kAng2Bohr};
    const auto d4 = cpu::scf::dftd4(Zv, pos_bohr, cell_b, /*periodic=*/true);
    check(std::isfinite(d4.energy)
              && std::abs(d4.energy - kEDispRef) <= kEDispTol,
          "A |E_disp - dftd4 two-body pin|");
    double max_pin_err = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double want = -kGradPer[i][k];
            const double got = d4.forces[static_cast<std::size_t>(i)]
                                          [static_cast<std::size_t>(k)];
            max_pin_err = std::max(max_pin_err, std::abs(got - want));
        }
    }
    std::printf("A max|F_d4 + g_dftd4| = %.3e Ha/Bohr\n", max_pin_err);
    check(max_pin_err <= kForcePinTol, "A max|F + g_pin| vs dftd4 --grad");

    // ── (2) Standalone D4 vs CP2K multi-k FORCE Δ (D4 − plain) ────────────
    double max_inc = 0.0;
    double max_inc_cp2k = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double f = d4.forces[static_cast<std::size_t>(i)]
                                      [static_cast<std::size_t>(k)];
            const double want_cp2k = kFD4DeltaCp2k[i][k];
            max_inc = std::max(max_inc, std::abs(f));
            max_inc_cp2k = std::max(max_inc_cp2k, std::abs(f - want_cp2k));
        }
    }
    std::printf("B D4 force max|F|=%.3e  max|F-CP2K_Δ|=%.3e Ha/Bohr\n",
                max_inc, max_inc_cp2k);
    check(max_inc > kDispIncFloor, "B D4 force non-vacuous");
    check(max_inc_cp2k <= kCp2kDeltaTol,
          "B standalone D4 vs CP2K multi-k FORCE Δ");

    // ── (3) Product multi-k GPW with d4 (two-body) ────────────────────────
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
    eval.cutoff_ry = 120.0;
    eval.max_iter = 120;
    eval.tol = 1e-6;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = false;
    eval.k_mesh[0] = 1;
    eval.k_mesh[1] = 1;
    eval.k_mesh[2] = 2; // MP 1×1×2, TR-folded
    eval.d4 = true;
    eval.d4_atm = false;
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

    check(eval.last_result().converged, "C multi-k SCF converged");
    check(eval.last_result().n_kpts >= 1 && !eval.last_result().C_occ_k.empty(),
          "C multi-k C_occ_k payload present");
    const double e_disp = eval.last_result().e_disp;
    std::printf("C E_disp(multi-k product)=%.14e  |d pin|=%.2e  n_kpts=%d  "
                "iters=%d\n",
                e_disp, std::abs(e_disp - kEDispRef),
                eval.last_result().n_kpts, eval.last_result().iters);
    check(std::isfinite(e_disp) && std::abs(e_disp - kEDispRef) <= kEDispTol,
          "C product E_disp equals dftd4 two-body pin (k-mesh independent)");
    check(wall_product_s <= kWallBudget, "C product wall <= 60 s");

    // ── (4) Wiring: F(d4) − F(plain) on the multi-k assembly ───────────────
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& kind = (z[i] == 8) ? O : H;
        const double p[3] = {xs[i] * kAng2Bohr, ys[i] * kAng2Bohr,
                             zs[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(p, kind.basis, kind.pot, z[i]));
    }

    auto make_cfg = [&](bool d4_on) {
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
        m.d4 = d4_on;
        m.d4_atm = false;
        return m;
    };

    cpu::gpw::GpwDft method_d4 = make_cfg(true);
    method_d4.build(atoms);
    const auto F_on = cpu::gpw::gpw_forces_from_scf(atoms, method_d4,
                                                    eval.last_result());
    cpu::gpw::GpwDft method_plain = make_cfg(false);
    const auto F_plain = cpu::gpw::gpw_forces_from_scf(
        atoms, method_plain, eval.last_result());

    double max_wire = 0.0;
    double max_f_on = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double on =
                F_on[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
            const double pl = F_plain[static_cast<std::size_t>(i)]
                                     [static_cast<std::size_t>(k)];
            const double want = d4.forces[static_cast<std::size_t>(i)]
                                            [static_cast<std::size_t>(k)];
            max_wire = std::max(max_wire, std::abs((on - pl) - want));
            max_f_on = std::max(max_f_on, std::abs(on));
            check(std::isfinite(on) && std::isfinite(pl),
                  "D finite multi-k force component");
        }
    }
    std::printf("D multi-k wiring max|(F_d4-F_plain)-F_standalone|=%.3e  "
                "max|F_on|=%.3e Ha/Bohr\n",
                max_wire, max_f_on);
    check(max_wire <= kWiringTol, "D multi-k wiring vs standalone D4");
    check(max_f_on > 1e-8, "D product multi-k forces nontrivial");

    // ── Fair wall report (not a hard fail above 10×; schema records ratio) ─
    const double ratio = wall_product_s / kOracleWallS;
    std::printf(
        "PASS: gpw-d4-kpoint-forces  max|F+g|=%.3e  max|F-cp2k_Δ|=%.3e"
        "  max|wiring|=%.3e Ha/Bohr  wall_prod=%.3f s  T_cp2k=%.3f s  "
        "ratio=%.3f  n_kpts=%d\n",
        max_pin_err, max_inc_cp2k, max_wire, wall_product_s, kOracleWallS,
        ratio, eval.last_result().n_kpts);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    return 0;
}
