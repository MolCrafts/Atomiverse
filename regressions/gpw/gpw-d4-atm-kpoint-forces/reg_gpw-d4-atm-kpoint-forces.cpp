/// @file regressions/gpw/gpw-d4-atm-kpoint-forces/reg_gpw-d4-atm-kpoint-forces.cpp
/// @brief END-TO-END: multi-k GPW PBE-D4+ATM (s9=1) analytic ionic forces.
///
/// Capability: `assemble_gpw_forces_kpoints` folds geometry-only
/// \(F_{\mathrm{disp}} = -\nabla(E_2 + E_{\mathrm{ATM}})\) with \(s_9=1\)
/// when `d4` + `d4_atm` are set — same path as Γ `gpw-d4-atm-forces`, now
/// with a dedicated multi-k offline pin.
///
/// Hard pins:
///   1. `E_disp` + standalone forces — offline dftd4 3.7.0 `--grad`
///      (geometry-only; bit-identical to the Γ pin).
///   2. ATM force increment — offline CP2K 2026.1 multi-k ENERGY_FORCE
///      FORCE tables with DFTD4 ATM on minus two-body (libdftd4 rebuild).
/// Decisive dual gate: product multi-k \(F(\mathrm{d4\_atm})-F(\mathrm{plain})\)
/// equals standalone D4+ATM (same multi-k density). Fair wall vs the same
/// CP2K D4+ATM multi-k ENERGY_FORCE twin (1 MPI × 8 OMP).
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

// Same Å→Bohr as Γ gpw-d4-atm-forces / pin scripts (CODATA-ish constant).
constexpr double kAng2Bohr = 1.8897259885789;

// ── Offline dftd4 3.7.0 (s9=1), periodic 12 Å water dimer (REFERENCE) ─────
// Geometry-only ⇒ independent of electronic k mesh; same as Γ gpw-d4-atm-forces.
constexpr double kEDispAtmRef = -9.9554475834240310e-04; // Ha
constexpr double kEDispTol = 1.0e-8;                     // Ha

// dftd4 3.7.0 --grad, ATM on, remapped O H H O H H. Force = −gradient.
constexpr double kGradPerAtm[6][3] = {
    {-5.3595427766884764e-05, -3.3533040430274789e-24, -4.7093130846514167e-22},
    {-5.9463596930597288e-05, -8.8497804033039618e-24, -1.0397154303254306e-05},
    {-5.9463596930597301e-05, -4.9419094613683133e-24, 1.0397154303254314e-05},
    {1.3610085695186597e-04, 1.3230988245241392e-23, -8.4467780181584667e-20},
    {1.8210882338106763e-05, -1.1201532595111321e-24, -8.1642912818385456e-06},
    {1.8210882338106668e-05, 1.8647645948199754e-23, 8.1642912818386235e-06},
};
// Two-body twin (dftd4 --mbdscale 0 --grad).
constexpr double kGradPer2b[6][3] = {
    {-5.3738961271913915e-05, -1.5591535100726387e-24, -6.1987516039752966e-21},
    {-5.9550148958160894e-05, -3.0412121452621079e-25, -1.0810592129949165e-05},
    {-5.9550148958160921e-05, -5.7104654456581212e-25, 1.0810592129949182e-05},
    {1.3609823922940861e-04, 6.8060330704498828e-25, -8.4063699110321910e-20},
    {1.8370509979413634e-05, 5.7191630572539285e-25, -8.5497124425277334e-06},
    {1.8370509979413539e-05, 1.2193640572572847e-24, 8.5497124425278096e-06},
};

// ── Offline CP2K 2026.1 multi-k FORCE Δ (ATM − 2b) [Ha/Bohr] ──────────────
// FORCES| tables from ENERGY_FORCE + MP 1×1×2 + DFTD4, libdftd4 rebuild.
// Atom order O H H O H H (same COORD as the .inp files).
constexpr double kFAtmDeltaCp2k[6][3] = {
    {-1.195e-07, 6.300e-15, -1.739e-15},
    {-9.910e-08, 2.321e-16, -4.190e-07},
    {-9.910e-08, -2.590e-16, 4.190e-07},
    {2.160e-08, -2.265e-15, -6.594e-15},
    {1.481e-07, 4.099e-16, -3.910e-07},
    {1.481e-07, -3.080e-16, 3.910e-07},
};

// Direct pin vs dftd4 printed gradient absorbs oracle non-conservativeness
// (~2.4e-8 Ha/Bohr periodic ATM; ATM increment ~4.1e-7).
constexpr double kForcePinTol = 5.0e-8; // Ha/Bohr
// CP2K multi-k Δ vs ATV ATM increment: residual ~2.4e-8 (same defect + cutoff).
constexpr double kAtmDeltaTol = 5.0e-8; // Ha/Bohr
constexpr double kWiringTol = 1.0e-7;   // Ha/Bohr (SCF residual)
constexpr double kAtmIncFloor = 1.0e-8; // Ha/Bohr — non-vacuity
// Fair wall: CP2K internal timer on the multi-k D4+ATM ENERGY_FORCE twin
// (1 MPI × 8 OMP, 2026-08-01, private libdftd4 rebuild git:5e54ba2).
constexpr double kOracleWallS = 15.806;
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

    // ── (1) Standalone D4+ATM vs offline dftd4 pins ───────────────────────
    std::vector<int> Zv(6);
    std::vector<std::array<double, 3>> pos_bohr(6);
    for (int i = 0; i < 6; ++i) {
        Zv[static_cast<std::size_t>(i)] = z[i];
        pos_bohr[static_cast<std::size_t>(i)] = {
            xs[i] * kAng2Bohr, ys[i] * kAng2Bohr, zs[i] * kAng2Bohr};
    }
    const double cell_b[3] = {L * kAng2Bohr, L * kAng2Bohr, L * kAng2Bohr};
    const auto d4atm = cpu::scf::dftd4(
        Zv, pos_bohr, cell_b, /*periodic=*/true, /*s6=*/1.0,
        /*s8=*/0.95948085, /*a1=*/0.38574991, /*a2=*/4.80688534, /*ga=*/3.0,
        /*gc=*/2.0, /*wf=*/6.0, /*charge=*/0.0, /*cn_cut=*/30.0,
        /*disp2_cut=*/60.0, /*s9=*/1.0);
    check(std::isfinite(d4atm.energy)
              && std::abs(d4atm.energy - kEDispAtmRef) <= kEDispTol,
          "A |E_disp - dftd4 ATM pin|");
    double max_pin_err = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double want = -kGradPerAtm[i][k];
            const double got = d4atm.forces[static_cast<std::size_t>(i)]
                                           [static_cast<std::size_t>(k)];
            max_pin_err = std::max(max_pin_err, std::abs(got - want));
        }
    }
    std::printf("A max|F_d4atm + g_dftd4| = %.3e Ha/Bohr\n", max_pin_err);
    check(max_pin_err <= kForcePinTol, "A max|F + g_pin| vs dftd4 --grad");

    // ── (2) ATM-only increment vs CP2K multi-k FORCE Δ ────────────────────
    const auto d4_2b = cpu::scf::dftd4(Zv, pos_bohr, cell_b, /*periodic=*/true);
    double max_inc = 0.0;
    double max_inc_cp2k = 0.0;
    double max_inc_dftd4 = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double inc = d4atm.forces[static_cast<std::size_t>(i)]
                                           [static_cast<std::size_t>(k)]
                               - d4_2b.forces[static_cast<std::size_t>(i)]
                                             [static_cast<std::size_t>(k)];
            const double want_cp2k = kFAtmDeltaCp2k[i][k];
            const double want_dftd4 = -(kGradPerAtm[i][k] - kGradPer2b[i][k]);
            max_inc = std::max(max_inc, std::abs(inc));
            max_inc_cp2k =
                std::max(max_inc_cp2k, std::abs(inc - want_cp2k));
            max_inc_dftd4 =
                std::max(max_inc_dftd4, std::abs(inc - want_dftd4));
        }
    }
    std::printf("B ATM increment max|dF|=%.3e  max|dF-CP2K|=%.3e  "
                "max|dF-dftd4|=%.3e Ha/Bohr\n",
                max_inc, max_inc_cp2k, max_inc_dftd4);
    check(max_inc > kAtmIncFloor, "B ATM force increment non-vacuous");
    check(max_inc_cp2k <= kAtmDeltaTol,
          "B ATM increment vs CP2K multi-k FORCE Δ");
    check(max_inc_dftd4 <= kForcePinTol,
          "B ATM increment vs dftd4 --grad delta");

    // ── (3) Product multi-k GPW with d4 + d4_atm ──────────────────────────
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
    eval.d4_atm = true;
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
                e_disp, std::abs(e_disp - kEDispAtmRef),
                eval.last_result().n_kpts, eval.last_result().iters);
    check(std::isfinite(e_disp) && std::abs(e_disp - kEDispAtmRef) <= kEDispTol,
          "C product E_disp equals dftd4 ATM pin (k-mesh independent)");
    check(wall_product_s <= kWallBudget, "C product wall <= 60 s");

    // ── (4) Wiring: F(d4_atm) − F(plain) on the multi-k assembly ──────────
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& kind = (z[i] == 8) ? O : H;
        const double p[3] = {xs[i] * kAng2Bohr, ys[i] * kAng2Bohr,
                             zs[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(p, kind.basis, kind.pot, z[i]));
    }

    auto make_cfg = [&](bool d4, bool d4_atm) {
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
        m.d4 = d4;
        m.d4_atm = d4_atm;
        return m;
    };

    cpu::gpw::GpwDft method_atm = make_cfg(true, true);
    method_atm.build(atoms);
    const auto F_on = cpu::gpw::gpw_forces_from_scf(atoms, method_atm,
                                                    eval.last_result());
    cpu::gpw::GpwDft method_plain = make_cfg(false, false);
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
            const double want = d4atm.forces[static_cast<std::size_t>(i)]
                                            [static_cast<std::size_t>(k)];
            max_wire = std::max(max_wire, std::abs((on - pl) - want));
            max_f_on = std::max(max_f_on, std::abs(on));
            check(std::isfinite(on) && std::isfinite(pl),
                  "D finite multi-k force component");
        }
    }
    std::printf("D multi-k wiring max|(F_atm-F_plain)-F_standalone|=%.3e  "
                "max|F_on|=%.3e Ha/Bohr\n",
                max_wire, max_f_on);
    check(max_wire <= kWiringTol, "D multi-k wiring vs standalone D4+ATM");
    check(max_f_on > 1e-8, "D product multi-k forces nontrivial");

    // ── Fair wall report (not a hard fail above 10×; schema records ratio) ─
    const double ratio = wall_product_s / kOracleWallS;
    std::printf(
        "PASS: gpw-d4-atm-kpoint-forces  max|F+g|=%.3e  max|dF_atm-cp2k|=%.3e"
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
