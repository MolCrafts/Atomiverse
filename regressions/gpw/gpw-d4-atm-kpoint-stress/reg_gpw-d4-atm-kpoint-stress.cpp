/// @file regressions/gpw/gpw-d4-atm-kpoint-stress/reg_gpw-d4-atm-kpoint-stress.cpp
/// @brief END-TO-END: multi-k GPW PBE-D4+ATM analytic cell stress vs offline
///        CP2K Quickstep NUMERICAL-stress twins (+ dftd4 strained-cell fidelity).
///
/// Capability: the k-summed fixed-\(P_k\) stress Lagrangian rebuilds the
/// geometry-only \(E_{\mathrm{disp}}^{\mathrm{D4+ATM}}\) (\(s_9 = 1\)) at every
/// strained cell (`energy_fixed_P_k` in `gpw_stress.cpp`). Twin of
/// `gpw-d4-atm-stress` (Γ) on the same 12 Å water dimer with electronic mesh
/// 1×1×2; deps closed (`gpw-d4-atm-stress`, `gpw-d4-kpoint-stress`).
///
/// Hard pins:
///   1. `E_disp` / strained-cell energies — offline dftd4 3.7.0 s9=1 and two-body
///      (geometry-only; bit-identical to the Γ pin; k-mesh independent).
///   2. `σ_ATM` — σ(s9=1) − σ(s9=0) from two offline CP2K 2026.1 multi-k
///      `STRESS_TENSOR NUMERICAL` runs (`libdftd4` build).
/// Decisive dual gate: product multi-k analytic fixed-\(P_k\) stress
/// (`d4 = d4_atm = true`, `k_mesh = {1,1,2}`, `stress_use_fd = false`) vs
/// multi-SCF strain-FD (`gpw_stress_fd`) on the same method (x-axis e2e;
/// all-axes unit twin `D4AtmMultiKAnalyticMatchesFd`).
/// Fair wall: product path vs the CP2K ATM multi-k NUMERICAL-stress twin.
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
#include "atomiverse/cpu/gpw/internal/gpw_stress.h"
#include "atomiverse/cpu/scf/dftd4.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

// Same Å→Bohr as dftd4 3.7.0 pin scripts (gpw-d4-water / pin_d4_strain.py).
constexpr double kAng2Bohr = 1.0 / 0.52917720859;

// ── Offline dftd4 3.7.0 pins (tools/cp2k_oracle/inputs/gpw-d4-atm/) ──────
// Base cell (12 Å cubic water dimer), s9=1 — shared with gpw-d4-atm.
constexpr double kEAtmRef = -0.0009955447583424029; // Ha
// Strained cells: axis a of the cell AND Cartesian a-components scaled by
// (1 ± h), h = 2e-3 — the scale_axis fixed-fractional convention.
// pin_strain_<ax>_<pm>_{atm,2b}.json. Order per axis: {+h, −h}.
constexpr double kStrainH = 2e-3;
constexpr double kEs9[3][2] = {
    {-0.0009937576764997651, -0.0009973321246999301}, // x ± h
    {-0.0009955375822020631, -0.0009955520094036394}, // y ± h
    {-0.0009955898926559920, -0.0009954965921409587}, // z ± h
};
constexpr double kE2b[3][2] = {
    {-0.0009928187113161498, -0.0009963582970691216}, // x ± h
    {-0.0009945676376042270, -0.0009946094592497990}, // y ± h
    {-0.0009946392573410504, -0.0009945345483101898}, // z ± h
};

// ── Offline CP2K 2026.1 multi-k pins (libdftd4; REFERENCE.md) ────────────
// σ(s9=1) − σ(s9=0), diagonal, from two multi-k
// `STRESS| Numerical stress tensor [bar]` tables converted at
// 2.9421015697e8 bar per Ha/Bohr³ (CODATA 2018).
// Fresh pair 2026-08-01: ATM out − 2b out (both multi-k NUMERICAL, same mesh).
constexpr double kSigmaAtmCp2k[3] = {-7.676250620509834e-10,
                                     +6.017319919313666e-10,
                                     -2.512853422932131e-10}; // Ha/Bohr³
// Residual floor: CP2K↔dftd4 cutoff/CN parity (~2e-11 measured) + print noise.
constexpr double kSigmaAtmTol = 5.0e-11; // Ha/Bohr³
constexpr double kSigmaAtmMin = 5.0e-10; // Ha/Bohr³ engage floor (pin ~7.7e-10)

// Tolerances (REFERENCE.md)
constexpr double kEPinTol = 1.0e-9;     // Ha, strained-energy parity
constexpr double kSigPinTol = 1.0e-12;  // Ha/Bohr³, disp/ATM stress from dftd4 E
constexpr double kEDispTol = 1.0e-8;    // Ha, product e_disp vs base ATM pin
constexpr double kAnFdTol = 2.0e-5;     // Ha/Bohr³, analytic vs FD dual gate
constexpr double kOffDiagTol = 1.0e-12; // Ha/Bohr³
constexpr double kDiagFloor = 1.0e-10;  // Ha/Bohr³
// Product ATM fold wiring: σ_xx(ATM) − σ_xx(2b) vs standalone σ^ATM_xx.
// Independent SCF tol 1e-6 leaves ~1e-10 noise; ATM signal ~7.5e-10.
constexpr double kWiringTol = 3.0e-10; // Ha/Bohr³
constexpr double kWallBudget = 60.0;   // s product path
// Fair wall: CP2K internal timer on ATM multi-k NUMERICAL-stress twin
// (1 MPI × 8 OMP, 2026-08-01, private libdftd4 rebuild git:5e54ba2).
constexpr double kOracleWallS = 137.945;

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

    // Periodic water dimer, 12 Å cubic cell — same as gpw-d4-atm-stress /
    // gpw-d4-kpoint-stress.
    constexpr double L = 12.0; // Å
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double xs[6] = {0.000000, 0.758602, 0.758602,
                          3.000000, 3.758602, 3.758602};
    const double ys[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double zs[6] = {0.000000, 0.504284, -0.504284,
                          0.000000, 0.504284, -0.504284};

    // ── (1) Standalone strained-cell dispersion: energy parity + σ_ATM ───
    const std::vector<int> Z(z, z + 6);
    std::vector<std::array<double, 3>> pos0(6);
    for (int i = 0; i < 6; ++i)
        pos0[static_cast<std::size_t>(i)] = {
            xs[i] * kAng2Bohr, ys[i] * kAng2Bohr, zs[i] * kAng2Bohr};
    const double cell0[3] = {L * kAng2Bohr, L * kAng2Bohr, L * kAng2Bohr};
    const double vol = cell0[0] * cell0[1] * cell0[2]; // Bohr³

    double e_atv_s9[3][2];
    double e_atv_2b[3][2];
    double max_epin_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        for (int s = 0; s < 2; ++s) {
            const double scale = 1.0 + (s == 0 ? kStrainH : -kStrainH);
            auto pos = pos0;
            for (auto& p : pos)
                p[static_cast<std::size_t>(a)] *= scale;
            double cell[3] = {cell0[0], cell0[1], cell0[2]};
            cell[a] *= scale;
            e_atv_s9[a][s] = cpu::scf::dftd4_energy(
                Z, pos, cell, /*periodic=*/true, /*s6=*/1.0,
                /*s8=*/0.95948085, /*a1=*/0.38574991, /*a2=*/4.80688534,
                /*ga=*/3.0, /*gc=*/2.0, /*wf=*/6.0, /*charge=*/0.0,
                /*cn_cut=*/30.0, /*disp2_cut=*/60.0, /*s9=*/1.0);
            e_atv_2b[a][s] =
                cpu::scf::dftd4_energy(Z, pos, cell, /*periodic=*/true);
            const double d1 = std::abs(e_atv_s9[a][s] - kEs9[a][s]);
            const double d0 = std::abs(e_atv_2b[a][s] - kE2b[a][s]);
            max_epin_err = std::max({max_epin_err, d1, d0});
            if (!std::isfinite(e_atv_s9[a][s]) || d1 > kEPinTol || d0 > kEPinTol) {
                std::printf("FAIL: strained E_disp parity axis=%d s=%d "
                            "|d_s9|=%.3e |d_2b|=%.3e Ha (tol %g)\n",
                            a, s, d1, d0, kEPinTol);
                return 1;
            }
        }
    }
    std::printf("strained E_disp max|d pin|=%.3e Ha (12 values)\n",
                max_epin_err);

    double sig_atm_atv[3];
    double max_sig_err = 0.0, max_atm_err = 0.0, max_atm_pin = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int a = 0; a < 3; ++a) {
        const double denom = 2.0 * kStrainH * vol;
        const double s1_atv = -(e_atv_s9[a][0] - e_atv_s9[a][1]) / denom;
        const double s1_pin = -(kEs9[a][0] - kEs9[a][1]) / denom;
        const double s0_atv = -(e_atv_2b[a][0] - e_atv_2b[a][1]) / denom;
        const double s0_pin = -(kE2b[a][0] - kE2b[a][1]) / denom;
        const double atm_atv = s1_atv - s0_atv;
        const double atm_pin = s1_pin - s0_pin;
        sig_atm_atv[a] = atm_atv;
        const double e1 = std::abs(s1_atv - s1_pin);
        const double ea = std::abs(atm_atv - atm_pin);
        max_sig_err = std::max(max_sig_err, e1);
        max_atm_err = std::max(max_atm_err, ea);
        max_atm_pin = std::max(max_atm_pin, std::abs(atm_pin));
        std::printf("  σ_%s^disp(s9=1) atv=% .10e pin=% .10e |d|=%.2e ; "
                    "σ^ATM atv=% .6e pin=% .6e |d|=%.2e Ha/Bohr3\n",
                    names[a], s1_atv, s1_pin, e1, atm_atv, atm_pin, ea);
    }
    check(max_sig_err <= kSigPinTol && max_atm_err <= kSigPinTol,
          "A dftd4 strained-cell σ_disp / σ_ATM parity");
    check(max_atm_pin >= kSigmaAtmMin, "A ATM stress non-vacuous vs pin floor");

    // Standalone σ_ATM vs multi-k CP2K NUMERICAL difference.
    double max_cp2k_atm = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double d = std::abs(sig_atm_atv[a] - kSigmaAtmCp2k[a]);
        max_cp2k_atm = std::max(max_cp2k_atm, d);
        std::printf("  σ_ATM_%s  atv=% .8e  cp2k=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[a], sig_atm_atv[a], kSigmaAtmCp2k[a], d);
    }
    check(max_cp2k_atm <= kSigmaAtmTol,
          "B standalone σ_ATM equals multi-k CP2K Δσ pin");

    // ── (2) Product multi-k GPW evaluator with D4+ATM, analytic stress ───
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = xs[i];
        frame.span(kAtom_y)[i] = ys[i];
        frame.span(kAtom_z)[i] = zs[i];
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
    eval.compute_forces = false;
    eval.compute_stress = true;
    eval.stress_use_fd = false;
    eval.stress_strain_h = kStrainH;

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
    std::printf("C E_disp(multi-k ATM) = %.14e Ha (dftd4 pin %.14e, |d|=%.2e)\n",
                e_disp, kEAtmRef, std::abs(e_disp - kEAtmRef));
    check(std::isfinite(e_disp) && std::abs(e_disp - kEAtmRef) <= kEDispTol,
          "C E_disp equals the dftd4 ATM pin (k-mesh independent)");

    const auto& san = eval.last_stress_ha_bohr3();
    for (int k = 3; k < 6; ++k)
        check(std::abs(san[static_cast<std::size_t>(k)]) <= kOffDiagTol,
              "C orthorhombic off-diagonal exactly 0");
    const double max_diag =
        std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
    check(max_diag > kDiagFloor, "C stress diagonal nonzero");

    // Frame kStress unit conversion (eV/Å³).
    // kStress is the extensive Voigt virial [eV]: the method's Ha/Bohr^3
    // return stays intensive, so the published value is sigma * V.
    const double kStressVolA3 = frame.box.volume();
    const auto& st = frame.meta.get(kStress);
    const double ha_bohr3_to_ev_a3 =
        units::kHartreeInEv
        / (units::kBohrInAngstrom * units::kBohrInAngstrom
           * units::kBohrInAngstrom);
    bool conv_ok = true;
    for (int k = 0; k < 3; ++k) {
        const double expect =
            san[static_cast<std::size_t>(k)] * ha_bohr3_to_ev_a3 * kStressVolA3;
        if (std::abs(st[static_cast<std::size_t>(k)] - expect) > 1e-9 * kStressVolA3)
            conv_ok = false;
    }
    check(conv_ok, "C kStress unit conversion");

    // ── (3) Dual gate: multi-k analytic vs multi-SCF strain-FD (x only) ──
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& k = (z[i] == 8) ? O : H;
        const double p[3] = {xs[i] * kAng2Bohr, ys[i] * kAng2Bohr,
                             zs[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(p, k.basis, k.pot, z[i]));
    }

    auto make_cfg = [&](bool atm) {
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
        m.d4 = true;
        m.d4_atm = atm;
        return m;
    };

    const auto t2 = clock::now();
    const auto sfd = cpu::gpw::internal::gpw_stress_fd_impl(
        atoms, make_cfg(true), kStrainH, /*axes*/ {true, false, false});
    const auto t3 = clock::now();
    const double wall_fd_s = std::chrono::duration<double>(t3 - t2).count();

    const double an_xx = san[0];
    const double fd_xx = sfd[0];
    const double anfd_err = std::abs(an_xx - fd_xx);
    std::printf("  σ_xx  analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n", an_xx,
                fd_xx, anfd_err);
    std::printf("  σ_yy  analytic=% .8e   σ_zz  analytic=% .8e\n", san[1],
                san[2]);
    check(std::isfinite(an_xx) && std::isfinite(fd_xx),
          "D finite stress components");
    check(anfd_err <= kAnFdTol, "D multi-k analytic vs multi-SCF FD (xx)");

    // ── (4) Wiring: product σ(ATM) − σ(2b) vs standalone σ_ATM ───────────
    const auto s_2b = cpu::gpw::internal::gpw_stress_analytic_impl(
        atoms, make_cfg(false), kStrainH, /*axes*/ {true, false, false});
    const double wire_xx = san[0] - s_2b[0];
    const double wire_err = std::abs(wire_xx - sig_atm_atv[0]);
    std::printf("  wiring σ_xx(ATM)-σ_xx(2b)=% .6e  standalone_ATM=% .6e  "
                "|d|=%.3e\n",
                wire_xx, sig_atm_atv[0], wire_err);
    check(wire_err <= kWiringTol,
          "E product ATM fold wiring (s9 in fixed-P_k rebuild)");

    // Product wiring also vs CP2K multi-k Δσ (xx).
    const double wire_cp2k = std::abs(wire_xx - kSigmaAtmCp2k[0]);
    std::printf("  wiring vs CP2K Δσ_xx |d|=%.3e\n", wire_cp2k);
    check(wire_cp2k <= kSigmaAtmTol + kWiringTol,
          "E product Δσ_xx matches multi-k CP2K ATM pin (soft)");

    check(wall_product_s <= kWallBudget, "product wall <= 60 s");
    const double ratio = wall_product_s / kOracleWallS;
    std::printf("fair_wall ratio T_atv/T_cp2k = %.3f  (cp2k=%.3f s  "
                "product=%.3f s  fd_twin=%.3f s)\n",
                ratio, kOracleWallS, wall_product_s, wall_fd_s);
    check(ratio <= 10.0, "fair wall ratio T_atv/T_cp2k <= 10");

    std::printf(
        "gpw-d4-atm-kpoint-stress  |dE_disp|=%.2e Ha  max|σ_ATM-cp2k|=%.3e  "
        "|dσ_an-fd|_xx=%.3e  |wire|=%.3e  iters=%d  nk=%d  "
        "wall_product=%.2f s  ratio=%.3f\n",
        std::abs(e_disp - kEAtmRef), max_cp2k_atm, anfd_err, wire_err,
        eval.last_result().iters, eval.last_result().n_kpts, wall_product_s,
        ratio);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-d4-atm-kpoint-stress\n");
    return 0;
}
