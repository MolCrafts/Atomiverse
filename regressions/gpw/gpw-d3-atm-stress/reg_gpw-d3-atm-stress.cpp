/// @file regressions/gpw/gpw-d3-atm-stress/reg_gpw-d3-atm-stress.cpp
/// @brief END-TO-END: periodic water-dimer GPW PBE-D3 cell stress with the
///        three-body Axilrod–Teller–Muto (C9) term folded, vs an offline CP2K
///        Quickstep NUMERICAL-stress twin pair.
///
/// Twin of the shipped `gpw-d3bj-stress` (two-body BJ stress pin) on the
/// `gpw-d3-atm-forces` periodic 6 Å geometry. Hard pins from two offline CP2K
/// 2026.1 runs (`STRESS_TENSOR NUMERICAL`, `CALCULATE_C9_TERM` T/F, otherwise
/// identical):
///   1. `E_disp` — CP2K's printed "Dispersion energy", ATM on and off.
///   2. `σ_ATM` — σ(C9) − σ(two-body), diagonal. D3 (any damping, ± C9) is
///      geometry-only, so the SCF density AND the two-body dispersion are
///      identical at every perturbed cell; the difference isolates the pure
///      three-body ATM stress −Ω⁻¹ ∂E_ATM/∂ε.
/// Decisive dual gate: product analytic fixed-P stress (`d3 = d3_atm = true`,
/// `stress_use_fd = false`) vs multi-SCF strain-FD (`gpw_stress_fd`) on the
/// same method. The cheap periodic-H₂ unit twin is
/// tests/cpu/gpw/test_cpu_gpw_stress.cpp (`AnalyticMatchesFdWithD3Atm`).
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
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/dftd3_params.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

constexpr double kAng2Bohr = 1.8897259885789;

// ── Offline CP2K 2026.1 pins (see REFERENCE.md) ────────────────────────────
// Final "Dispersion energy" printed by each NUMERICAL-stress run (per 6 Å
// cell, R_CUTOFF 10.0, EPS_SCF 1e-8); reproduces the gpw-d3-atm-forces
// energy pins exactly, and the total-energy difference of the two runs
// reproduces E_disp(C9) − E_disp(2b) to 4.7e-15 Ha.
constexpr double kEDispAtmCp2k = -0.00195809486938; // Ha (CALCULATE_C9 .TRUE.)
constexpr double kEDispTwoCp2k = -0.00195887984277; // Ha (.FALSE.)
constexpr double kEDispTol = 1.0e-9;

// σ(C9) − σ(two-body), diagonal, from the two `STRESS| Numerical stress
// tensor [bar]` tables converted at 2.9421015697e8 bar per Ha/Bohr³.
// Electronic and two-body dispersion stress cancel (geometry-only, identical
// SCF path); what remains is CP2K's numerical strain-derivative of E_ATM.
constexpr double kSigmaAtmCp2k[3] = {-1.7148335571e-08, +4.3332828109e-08,
                                     +9.0917867267e-09}; // Ha/Bohr³
// Floor: CP2K stress print/solver noise (~1e-5 bar off-diagonal residuals
// ≈ 4e-14 Ha/Bohr³) + the residual FD mismatch at the shared step (the ATM
// hard gates make the sum step-sensitive — see the hd comment below).
// Measured max residual 1.13e-13 (xx; yy 1.1e-14, zz 3.7e-14) against the
// 9.1e-9..4.3e-8 signals; 5e-12 sits ~40x above the measured residual and
// keeps ~1800x separation from the smallest signal.
constexpr double kSigmaAtmTol = 5.0e-12; // Ha/Bohr³

// ATM engage floor: the C9 fold must actually move the stress.
constexpr double kSigmaAtmMin = 1.0e-9; // Ha/Bohr³

// Analytic fixed-P vs multi-SCF strain-FD (both d3 + d3_atm). The e2e twin
// re-SCFs the x axis only to hold the 60 s budget; the all-axes identity is
// the cheap periodic-H₂ unit twin (AnalyticMatchesFdWithD3Atm). Gate at the
// gpw-d3bj-stress scale (SCF-convergence noise at tol 1e-6, not the fold).
constexpr double kAnFdTol = 2.0e-5;    // Ha/Bohr³
constexpr double kOffDiagTol = 1.0e-8; // Ha/Bohr³
constexpr double kDiagFloor = 1.0e-10; // Ha/Bohr³

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    // Periodic water dimer, 6 Å cubic cell — verbatim the merged
    // gpw-d3-atm-forces periodic twin geometry (REFERENCE.md).
    constexpr double L = 6.0; // Å
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double xs[6] = {0.000000, 0.758602, 0.758602,
                          3.000000, 3.758602, 3.758602};
    const double ys[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double zs[6] = {0.000000, 0.504284, -0.504284,
                          0.000000, 0.504284, -0.504284};

    std::vector<int> Zv(6);
    std::vector<std::array<double, 3>> pos_bohr(6);
    for (int i = 0; i < 6; ++i) {
        Zv[static_cast<std::size_t>(i)] = z[i];
        pos_bohr[static_cast<std::size_t>(i)] = {
            xs[i] * kAng2Bohr, ys[i] * kAng2Bohr, zs[i] * kAng2Bohr};
    }
    const double Lb = L * kAng2Bohr;
    const double cell_b[3] = {Lb, Lb, Lb};
    // CP2K R_CUTOFF 10.0 Å is a half cutoff: pairs/triples sum to 2×.
    const double rcut = 2.0 * 10.0 * kAng2Bohr; // Bohr

    // ── (1) Standalone D3 (zero damping): E_disp pins, C9 on and off ──────
    const cpu::gpw::D3Result d3_on = cpu::gpw::dftd3_pbe(
        Zv, pos_bohr, cell_b, /*periodic=*/true, cpu::gpw::D3Params{}, rcut,
        /*calculate_c9=*/true);
    const cpu::gpw::D3Result d3_off = cpu::gpw::dftd3_pbe(
        Zv, pos_bohr, cell_b, /*periodic=*/true, cpu::gpw::D3Params{}, rcut,
        /*calculate_c9=*/false);
    if (std::abs(d3_on.energy - kEDispAtmCp2k) > kEDispTol
        || std::abs(d3_off.energy - kEDispTwoCp2k) > kEDispTol) {
        std::printf("FAIL: E_disp pins  ATM got=%.14e want=%.14e  two-body "
                    "got=%.14e want=%.14e (tol %.1e Ha)\n",
                    d3_on.energy, kEDispAtmCp2k, d3_off.energy, kEDispTwoCp2k,
                    kEDispTol);
        return 1;
    }
    std::printf("E_disp  ATM=%.14e (|d|=%.2e)  two-body=%.14e (|d|=%.2e) Ha\n",
                d3_on.energy, std::abs(d3_on.energy - kEDispAtmCp2k),
                d3_off.energy, std::abs(d3_off.energy - kEDispTwoCp2k));

    // ── (2) ATM stress vs the CP2K NUMERICAL-stress difference ────────────
    // Central strain-FD of the geometry-only E_ATM = E(C9) − E(2b) at fixed
    // fractional coordinates (scaled positions + cell) — the identical
    // convention the analytic fixed-P fold applies inside `gpw_stress`.
    // σ_aa = −Ω⁻¹ ∂E_ATM/∂ε_aa (CP2K STRESS_TENSOR sign).
    const double vol = Lb * Lb * Lb; // Bohr³
    // Strain step = EXACTLY CP2K's numerical-stress cell perturbation
    // (&NUMERICAL_STRESS default DX 0.001 Bohr on each h component →
    // ε = DX/L). Unlike the smooth-cutoff two-body sum (gpw-d3bj-stress,
    // step-independent to 1e-13), the ATM triple enumeration passes HARD
    // r < r_cut and empirical-scaling gates: the included boundary set jumps
    // by ~5e-11 Ha at discrete strains, so an FD window that differs from the
    // oracle's straddles different gate crossings (h = 1e-4 was off by
    // 1.6e-10 on xx/zz; the CP2K step reproduces the pin to ~1e-12).
    const double hd = 0.001 / Lb; // ε per axis (cubic cell: same for all)
    double sigma_atm[3];
    for (int a = 0; a < 3; ++a) {
        double e_pm[2];
        for (int sgn = 0; sgn < 2; ++sgn) {
            const double eps = (sgn == 0 ? +hd : -hd);
            double cell_s[3] = {cell_b[0], cell_b[1], cell_b[2]};
            cell_s[a] *= (1.0 + eps);
            std::vector<std::array<double, 3>> pos_s = pos_bohr;
            for (auto& p : pos_s)
                p[static_cast<std::size_t>(a)] *= (1.0 + eps);
            const double e_on =
                cpu::gpw::dftd3_pbe(Zv, pos_s, cell_s, /*periodic=*/true,
                                    cpu::gpw::D3Params{}, rcut,
                                    /*calculate_c9=*/true)
                    .energy;
            const double e_off =
                cpu::gpw::dftd3_pbe(Zv, pos_s, cell_s, /*periodic=*/true,
                                    cpu::gpw::D3Params{}, rcut,
                                    /*calculate_c9=*/false)
                    .energy;
            e_pm[sgn] = e_on - e_off;
        }
        sigma_atm[a] = -(e_pm[0] - e_pm[1]) / (2.0 * hd) / vol;
    }
    double max_pin_err = 0.0;
    double max_signal = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(sigma_atm[a] - kSigmaAtmCp2k[a]);
        max_pin_err = std::max(max_pin_err, err);
        max_signal = std::max(max_signal, std::abs(kSigmaAtmCp2k[a]));
        std::printf("  σ_ATM_%s  atv=% .8e  cp2k=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[a], sigma_atm[a], kSigmaAtmCp2k[a], err);
    }
    if (max_pin_err > kSigmaAtmTol) {
        std::printf("FAIL: ATM stress vs CP2K diff %.3e > tol %g\n",
                    max_pin_err, kSigmaAtmTol);
        return 1;
    }
    if (max_signal < kSigmaAtmMin) {
        std::printf("FAIL: ATM stress signal %.3e below engage floor %.1e\n",
                    max_signal, kSigmaAtmMin);
        return 1;
    }

    // ── (3) Product analytic fixed-P stress with d3 + d3_atm ──────────────
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
    eval.cutoff_ry = 150.0;
    eval.max_iter = 120;
    eval.tol = 1e-6;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = false;
    eval.d3 = true;
    eval.d3_atm = true;
    eval.d3_rcut_bohr = rcut; // CP2K R_CUTOFF 10.0 twin setting
    eval.compute_forces = false;
    eval.compute_stress = true;
    eval.stress_use_fd = false;
    eval.stress_strain_h = 2e-3;

    const auto t0 = clock::now();
    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);
    const auto t1 = clock::now();
    const double wall_product_s =
        std::chrono::duration<double>(t1 - t0).count();

    if (!eval.last_result().converged) {
        std::printf("FAIL: GPW SCF did not converge (iters=%d)\n",
                    eval.last_result().iters);
        return 1;
    }
    const double e_disp_prod = eval.last_result().e_disp;
    if (std::abs(e_disp_prod - kEDispAtmCp2k) > kEDispTol) {
        std::printf("FAIL: product |E_disp-pin|=%.3e Ha  got=%.14e\n",
                    std::abs(e_disp_prod - kEDispAtmCp2k), e_disp_prod);
        return 1;
    }

    const auto& san = eval.last_stress_ha_bohr3();
    for (int k = 3; k < 6; ++k) {
        if (std::abs(san[static_cast<std::size_t>(k)]) > kOffDiagTol) {
            std::printf("FAIL: analytic off-diag σ[%d]=%.3e (expect ~0)\n", k,
                        san[static_cast<std::size_t>(k)]);
            return 1;
        }
    }
    const double max_diag =
        std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
    if (max_diag < kDiagFloor) {
        std::printf("FAIL: max |σ_diag|=%.3e below floor %.1e\n", max_diag,
                    kDiagFloor);
        return 1;
    }

    // Frame kStress unit conversion check (eV/Å³).
    // kStress is the extensive Voigt virial [eV]: the method's Ha/Bohr^3
    // return stays intensive, so the published value is sigma * V.
    const double kStressVolA3 = frame.box.volume();
    const auto& st = frame.meta.get(kStress);
    const double ha_bohr3_to_ev_a3 =
        units::kHartreeInEv
        / (units::kBohrInAngstrom * units::kBohrInAngstrom
           * units::kBohrInAngstrom);
    for (int k = 0; k < 3; ++k) {
        const double expect =
            san[static_cast<std::size_t>(k)] * ha_bohr3_to_ev_a3 * kStressVolA3;
        if (std::abs(st[static_cast<std::size_t>(k)] - expect)
            > 1e-9 * kStressVolA3) {
            std::printf("FAIL: kStress unit conversion mismatch component %d\n",
                        k);
            return 1;
        }
    }

    // ── (4) Decisive dual gate: analytic vs multi-SCF strain-FD ───────────
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& k = (z[i] == 8) ? O : H;
        const double pos[3] = {xs[i] * kAng2Bohr, ys[i] * kAng2Bohr,
                               zs[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(pos, k.basis, k.pot, z[i]));
    }
    cpu::gpw::GpwDft method;
    method.cell[0] = method.cell[1] = method.cell[2] = Lb;
    method.cutoff_ry = 150.0;
    method.n_occ = 8; // 2×O q6 + 4×H q1 = 16 e → 8 doubly occupied
    method.max_iter = 120;
    method.tol = 1e-6;
    method.mix = 0.4;
    method.xc = cpu::gpw::XcKind::PBE;
    method.nonperiodic = false;
    method.local_pp_on_grid = true;
    method.d3 = true;
    method.d3_atm = true;
    method.d3_rcut_bohr = rcut;
    method.build(atoms);

    const auto t2 = clock::now();
    const auto sfd = cpu::gpw::gpw_stress_fd(
        atoms, method, /*h=*/2e-3, /*axes*/ {true, false, false},
        eval.last_result().C.size() > 0 ? &eval.last_result().C : nullptr);
    const auto t3 = clock::now();
    const double wall_fd_s = std::chrono::duration<double>(t3 - t2).count();

    const double an_xx = san[0];
    const double fd_xx = sfd[0];
    const double anfd_err = std::abs(an_xx - fd_xx);
    std::printf("  σ_xx  analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n", an_xx,
                fd_xx, anfd_err);
    std::printf("  σ_yy  analytic=% .8e   σ_zz  analytic=% .8e\n", san[1],
                san[2]);
    if (!std::isfinite(an_xx) || !std::isfinite(fd_xx)) {
        std::printf("FAIL: non-finite stress component xx\n");
        return 1;
    }
    if (anfd_err > kAnFdTol) {
        std::printf("FAIL: analytic vs FD σ_xx err %.6e > tol %g Ha/Bohr3\n",
                    anfd_err, kAnFdTol);
        return 1;
    }

    std::printf("E_atv=%.12f Ha  E_disp=%.14e Ha (pin %.14e)\n",
                eval.last_energy_hartree(), e_disp_prod, kEDispAtmCp2k);
    std::printf("wall_product=%.3f s  wall_fd_twin=%.3f s  iters=%d\n",
                wall_product_s, wall_fd_s, eval.last_result().iters);
    std::printf(
        "PASS: gpw-d3-atm-stress  max|σ_ATM-cp2k|=%.3e  |dσ_an-fd|_xx=%.3e  "
        "|dE_disp|=%.3e\n",
        max_pin_err, anfd_err, std::abs(e_disp_prod - kEDispAtmCp2k));
    return 0;
}
