/// @file regressions/gpw/gpw-d4-atm-stress/reg_gpw-d4-atm-stress.cpp
/// @brief END-TO-END: periodic water-dimer GPW PBE-D4+ATM analytic cell stress.
///
/// Decisive dedicated pins: dftd4 3.7.0 strained-cell energies (s9=1 and
/// two-body) at the exact ±2e-3 fixed-fractional strains the analytic stress
/// path samples — the ATM stress contribution (~1e-10 Ha/Bohr³) is far below
/// the analytic-vs-FD dual-gate residual (~1e-6), so only these energy pins
/// resolve it. Product dual gate: analytic fixed-P vs multi-SCF strain-FD on
/// the same d4+ATM method. Public API only. See REFERENCE.md.

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
#include "atomiverse/cpu/scf/dftd4.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

// ── Offline dftd4 3.7.0 pins (tools/cp2k_oracle/inputs/gpw-d4-atm/) ──────
// Base cell (12 Å cubic water dimer), s9=1 vs two-body (--mbdscale 0):
constexpr double kEAtmRef = -0.0009955447583424029; // Ha (pin_periodic_atm)
// Strained cells: axis a of the cell AND the Cartesian a-components scaled
// by (1 ± h), h = 2e-3 — the scale_axis fixed-fractional convention of the
// analytic stress path. pin_strain_<ax>_<pm>_{atm,2b}.json. Order per axis:
// {+h, −h}; axes x, y, z.
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

// ── Tolerances (REFERENCE.md) ────────────────────────────────────────────
constexpr double kEPinTol = 1.0e-9;     // Ha, strained-energy parity
constexpr double kSigPinTol = 1.0e-12;  // Ha/Bohr³, disp/ATM stress parity
constexpr double kAtmFloor = 5.0e-10;   // Ha/Bohr³, ATM non-vacuity (pin 7.5e-10)
constexpr double kEDispTol = 1.0e-8;    // Ha, product e_disp vs base ATM pin
constexpr double kAnFdTol = 2.0e-5;     // Ha/Bohr³, analytic vs FD dual gate
constexpr double kOffDiagTol = 1.0e-8;  // Ha/Bohr³
constexpr double kDiagFloor = 1.0e-10;  // Ha/Bohr³
// Product ATM fold wiring: σ_xx(ATM) − σ_xx(2b) vs the standalone σ^ATM_xx.
// The two analytic runs converge their reference densities independently
// (SCF tol 1e-6), which leaves ~7e-11 Ha/Bohr³ of fixed-P Lagrangian noise
// in the difference; the ATM signal is 7.5e-10, so 3e-10 separates cleanly:
// dropping the s9 fold inside the stress path leaves |d| = 7.5e-10 > tol.
constexpr double kWiringTol = 3.0e-10;  // Ha/Bohr³

constexpr double kAng2Bohr = 1.8897259885789;

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    // Periodic water dimer, 12 Å cubic cell (order O H H O H H).
    constexpr double L = 12.0; // Å
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double x[6] = {0.000000, 0.758602, 0.758602,
                         3.000000, 3.758602, 3.758602};
    const double y[6] = {0.000000, 0.000000, 0.000000,
                         0.000000, 0.000000, 0.000000};
    const double zc[6] = {0.000000, 0.504284, -0.504284,
                          0.000000, 0.504284, -0.504284};

    // ── 1+2. Standalone strained-cell dispersion: energy parity + stress ──
    const std::vector<int> Z(z, z + 6);
    std::vector<std::array<double, 3>> pos0(6);
    for (int i = 0; i < 6; ++i)
        pos0[static_cast<std::size_t>(i)] = {x[i] * kAng2Bohr, y[i] * kAng2Bohr,
                                             zc[i] * kAng2Bohr};
    const double cell0[3] = {L * kAng2Bohr, L * kAng2Bohr, L * kAng2Bohr};
    const double vol = cell0[0] * cell0[1] * cell0[2]; // Bohr³ (unstrained)

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
            if (!std::isfinite(e_atv_s9[a][s]) || d1 > kEPinTol
                || d0 > kEPinTol) {
                std::printf("FAIL: strained E_disp parity axis=%d s=%d "
                            "|d_s9|=%.3e |d_2b|=%.3e Ha (tol %g)\n",
                            a, s, d1, d0, kEPinTol);
                return 1;
            }
        }
    }

    // Dispersion cell stress from strained energies (σ_aa = −ΔE/(2hΩ)),
    // ATV vs the same FD over the oracle-printed energies.
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
    if (max_sig_err > kSigPinTol || max_atm_err > kSigPinTol) {
        std::printf("FAIL: dispersion stress parity max|dσ_s9|=%.3e "
                    "max|dσ_ATM|=%.3e > tol %g Ha/Bohr3\n",
                    max_sig_err, max_atm_err, kSigPinTol);
        return 1;
    }
    if (max_atm_pin < kAtmFloor) {
        std::printf("FAIL: ATM stress non-vacuity max|σ^ATM_pin|=%.3e < %g\n",
                    max_atm_pin, kAtmFloor);
        return 1;
    }

    // ── 3. Product: GPW PBE-D4+ATM analytic fixed-P stress ───────────────
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = x[i];
        frame.span(kAtom_y)[i] = y[i];
        frame.span(kAtom_z)[i] = zc[i];
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
    eval.cutoff_ry = 200.0;
    eval.max_iter = 120;
    eval.tol = 1e-6;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = false;
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

    if (!eval.last_result().converged) {
        std::printf("FAIL: GPW SCF did not converge (iters=%d)\n",
                    eval.last_result().iters);
        return 1;
    }
    const double e_disp = eval.last_result().e_disp;
    if (!std::isfinite(e_disp) || std::abs(e_disp - kEAtmRef) > kEDispTol) {
        std::printf("FAIL: |E_disp-ATM pin|=%.6e Ha  got=%.14e  want=%.14e\n",
                    std::abs(e_disp - kEAtmRef), e_disp, kEAtmRef);
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
    const auto& st = frame.meta.get(kStress);
    const double ha_bohr3_to_ev_a3 =
        units::kHartreeInEv
        / (units::kBohrInAngstrom * units::kBohrInAngstrom
           * units::kBohrInAngstrom);
    for (int k = 0; k < 3; ++k) {
        const double expect =
            san[static_cast<std::size_t>(k)] * ha_bohr3_to_ev_a3;
        if (std::abs(st[static_cast<std::size_t>(k)] - expect) > 1e-9) {
            std::printf("FAIL: kStress unit conversion mismatch component %d\n",
                        k);
            return 1;
        }
    }
    for (int a = 0; a < 3; ++a)
        std::printf("  σ_%s analytic(d4+ATM) = % .8e Ha/Bohr3\n", names[a],
                    san[static_cast<std::size_t>(a)]);

    // ── 4+5. FD twin (dual gate) + two-body twin (ATM fold wiring), x-axis,
    //          warm-started from the converged MOs ───────────────────────
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& k = (z[i] == 8) ? O : H;
        const double pos[3] = {x[i] * kAng2Bohr, y[i] * kAng2Bohr,
                               zc[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(pos, k.basis, k.pot, z[i]));
    }
    cpu::gpw::GpwDft method;
    method.cell[0] = method.cell[1] = method.cell[2] = L * kAng2Bohr;
    method.cutoff_ry = 200.0;
    method.n_occ = 8; // 2×O q6 + 4×H q1 = 16 e → 8 doubly occupied
    method.max_iter = 120;
    method.tol = 1e-6;
    method.mix = 0.4;
    method.xc = cpu::gpw::XcKind::PBE;
    method.nonperiodic = false;
    method.local_pp_on_grid = true;
    method.d4 = true;
    method.d4_atm = true;
    method.build(atoms);

    const cpu::linalg::RealMatrix* guess =
        eval.last_result().C.size() > 0 ? &eval.last_result().C : nullptr;

    const auto t2 = clock::now();
    const auto sfd = cpu::gpw::gpw_stress_fd(atoms, method, kStrainH,
                                             {true, false, false}, guess);
    const auto t3 = clock::now();
    const double wall_fd_s = std::chrono::duration<double>(t3 - t2).count();

    const double an_fd_err = std::abs(san[0] - sfd[0]);
    std::printf("  σ_xx analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n", san[0],
                sfd[0], an_fd_err);
    if (!std::isfinite(sfd[0]) || an_fd_err > kAnFdTol) {
        std::printf("FAIL: analytic vs FD σ_xx err %.6e > tol %g Ha/Bohr3\n",
                    an_fd_err, kAnFdTol);
        return 1;
    }

    cpu::gpw::GpwDft method2b;
    method2b.cell[0] = method2b.cell[1] = method2b.cell[2] = L * kAng2Bohr;
    method2b.cutoff_ry = 200.0;
    method2b.n_occ = 8;
    method2b.max_iter = 120;
    method2b.tol = 1e-6;
    method2b.mix = 0.4;
    method2b.xc = cpu::gpw::XcKind::PBE;
    method2b.nonperiodic = false;
    method2b.local_pp_on_grid = true;
    method2b.d4 = true;
    method2b.d4_atm = false; // two-body only
    method2b.build(atoms);

    const auto t4 = clock::now();
    const auto s2b = cpu::gpw::gpw_stress(atoms, method2b, kStrainH,
                                          {true, false, false}, guess);
    const auto t5 = clock::now();
    const double wall_2b_s = std::chrono::duration<double>(t5 - t4).count();

    // Dispersion is additive (never enters the SCF density), so the analytic
    // ATM-minus-two-body stress difference must equal the standalone FD of
    // the ATM energy — proves cfg.d4_atm reaches the strained rebuild.
    const double wiring = (san[0] - s2b[0]) - sig_atm_atv[0];
    std::printf("  wiring: σ_xx(ATM)−σ_xx(2b)=% .6e  standalone σ^ATM_xx="
                "% .6e  |d|=%.3e Ha/Bohr3\n",
                san[0] - s2b[0], sig_atm_atv[0], std::abs(wiring));
    if (!std::isfinite(s2b[0]) || std::abs(wiring) > kWiringTol) {
        std::printf("FAIL: product ATM fold wiring |d|=%.6e > tol %g\n",
                    std::abs(wiring), kWiringTol);
        return 1;
    }

    std::printf("E_atv=%.12f Ha  E_disp=%.14e Ha (ATM pin %.14e)\n",
                eval.last_energy_hartree(), e_disp, kEAtmRef);
    std::printf("wall_product=%.3f s  wall_fd_twin_x=%.3f s  wall_2b_x=%.3f s\n",
                wall_product_s, wall_fd_s, wall_2b_s);
    std::printf("PASS: gpw-d4-atm-stress  max|dE_pin|=%.3e Ha  "
                "max|dσ_ATM|=%.3e  |dσ_an-fd|=%.3e  wiring=%.3e Ha/Bohr3\n",
                max_epin_err, max_atm_err, an_fd_err, std::abs(wiring));
    return 0;
}
