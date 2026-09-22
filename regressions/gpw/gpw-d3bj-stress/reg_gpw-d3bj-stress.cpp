/// @file regressions/gpw/gpw-d3bj-stress/reg_gpw-d3bj-stress.cpp
/// @brief END-TO-END: periodic water-dimer GPW PBE-D3(BJ) analytic cell stress
///        vs an offline CP2K Quickstep NUMERICAL-stress twin.
///
/// Twin of `gpw-d4-stress` (which has no CP2K stress pin — Arrhenius CP2K
/// lacks libdftd4); D3(BJ) is native in CP2K, so this case carries the real
/// oracle stress pin for the dispersion fold. Hard pins from two offline CP2K
/// 2026.1 runs (`STRESS_TENSOR NUMERICAL`, D3BJ + plain):
///   1. `E_disp` — CP2K's printed "Dispersion energy".
///   2. `σ_disp` — σ(D3BJ) − σ(plain), diagonal. D3 is geometry-only, so the
///      SCF density is identical at every strained cell and the electronic
///      stress cancels exactly in the difference; what is left is the
///      numerical strain-derivative of E_disp alone.
/// Decisive dual gate: product analytic fixed-P stress (`d3bj = true`,
/// `stress_use_fd = false`) vs multi-SCF strain-FD (`gpw_stress_fd`) on the
/// same method. The cheap-H₂ unit twin is
/// tests/cpu/gpw/test_cpu_gpw_stress.cpp (`AnalyticMatchesFdWithD3Bj`).
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
// "Dispersion energy" printed by the D3BJ stress run (EPS_SCF 1e-8); equals
// the total-energy difference of the two runs to 2e-16 Ha.
constexpr double kEDispCp2k = -0.00137627769481; // Ha
constexpr double kEDispTol = 1.0e-8;

// σ(D3BJ) − σ(plain), diagonal, from the two `STRESS| Numerical stress
// tensor [bar]` tables converted at 2.9421015697e8 bar per Ha/Bohr³.
// Electronic stress cancels (D3 never enters the KS potential); both runs
// share CP2K's strain-FD scheme, so its truncation cancels too.
constexpr double kSigmaDispCp2k[3] = {-1.1485894562e-07, -8.9319862445e-10,
                                      +9.6563695465e-10}; // Ha/Bohr³
// Floor: CP2K SCF-noise in the FD difference (EPS_SCF 1e-8, ≲1e-11) + C6/CN
// table parity (E_disp parity 7.6e-12 Ha). Measured max residual 3.9e-14
// against the 1.15e-7 xx signal; 1e-10 keeps ~1000x separation from the
// signal while sitting well above the measured floor.
constexpr double kSigmaDispTol = 1.0e-10; // Ha/Bohr³

// Analytic fixed-P vs multi-SCF strain-FD (both d3bj = true). The e2e twin
// re-SCFs the x axis only — the dimer axis, i.e. the component carrying both
// the dispersion-stress signal (1.15e-7) and the largest measured residual
// (pinning ran all three: xx 7.39e-7, yy 2.46e-8, zz 1.24e-9 Ha/Bohr³) —
// to keep the wall inside the 60 s budget; the all-axes identity lives in
// tests/cpu/gpw/test_cpu_gpw_stress.cpp (AnalyticMatchesFdWithD3Bj).
constexpr double kAnFdTol = 2.0e-5;    // Ha/Bohr³ (measured 7.4e-7)
constexpr double kOffDiagTol = 1.0e-8; // Ha/Bohr³
constexpr double kDiagFloor = 1.0e-10; // Ha/Bohr³

// D3(BJ) PBE parameters — CP2K dftd3.dat values, CP2K CN form, ATM off.
constexpr double kS6 = 1.0, kS8 = 0.7875, kA1 = 0.4289, kA2 = 4.4407;

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    // Periodic water dimer, 12 Å cubic cell — verbatim the shipped
    // gpw-d3bj-forces twin geometry (REFERENCE.md).
    constexpr double L = 12.0; // Å
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

    // ── (1) Standalone D3(BJ): E_disp vs the CP2K printed pin ─────────────
    const cpu::gpw::D3Result d3 = cpu::gpw::dftd3_pbe_bj(
        Zv, pos_bohr, cell_b, /*periodic=*/true, cpu::gpw::D3Params{},
        cpu::gpw::kD3Cp2kPairCutoffBohr, kS6, kS8, kA1, kA2,
        /*cp2k_cn=*/true, /*calculate_c9=*/false);
    if (std::abs(d3.energy - kEDispCp2k) > kEDispTol) {
        std::printf("FAIL: |E_disp-pin|=%.3e Ha  got=%.14e  want=%.14e\n",
                    std::abs(d3.energy - kEDispCp2k), d3.energy, kEDispCp2k);
        return 1;
    }

    // ── (2) Dispersion stress vs the CP2K NUMERICAL-stress difference ─────
    // Central strain-FD of the geometry-only E_disp at fixed fractional
    // coordinates (scaled positions + cell) — the identical convention the
    // analytic fixed-P fold applies inside `gpw_stress`.
    // σ_aa = −Ω⁻¹ ∂E_disp/∂ε_aa (CP2K STRESS_TENSOR sign).
    const double vol = Lb * Lb * Lb; // Bohr³
    constexpr double hd = 1.0e-4;    // pure function of geometry — tiny h OK
    double sigma_disp[3];
    for (int a = 0; a < 3; ++a) {
        double e_pm[2];
        for (int sgn = 0; sgn < 2; ++sgn) {
            const double eps = (sgn == 0 ? +hd : -hd);
            double cell_s[3] = {cell_b[0], cell_b[1], cell_b[2]};
            cell_s[a] *= (1.0 + eps);
            std::vector<std::array<double, 3>> pos_s = pos_bohr;
            for (auto& p : pos_s)
                p[static_cast<std::size_t>(a)] *= (1.0 + eps);
            e_pm[sgn] =
                cpu::gpw::dftd3_pbe_bj(
                    Zv, pos_s, cell_s, /*periodic=*/true, cpu::gpw::D3Params{},
                    cpu::gpw::kD3Cp2kPairCutoffBohr, kS6, kS8, kA1, kA2,
                    /*cp2k_cn=*/true, /*calculate_c9=*/false)
                    .energy;
        }
        sigma_disp[a] = -(e_pm[0] - e_pm[1]) / (2.0 * hd) / vol;
    }
    double max_pin_err = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(sigma_disp[a] - kSigmaDispCp2k[a]);
        max_pin_err = std::max(max_pin_err, err);
        std::printf("  σ_disp_%s  atv=% .8e  cp2k=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[a], sigma_disp[a], kSigmaDispCp2k[a], err);
    }
    if (max_pin_err > kSigmaDispTol) {
        std::printf("FAIL: dispersion stress vs CP2K diff %.3e > tol %g\n",
                    max_pin_err, kSigmaDispTol);
        return 1;
    }

    // ── (3) Product analytic fixed-P stress with d3bj = true ──────────────
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
    eval.cutoff_ry = 200.0;
    eval.max_iter = 120;
    eval.tol = 1e-6;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = false;
    eval.d3bj = true;
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
    if (std::abs(e_disp_prod - kEDispCp2k) > kEDispTol) {
        std::printf("FAIL: product |E_disp-pin|=%.3e Ha  got=%.14e\n",
                    std::abs(e_disp_prod - kEDispCp2k), e_disp_prod);
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
    method.cutoff_ry = 200.0;
    method.n_occ = 8; // 2×O q6 + 4×H q1 = 16 e → 8 doubly occupied
    method.max_iter = 120;
    method.tol = 1e-6;
    method.mix = 0.4;
    method.xc = cpu::gpw::XcKind::PBE;
    method.nonperiodic = false;
    method.local_pp_on_grid = true;
    method.d3bj = true;
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
                eval.last_energy_hartree(), e_disp_prod, kEDispCp2k);
    std::printf("wall_product=%.3f s  wall_fd_twin=%.3f s  iters=%d\n",
                wall_product_s, wall_fd_s, eval.last_result().iters);
    std::printf(
        "PASS: gpw-d3bj-stress  max|σ_disp-cp2k|=%.3e  |dσ_an-fd|_xx=%.3e  "
        "|dE_disp|=%.3e\n",
        max_pin_err, anfd_err, std::abs(e_disp_prod - kEDispCp2k));
    return 0;
}
