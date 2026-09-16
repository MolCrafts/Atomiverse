/// @file regressions/gpw/gpw-fd-phonon-si/reg_gpw-fd-phonon-si.cpp
/// @brief END-TO-END: finite-displacement vibrational spectrum from GPW
///        analytic forces — Si₂ UKS triplet (isolated, centred MT cell).
///
/// Product residual: `cpu::vib::Phonon` + `GPWEvaluator{compute_forces=true}`
/// without a NeighborList. Scientific dual-gate is the **1D bond-stretch
/// energy curvature** at the same electronic structure (not a mismatched
/// CP2K box/cutoff). Offline CP2K pins at 10 Å / 400 Ry remain in REFERENCE
/// for a future mixer-window twin; this gate deliberately uses the
/// Atomiverse-convergent window (8 Å / 150 Ry) so e2e ≤ 60 s.
///
/// Physics setup (see REFERENCE.md "History"):
///   - UKS triplet (n_α=5, n_β=3) — closed-shell OT lands on a saddle.
///   - Dimers **centred** in the MT box — off-centre MT is translation-broken.
///   - Isolated diatomic → 1 stretch + 5 near-zero (T+R), not 3 crystal optics.
///
/// Public API only. No live CP2K. See REFERENCE.md.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/vibration/nist_constants.h"
#include "atomiverse/cpu/vibration/phonon.h"

using namespace ATV;

namespace {

// Atomiverse-convergent twin (REFERENCE: 8 Å / 150 Ry, ~3 s/SCF, UKS triplet).
constexpr double kL = 8.0;      // Å
constexpr double kBond = 2.246; // Å Si–Si
constexpr double kCutoffRy = 150.0;
constexpr double kMix = 0.2;
constexpr int kMaxIter = 200;
constexpr double kTol = 1e-7;
constexpr double kMassSi = 28.0855; // amu

constexpr int kNAtoms = 2;
constexpr int kNModes = 3 * kNAtoms;
constexpr int kNZeroModes = 5;
constexpr double kZeroModeTol = 80.0; // cm⁻¹ (MT box + coarse grid residual)
// Stretch vs 1D energy curvature dual-gate (same method / same window).
// Coarse MT (8 Å / 150 Ry) + residual force at the non-eq bond leave
// ~70 cm⁻¹ dual residual; gate is consistency, not absolute accuracy.
constexpr double kOmegaDualTol = 100.0; // cm⁻¹
// Phonon wall only (12 force SCFs + finalize). Warm UKS density guess makes
// this ~65 s at OMP=4 on epyc; budget is the e2e tier with 15 s headroom for
// host load (cold first SCF dominates).
constexpr double kWallBudget = 90.0; // s e2e (phonon phase)

cpu::Frame make_si2(double bond_A) {
    const double c = 0.5 * kL;
    const double half = 0.5 * bond_A;
    cpu::Frame frame(kNAtoms);
    for (int i = 0; i < kNAtoms; ++i) {
        frame.span(kAtom_type)[i] = 14;
        frame.span(kAtom_mass)[i] = static_cast<StorageReal>(kMassSi);
        frame.span(kAtom_tag)[i] = static_cast<TagInt>(i + 1);
        frame.span(kAtom_x)[i] =
            static_cast<StorageReal>(c + (i == 0 ? -half : half));
        frame.span(kAtom_y)[i] = static_cast<StorageReal>(c);
        frame.span(kAtom_z)[i] = static_cast<StorageReal>(c);
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    }
    frame.box = cpu::Box::cubic(kL);
    return frame;
}

cpu::gpw::GPWEvaluator make_eval(bool forces) {
    const cpu::gpw::Kind Si{
        molrs::Element::Si,
        cpu::gpw::GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::Si, "GTH-PBE-q4")};
    cpu::gpw::GPWEvaluator eval({Si});
    eval.cutoff_ry = kCutoffRy;
    eval.max_iter = kMaxIter;
    eval.tol = kTol;
    eval.mix = kMix;
    eval.xc = cpu::gpw::XcKind::PADE;
    eval.nonperiodic = true;
    eval.n_alpha = 5;
    eval.n_beta = 3;
    eval.compute_forces = forces;
    return eval;
}

/// @brief Harmonic stretch wavenumber from 1D energy curvature [cm⁻¹].
///
/// Matches `MassWeighting` + `PhononDiagonalizer` unit convention:
/// Hessian in Ha/Bohr², mass in **amu** (not electron mass), then
/// \(\omega_{\mathrm{cm}} = \sqrt{k/\mu}\times\mathrm{au2wavenumber()}\).
double omega_from_energy_curvature(double E0_ha, double Ep_ha, double Em_ha,
                                   double h_A) {
    const double k_ha_per_a2 = (Ep_ha + Em_ha - 2.0 * E0_ha) / (h_A * h_A);
    // r_A = r_Bohr * BOHR  ⇒  k_Bohr2 = k_A2 * BOHR²
    const double k_ha_per_bohr2 =
        k_ha_per_a2 * cpu::vib::nist::BOHR * cpu::vib::nist::BOHR;
    const double mu_amu = 0.5 * kMassSi;
    if (!(k_ha_per_bohr2 > 0.0) || !(mu_amu > 0.0))
        return 0.0;
    return std::sqrt(k_ha_per_bohr2 / mu_amu) * cpu::vib::nist::au2wavenumber();
}

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    // ── 1D energy curvature dual-gate (3 energy-only SCFs) ──────────────
    // h=5e-3 Å improves second-difference SNR vs h=1e-3 on this coarse MT
    // grid (non-stationary bond leaves a residual force that pollutes tiny h).
    const double h_bond = 5.0e-3; // Å
    auto eval_e = make_eval(/*forces=*/false);
    cpu::Frame f0 = make_si2(kBond);
    cpu::Frame fp = make_si2(kBond + h_bond);
    cpu::Frame fm = make_si2(kBond - h_bond);

    try {
        cpu::single_point(f0, eval_e);
    } catch (const std::exception& e) {
        std::printf("FAIL: eq energy SCF throw: %s\n", e.what());
        return 1;
    }
    if (!eval_e.last_result().converged) {
        std::printf("FAIL: eq energy SCF not converged\n");
        return 1;
    }
    const double E0 = eval_e.last_result().e_total;

    try {
        cpu::single_point(fp, eval_e);
    } catch (const std::exception& e) {
        std::printf("FAIL: +h energy SCF throw: %s\n", e.what());
        return 1;
    }
    if (!eval_e.last_result().converged) {
        std::printf("FAIL: +h energy SCF not converged\n");
        return 1;
    }
    const double Ep = eval_e.last_result().e_total;

    try {
        cpu::single_point(fm, eval_e);
    } catch (const std::exception& e) {
        std::printf("FAIL: -h energy SCF throw: %s\n", e.what());
        return 1;
    }
    if (!eval_e.last_result().converged) {
        std::printf("FAIL: -h energy SCF not converged\n");
        return 1;
    }
    const double Em = eval_e.last_result().e_total;

    const double omega_1d = omega_from_energy_curvature(E0, Ep, Em, h_bond);
    std::printf("  1D curvature: E0=%.8f Ep=%.8f Em=%.8f Ha  omega_1d=%.3f "
                "cm^-1\n",
                E0, Ep, Em, omega_1d);
    if (!(omega_1d > 50.0) || !(omega_1d < 1200.0)) {
        std::printf("FAIL: 1D stretch omega=%.3f cm^-1 outside physical band "
                    "[50,1200]\n",
                    omega_1d);
        return 1;
    }

    // ── Full FD phonon (12 force SCFs) ──────────────────────────────────
    cpu::Frame frame = make_si2(kBond);
    cpu::gpw::GPWEvaluator eval = make_eval(/*forces=*/true);

    cpu::vib::Phonon phonon(/*na=*/1, /*nb=*/1, /*nc=*/1);
    phonon.on(ATV::vib::Phase::force).add_ref(eval);
    phonon.enable_timing(false);
    phonon.build();

    const auto t0 = clock::now();
    cpu::vib::PhononResult r;
    try {
        r = phonon.run(frame);
    } catch (const std::exception& e) {
        std::printf("FAIL: Phonon::run throw: %s\n", e.what());
        return 1;
    }
    const auto t1 = clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();

    if (r.n_atoms != kNAtoms) {
        std::printf("FAIL: n_atoms got=%d want=%d\n", r.n_atoms, kNAtoms);
        return 1;
    }
    if (static_cast<int>(r.frequencies.size()) != kNModes) {
        std::printf("FAIL: n_modes got=%zu want=%d\n",
                    static_cast<std::size_t>(r.frequencies.size()), kNModes);
        return 1;
    }

    std::vector<double> w(kNModes);
    for (int k = 0; k < kNModes; ++k) {
        w[static_cast<std::size_t>(k)] = r.frequencies(k);
        if (!std::isfinite(w[static_cast<std::size_t>(k)])) {
            std::printf("FAIL: non-finite omega[%d]=%g\n", k,
                        w[static_cast<std::size_t>(k)]);
            return 1;
        }
        std::printf("  omega[%d] = % .4f cm^-1\n", k,
                    w[static_cast<std::size_t>(k)]);
    }

    std::vector<double> by_abs = w;
    std::sort(by_abs.begin(), by_abs.end(),
              [](double a, double b) { return std::abs(a) < std::abs(b); });
    int n_zero = 0;
    for (int k = 0; k < kNZeroModes; ++k)
        if (std::abs(by_abs[static_cast<std::size_t>(k)]) < kZeroModeTol)
            ++n_zero;
    if (n_zero != kNZeroModes) {
        std::printf(
            "FAIL: got %d of %d near-zero modes (|omega| < %.0f cm^-1)\n",
            n_zero, kNZeroModes, kZeroModeTol);
        return 1;
    }

    const double omega = by_abs[static_cast<std::size_t>(kNModes - 1)];
    const double d_omega = std::abs(omega - omega_1d);
    std::printf("  stretch: phonon=%.4f  1d_E=%.4f  |d|=%.3f cm^-1\n", omega,
                omega_1d, d_omega);
    if (d_omega > kOmegaDualTol) {
        std::printf("FAIL: |omega_phonon - omega_1d| = %.3f > %.1f cm^-1\n",
                    d_omega, kOmegaDualTol);
        return 1;
    }

    if (wall > kWallBudget) {
        std::printf("FAIL: wall=%.2f s exceeds %.0f s e2e budget\n", wall,
                    kWallBudget);
        return 1;
    }

    std::printf("PASS: gpw-fd-phonon-si  modes=%d  near-zero=%d  "
                "stretch=%.4f cm^-1 (1d %.4f, |d|=%.3f)  wall=%.2f s  "
                "perf_exempt (no fair CP2K twin at ATV-convergent window; "
                "offline 10A/400Ry pin in REFERENCE)\n",
                kNModes, n_zero, omega, omega_1d, d_omega, wall);
    return 0;
}
