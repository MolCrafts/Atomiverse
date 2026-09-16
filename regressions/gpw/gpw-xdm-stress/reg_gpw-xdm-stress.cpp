/// @file regressions/gpw/gpw-xdm-stress/reg_gpw-xdm-stress.cpp
/// @brief END-TO-END: Γ GPW analytic cell stress folding free-atom XDM.
///
/// The capability: free-atom XDM is geometry-only, so it enters the fixed-\(P\)
/// stress Lagrangian and is **rebuilt at every strained cell** — otherwise the
/// analytic stress differentiates a different functional than the one the SCF
/// converged. The fold shipped with the dispersion-stress plumbing; this case
/// adds the dedicated pins. Gates:
///   1. `E_disp` (product SCF payload) matches the offline dual pure-Python
///      pin — bit-identical provenance with the `gpw-xdm-ice` pin.
///   2. `E_disp` under strain (±h per axis, fixed fractional coordinates,
///      rcut 50 Å — the exact evaluation the stress Lagrangian folds) matches
///      fresh dual pure-Python pins at each of the 6 strained cells.
///   3. Analytic fixed-\(P\) stress vs the multi-SCF strain-FD oracle
///      (`gpw_stress_fd_impl`, XDM folded through a full re-SCF per strain).
///   4. σ(xdm) − σ(plain) per axis equals the externally pinned XDM strain-FD
///      term −ΔE_disp/(2hΩ₀) — XDM is not silently dropped, and its stress
///      contribution is the pinned number.
///   5. Orthorhombic off-diagonals exactly zero; `kStress` unit conversion.
///
/// Public API only. No live third-party oracle. See REFERENCE.md.

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
#include "atomiverse/cpu/gpw/internal/xdm_dispersion.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

// Pin script / dual-python lattice sum (matches gpw-xdm-ice / pin_xdm_strain.py).
constexpr double kPinAng2Bohr = 1.0 / 0.52917720859;
// Frame / GPWEvaluator conversion (CODATA 2006; units::kAngstromInBohr).
// Slightly different from the pin constant → product E_disp differs at ~1e-11 Ha.
constexpr double kFrameAng2Bohr = units::kAngstromInBohr;
constexpr double kStrainH = 2e-3;

// Offline dual pure-Python free-atom XDM pins
// (tools/cp2k_oracle/inputs/gpw-xdm-stress/pin_xdm_strain.py, 2026-07-31).
// The unstrained value bit-reproduces the shipped gpw-xdm-ice periodic pin.
constexpr double kEDispRef = -2.1148157612879029e-03; // Ha, unstrained
// [axis][0]=+h, [axis][1]=−h — cell axis scaled 1±h, fixed fractional coords.
constexpr double kEDispStrainRef[3][2] = {
    {-2.1110849131940963e-03, -2.1185431863446984e-03}, // x
    {-2.1147774136190569e-03, -2.1148546246483736e-03}, // y
    {-2.1146664855098696e-03, -2.1149651549478578e-03}, // z
};
// σ_XDM per-axis strain-FD terms −ΔE/(2hΩ₀) from the same pins [Ha/Bohr³].
constexpr double kSigmaXdmRef[3] = {-1.5989612853e-07, -1.6553087314e-09,
                                    -6.4031024180e-09};

// Gate B (direct xdm_pbe + pin Å→Bohr): deterministic bit-class match.
constexpr double kEDispKernelTol = 1.0e-12; // Ha
// Gate A (product SCF via Frame): sibling gpw-xdm-ice / gpw-xdm-forces tier —
// residual is the pin vs units::kAngstromInBohr conversion, not the kernel.
constexpr double kEDispProductTol = 1.0e-8; // Ha
constexpr double kAnFdTol = 2.0e-5;   // Ha/Bohr³ — Γ dispersion-stress tier
constexpr double kShiftTol = 1.0e-8;  // Ha/Bohr³ — σ(xdm)−σ(plain) vs pin
constexpr double kOffDiagTol = 1.0e-12; // Ha/Bohr³ (must be exactly 0)
constexpr double kDiagFloor = 1.0e-10;  // Ha/Bohr³ — stress must be nonzero
constexpr double kWallBudget = 60.0;    // s

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    constexpr double L = 12.0; // Å, PERIODIC XYZ
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double xs[6] = {0.000000, 0.758602, 0.758602,
                          3.000000, 3.758602, 3.758602};
    const double ys[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double zs[6] = {0.000000, 0.504284, -0.504284,
                          0.000000, 0.504284, -0.504284};

    // ── (2) Strain-pin parity on the folded kernel (geometry-only) ────────
    // The stress Lagrangian folds xdm_pbe(Z, pos, cell, periodic, 50 Å) at
    // the strained geometry (fixed fractional coordinates). Check that exact
    // evaluation against the dual pure-Python pins, per axis and sign. Uses
    // the pin Å→Bohr constant so the comparison is bit-class (1e-12).
    {
        constexpr double kRcutXdm = 50.0 * kPinAng2Bohr;
        const std::vector<int> Z(z, z + 6);
        double max_err = 0.0;
        for (int a = 0; a < 3; ++a) {
            for (int s = 0; s < 2; ++s) {
                const double scale = 1.0 + (s == 0 ? kStrainH : -kStrainH);
                std::vector<std::array<double, 3>> pos(6);
                double cell[3] = {L * kPinAng2Bohr, L * kPinAng2Bohr,
                                  L * kPinAng2Bohr};
                cell[a] *= scale;
                for (int i = 0; i < 6; ++i) {
                    pos[static_cast<std::size_t>(i)] = {
                        xs[i] * kPinAng2Bohr, ys[i] * kPinAng2Bohr,
                        zs[i] * kPinAng2Bohr};
                    pos[static_cast<std::size_t>(i)][static_cast<std::size_t>(
                        a)] *= scale;
                }
                const double e =
                    cpu::gpw::xdm_pbe(Z, pos, cell, /*periodic=*/true,
                                      kRcutXdm)
                        .energy;
                const double ref = kEDispStrainRef[a][s];
                const double err = std::abs(e - ref);
                max_err = std::max(max_err, err);
                if (!std::isfinite(e) || err > kEDispKernelTol) {
                    std::printf("FAIL: strained E_disp axis %d %ch: got=%.16e "
                                "want=%.16e |d|=%.3e Ha (tol %.1e)\n",
                                a, s == 0 ? '+' : '-', e, ref, err,
                                kEDispKernelTol);
                    return 1;
                }
            }
        }
        std::printf("strained E_disp vs dual-python pins (6 cells): "
                    "max|d|=%.3e Ha\n",
                    max_err);
    }

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

    // ── (1) Product analytic stress with periodic free-atom XDM ───────────
    cpu::gpw::GPWEvaluator eval({O, H});
    eval.cutoff_ry = 200.0;
    eval.max_iter = 120;
    eval.tol = 1e-6;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = false;
    eval.xdm = true;
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
    if (!std::isfinite(e_disp)
        || std::abs(e_disp - kEDispRef) > kEDispProductTol) {
        std::printf("FAIL: |E_disp-pin|=%.6e Ha  got=%.16e  want=%.16e "
                    "(tol %.1e)\n",
                    std::abs(e_disp - kEDispRef), e_disp, kEDispRef,
                    kEDispProductTol);
        return 1;
    }
    std::printf("XDM E_disp=%.16e Ha (pin %.16e, |d|=%.2e)\n", e_disp,
                kEDispRef, std::abs(e_disp - kEDispRef));

    const auto& san = eval.last_stress_ha_bohr3();
    for (int k = 3; k < 6; ++k) {
        if (std::abs(san[static_cast<std::size_t>(k)]) > kOffDiagTol) {
            std::printf("FAIL: orthorhombic off-diagonal %d = %.3e != 0\n", k,
                        san[static_cast<std::size_t>(k)]);
            return 1;
        }
    }

    // ── (3) Multi-SCF strain-FD oracle on the same method ─────────────────
    // Frame-path Bohr conversion so twins share the product geometry (not the
    // pin constant — that residual is gated separately on E_disp only).
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& k = (z[i] == 8) ? O : H;
        const double p[3] = {xs[i] * kFrameAng2Bohr, ys[i] * kFrameAng2Bohr,
                             zs[i] * kFrameAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(p, k.basis, k.pot, z[i]));
    }

    cpu::gpw::GpwDft method;
    method.cell[0] = method.cell[1] = method.cell[2] = L * kFrameAng2Bohr;
    method.cutoff_ry = 200.0;
    method.n_occ = 8;
    method.max_iter = 120;
    method.tol = 1e-6;
    method.mix = 0.4;
    method.xc = cpu::gpw::XcKind::PBE;
    method.nonperiodic = false;
    method.local_pp_on_grid = true;
    method.xdm = true;

    const auto t2 = clock::now();
    const auto sfd =
        cpu::gpw::internal::gpw_stress_fd_impl(atoms, method, kStrainH);
    const auto t3 = clock::now();
    const double wall_fd_s = std::chrono::duration<double>(t3 - t2).count();

    double max_diag_err = 0.0;
    double max_diag = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int a = 0; a < 3; ++a) {
        const double an = san[static_cast<std::size_t>(a)];
        const double fd = sfd[static_cast<std::size_t>(a)];
        if (!std::isfinite(an) || !std::isfinite(fd)) {
            std::printf("FAIL: non-finite stress component %s\n", names[a]);
            return 1;
        }
        max_diag_err = std::max(max_diag_err, std::abs(an - fd));
        max_diag = std::max(max_diag, std::abs(an));
        std::printf("  sigma_%s  analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[a], an, fd, std::abs(an - fd));
    }
    if (max_diag < kDiagFloor) {
        std::printf("FAIL: stress diagonal is ~zero (max=%.3e)\n", max_diag);
        return 1;
    }
    if (max_diag_err > kAnFdTol) {
        std::printf("FAIL: analytic vs FD max diag err %.6e > %.1e Ha/Bohr3\n",
                    max_diag_err, kAnFdTol);
        return 1;
    }

    // ── (4) The XDM stress contribution is the pinned number ──────────────
    cpu::gpw::GpwDft plain;
    plain.cell[0] = plain.cell[1] = plain.cell[2] = L * kFrameAng2Bohr;
    plain.cutoff_ry = 200.0;
    plain.n_occ = 8;
    plain.max_iter = 120;
    plain.tol = 1e-6;
    plain.mix = 0.4;
    plain.xc = cpu::gpw::XcKind::PBE;
    plain.nonperiodic = false;
    plain.local_pp_on_grid = true;
    plain.xdm = false;
    const auto s_plain =
        cpu::gpw::internal::gpw_stress_analytic_impl(atoms, plain, kStrainH);

    double max_shift_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double shift = san[static_cast<std::size_t>(a)]
                             - s_plain[static_cast<std::size_t>(a)];
        const double err = std::abs(shift - kSigmaXdmRef[a]);
        max_shift_err = std::max(max_shift_err, err);
        std::printf("  sigma_XDM_%s  product=% .10e  pin=% .10e  |d|=%.3e\n",
                    names[a], shift, kSigmaXdmRef[a], err);
    }
    if (max_shift_err > kShiftTol) {
        std::printf("FAIL: sigma(xdm)-sigma(plain) vs dual-python FD term: "
                    "max|d|=%.6e > %.1e Ha/Bohr3\n",
                    max_shift_err, kShiftTol);
        return 1;
    }

    // ── (5) Frame kStress unit conversion (eV/Å³) ─────────────────────────
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
        if (std::abs(st[static_cast<std::size_t>(k)] - expect) > 1e-9 * kStressVolA3) {
            std::printf("FAIL: kStress unit conversion mismatch component %d "
                        "(got %.8e want %.8e)\n",
                        k, st[static_cast<std::size_t>(k)], expect);
            return 1;
        }
    }

    if (wall_product_s > kWallBudget) {
        std::printf("FAIL: product wall=%.2f s exceeds %.0f s e2e budget\n",
                    wall_product_s, kWallBudget);
        return 1;
    }

    std::printf("PASS: gpw-xdm-stress  |dE_disp|=%.2e Ha  "
                "max|sigma_an-sigma_fd|=%.3e  max|sigma|=%.3e  "
                "max|shift-pin|=%.3e Ha/Bohr3  iters=%d  "
                "wall_product=%.2f s  wall_fd_twin=%.2f s\n",
                std::abs(e_disp - kEDispRef), max_diag_err, max_diag,
                max_shift_err, eval.last_result().iters, wall_product_s,
                wall_fd_s);
    return 0;
}
