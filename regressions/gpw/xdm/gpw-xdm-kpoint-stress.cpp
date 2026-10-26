/// @file regressions/gpw/gpw-xdm-kpoint-stress/reg_gpw-xdm-kpoint-stress.cpp
/// @brief END-TO-END: multi-k GPW analytic cell stress folds free-atom XDM.
///
/// The capability: the k-summed fixed-\(P_k\) stress Lagrangian rebuilds the
/// geometry-only \(E_{\mathrm{disp}}^{\mathrm{XDM}}\) at every strained cell
/// (same `xdm_pbe` call as the Γ path). Gates:
///   A. `E_disp` from the multi-k SCF equals the Γ dual-Python pin — XDM is
///      independent of the electronic k mesh (Γ pin cross-check).
///   B. `E_disp` at each strained cell (±h per axis, fixed fractional
///      coordinates, rcut 50 Å) matches fresh dual pure-Python pins — the
///      strain derivative being folded is externally validated point by point.
///   C. Analytic fixed-\(P_k\) stress vs the multi-SCF strain-FD oracle
///      (`gpw_stress_fd`, XDM folded through the energy path).
///   D. \(\sigma(\mathrm{xdm}) - \sigma(\mathrm{plain})\) equals the dual-py
///      strain-FD dispersion term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\).
///   E. Orthorhombic off-diagonals exactly zero; multi-k payload present;
///      `kStress` unit conversion.
///
/// Public API only (GPWEvaluator + Evaluate, `gpw_stress` / `gpw_stress_fd`). Offline
/// dual-Python pins in REFERENCE.md; no live oracle at ctest time. CP2K has
/// no stock TYPE XDM — perf_exempt (oracle: literature).

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <utility>
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
#include "atomiverse/cpu/gpw/internal/xdm_dispersion.h"
#include "atomiverse/molrs.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_xdm_kpoint_stress {

using namespace ATV;

namespace {

// Pin script / dual-python lattice sum (matches gpw-xdm-stress /
// pin_xdm_strain.py). Geometry-only → independent of electronic k-mesh.
constexpr double kPinAng2Bohr = 1.0 / 0.52917720859;
// Frame / GPWEvaluator conversion is units::kAngstromInBohr (inside the
// library). Slightly different from the pin constant → product E_disp
// differs at ~1e-11 Ha.
constexpr double kStrainH = 2.0e-3;

// Offline dual pure-Python free-atom XDM pins
// (tools/cp2k_oracle/inputs/gpw-xdm-kpoint-stress/pin_xdm_strain.py, 2026-08-01).
// Bit-identical to gpw-xdm-stress / gpw-xdm-ice unstrained pin.
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
// Gate A (product SCF via Frame): sibling gpw-xdm-ice / gpw-xdm-stress tier.
constexpr double kEDispProductTol = 1.0e-8; // Ha
// C: analytic fixed-P_k vs multi-SCF strain-FD — same tier as Γ XDM /
// multi-k MBD stress regs.
constexpr double kAnFdTol = 2.0e-5;     // Ha/Bohr³
constexpr double kOffDiagTol = 1.0e-12; // Ha/Bohr³ (must be exactly 0)
constexpr double kDiagFloor = 1.0e-10;  // Ha/Bohr³ — stress must be nonzero
// D: σ(xdm)−σ(plain) vs dual-python FD term (Γ XDM stress tier).
constexpr double kShiftTol = 1.0e-8; // Ha/Bohr³
constexpr double kWallBudget = 60.0; // s (whole section, including all twins)

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

int run() {
    using clock = std::chrono::steady_clock;
    const auto whole_start = clock::now();

    constexpr double L = 12.0; // Å, PERIODIC XYZ
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double xs[6] = {0.000000, 0.758602, 0.758602,
                          3.000000, 3.758602, 3.758602};
    const double ys[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double zs[6] = {0.000000, 0.504284, -0.504284,
                          0.000000, 0.504284, -0.504284};

    // ── B: E_disp under strain vs dual pure-Python pins (geometry-only) ───
    // Uses the pin Å→Bohr constant so the comparison is bit-class (1e-12).
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
                    pos[static_cast<std::size_t>(i)] = {xs[i] * kPinAng2Bohr,
                                                        ys[i] * kPinAng2Bohr,
                                                        zs[i] * kPinAng2Bohr};
                    pos[static_cast<std::size_t>(i)]
                       [static_cast<std::size_t>(a)] *= scale;
                }
                const double e =
                    cpu::gpw::xdm_pbe(Z, pos, cell, /*periodic=*/true, kRcutXdm)
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
        std::printf("B strained E_disp vs dual-python pins (6 cells): "
                    "max|d|=%.3e Ha\n",
                    max_err);
        check(max_err <= kEDispKernelTol, "B strained E_disp vs dual-py pins");
    }

    // ── A + product: multi-k GPW evaluator with XDM, analytic stress ──────
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

    cpu::gpw::GpwDft dft{{O, H}, 120.0, cpu::gpw::XcKind::PBE};
    dft.max_iter = 120;
    dft.tol = 1e-6;
    dft.mix = 0.4;
    dft.nonperiodic = false;
    dft.k_mesh[0] = 1;
    dft.k_mesh[1] = 1;
    dft.k_mesh[2] = 2; // MP 1×1×2, TR-folded
    dft.xdm = true;
    cpu::gpw::GPWEvaluator eval(std::move(dft));
    eval.compute_forces = false;
    eval.compute_stress = true;
    eval.stress_use_fd = false;
    eval.method().stress_strain_h = kStrainH;

    const auto t0 = clock::now();
    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);
    const auto t1 = clock::now();
    const double wall_product_s =
        std::chrono::duration<double>(t1 - t0).count();

    check(eval.last_result().converged, "A multi-k SCF converged");
    check(eval.last_result().n_kpts >= 1 && !eval.last_result().C_occ_k.empty(),
          "E multi-k C_occ_k payload present");
    const double e_disp = eval.last_result().e_disp;
    std::printf("A E_disp(multi-k) = %.14e Ha (Γ pin %.14e, |d|=%.2e)\n",
                e_disp, kEDispRef, std::abs(e_disp - kEDispRef));
    check(std::isfinite(e_disp)
              && std::abs(e_disp - kEDispRef) <= kEDispProductTol,
          "A E_disp equals the Γ dual-python pin (k-mesh independent)");

    const auto& san = eval.last_stress_ha_bohr3();
    for (int k = 3; k < 6; ++k)
        check(std::abs(san[static_cast<std::size_t>(k)]) <= kOffDiagTol,
              "E orthorhombic off-diagonal exactly 0");

    // ── C: multi-SCF strain-FD oracle on the same multi-k method ──────────
    // Same Frame as the product path, so twins share the product geometry.
    auto make_cfg = [&](bool xdm) {
        cpu::gpw::GpwDft m({O, H}, 120.0, cpu::gpw::XcKind::PBE);
        m.n_occ = 8;
        m.max_iter = 120;
        m.tol = 1e-6;
        m.mix = 0.4;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.xdm = xdm;
        return m;
    };

    const auto t2 = clock::now();
    // Public paths return eV/Å³; gates and prints stay in Ha/Bohr³.
    constexpr double kToHaBohr3 = 1.0 / units::kHartreePerBohr3InEvPerAngstrom3;
    auto sfd = cpu::gpw::gpw_stress_fd(frame, make_cfg(true), kStrainH);
    for (auto& v : sfd)
        v *= kToHaBohr3;
    const auto t3 = clock::now();
    const double wall_fd_s = std::chrono::duration<double>(t3 - t2).count();

    double max_diag_err = 0.0;
    double max_diag = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int a = 0; a < 3; ++a) {
        const double an = san[static_cast<std::size_t>(a)];
        const double fd = sfd[static_cast<std::size_t>(a)];
        check(std::isfinite(an) && std::isfinite(fd),
              "C finite stress components");
        max_diag_err = std::max(max_diag_err, std::abs(an - fd));
        max_diag = std::max(max_diag, std::abs(an));
        std::printf("  sigma_%s  analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[a], an, fd, std::abs(an - fd));
    }
    check(max_diag > kDiagFloor, "C stress diagonal nonzero");
    std::printf("C max|sigma_an - sigma_fd| = %.3e Ha/Bohr3\n", max_diag_err);
    check(max_diag_err <= kAnFdTol, "C analytic vs multi-SCF FD diagonal");

    // ── D: XDM wiring — sigma(xdm) − sigma(plain) vs dual-python FD term ──
    // stress() writes kStress/kEnergy: run it on a copy so the kStress
    // gate below still reads the evaluator's published value.
    std::array<double, 6> s_plain{};
    const auto plain_start = clock::now();
    {
        cpu::Frame fp = frame;
        cpu::gpw::GpwDft plain = make_cfg(false);
        plain.stress_strain_h = kStrainH;
        plain.build(fp);
        s_plain = plain.stress(fp);
        for (auto& v : s_plain)
            v *= kToHaBohr3;
    }
    const double wall_plain_s =
        std::chrono::duration<double>(clock::now() - plain_start).count();
    double max_shift_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double shift = san[static_cast<std::size_t>(a)]
                             - s_plain[static_cast<std::size_t>(a)];
        const double err = std::abs(shift - kSigmaXdmRef[a]);
        max_shift_err = std::max(max_shift_err, err);
        std::printf("  sigma_XDM_%s  product=% .10e  pin=% .10e  |d|=%.3e\n",
                    names[a], shift, kSigmaXdmRef[a], err);
    }
    check(max_shift_err <= kShiftTol,
          "D sigma(xdm)-sigma(plain) equals the dual-python strain-FD term");

    // ── E: Frame kStress unit conversion (eV/Å³) ──────────────────────────
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
        if (std::abs(st[static_cast<std::size_t>(k)] - expect)
            > 1e-9 * kStressVolA3)
            conv_ok = false;
    }
    check(conv_ok, "E kStress unit conversion");

    check(wall_product_s <= kWallBudget, "product wall <= 60 s");
    const double wall_total_s =
        std::chrono::duration<double>(clock::now() - whole_start).count();
    check(wall_total_s <= kWallBudget, "whole section wall <= 60 s");

    std::printf("gpw-xdm-kpoint-stress  |dE_disp|=%.2e Ha  "
                "max|an-fd|=%.3e  max|shift-pin|=%.3e Ha/Bohr3  iters=%d  "
                "nk=%d  wall_product=%.2f s  wall_fd_twin=%.2f s  "
                "wall_plain=%.2f s  wall_total=%.2f s\n",
                std::abs(e_disp - kEDispRef), max_diag_err, max_shift_err,
                eval.last_result().iters, eval.last_result().n_kpts,
                wall_product_s, wall_fd_s, wall_plain_s, wall_total_s);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-xdm-kpoint-stress\n");
    return 0;
}

} // namespace atv_reg_gpw_gpw_xdm_kpoint_stress

ATV_REGRESSION("gpw/xdm", "gpw-xdm-kpoint-stress",
               atv_reg_gpw_gpw_xdm_kpoint_stress::run);
