/// @file regressions/gpw/gpw-mbd-stress/reg_gpw-mbd-stress.cpp
/// @brief END-TO-END: Γ GPW analytic cell stress with periodic MBD@rsSCS.
///
/// The capability: MBD@rsSCS is geometry-only, so it must enter the fixed-\(P\)
/// stress Lagrangian and be **rebuilt at every strained cell** — otherwise the
/// analytic stress differentiates a different functional than the one the SCF
/// converged. Gates:
///   1. `E_disp` matches the offline **pymbd** pin (the same pin
///      `gpw-mbd-periodic` uses), so the energy whose strain derivative we take
///      is the externally validated one.
///   2. Analytic fixed-\(P\) stress vs the multi-SCF strain-FD oracle
///      (`gpw_stress_fd`, which folds MBD through a full re-SCF at each strain).
///   3. Switching MBD on **moves** the diagonal — MBD is not silently dropped.
///   4. Orthorhombic off-diagonals are exactly zero; `kStress` unit conversion.
///
/// Multi-k MBD stress is covered by the sibling case
/// `regressions/gpw/gpw-mbd-kpoint-stress` (k-summed fixed-P_k Lagrangian
/// folds the same geometry-only E_disp). See REFERENCE.md.
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
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

constexpr double kAng2Bohr = 1.0 / 0.52917720859;

// Offline pure-Python pymbd pin, identical to regressions/gpw/gpw-mbd-periodic:
// PBE-MBD@rsSCS, free-atom volumes, periodic 12 Å cube, MBD k-mesh 2×2×2.
constexpr double kEDispRef = -1.4965720704989138e-03; // Ha
constexpr double kEDispTol = 1.0e-8;                  // Ha

// Analytic fixed-P vs multi-SCF strain-FD. Same tier as reg_gpw-d4-stress
// (2e-5): both differentiate the same functional, so the residual is the
// fixed-P (frozen density-matrix) approximation plus SCF noise.
constexpr double kAnFdTol = 2.0e-5;     // Ha/Bohr³
constexpr double kOffDiagTol = 1.0e-12; // Ha/Bohr³ (must be exactly 0)
constexpr double kDiagFloor = 1.0e-10;  // Ha/Bohr³ — stress must be nonzero
// MBD must move the diagonal. The MBD contribution is ~1e-7 Ha/Bohr³ on this
// cell (measured), so require a shift well above SCF noise but far below it.
constexpr double kMbdShiftFloor = 1.0e-10; // Ha/Bohr³
constexpr double kWallBudget = 60.0;       // s

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

    // ── (1) Product analytic stress with periodic MBD ─────────────────────
    cpu::gpw::GPWEvaluator eval({O, H});
    eval.cutoff_ry = 200.0;
    eval.max_iter = 120;
    eval.tol = 1e-6;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = false;
    eval.mbd = true;
    eval.mbd_k_mesh[0] = eval.mbd_k_mesh[1] = eval.mbd_k_mesh[2] = 2;
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

    const double e_disp = eval.last_result().e_disp;
    if (!std::isfinite(e_disp) || std::abs(e_disp - kEDispRef) > kEDispTol) {
        std::printf("FAIL: |E_disp-pymbd|=%.6e Ha  got=%.14e  want=%.14e "
                    "(tol %.1e)\n",
                    std::abs(e_disp - kEDispRef), e_disp, kEDispRef, kEDispTol);
        return 1;
    }
    std::printf("MBD E_disp=%.14e Ha (pin %.14e, |d|=%.2e)\n", e_disp,
                kEDispRef, std::abs(e_disp - kEDispRef));

    const auto& san = eval.last_stress_ha_bohr3();
    for (int k = 3; k < 6; ++k) {
        if (std::abs(san[static_cast<std::size_t>(k)]) > kOffDiagTol) {
            std::printf("FAIL: orthorhombic off-diagonal %d = %.3e != 0\n", k,
                        san[static_cast<std::size_t>(k)]);
            return 1;
        }
    }

    // ── (2) Multi-SCF strain-FD oracle on the same method ─────────────────
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& k = (z[i] == 8) ? O : H;
        const double p[3] = {xs[i] * kAng2Bohr, ys[i] * kAng2Bohr,
                             zs[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(p, k.basis, k.pot, z[i]));
    }

    cpu::gpw::GpwDft method;
    method.cell[0] = method.cell[1] = method.cell[2] = L * kAng2Bohr;
    method.cutoff_ry = 200.0;
    method.n_occ = 8;
    method.max_iter = 120;
    method.tol = 1e-6;
    method.mix = 0.4;
    method.xc = cpu::gpw::XcKind::PBE;
    method.nonperiodic = false;
    method.local_pp_on_grid = true;
    method.mbd = true;
    method.mbd_k_mesh[0] = method.mbd_k_mesh[1] = method.mbd_k_mesh[2] = 2;

    const auto t2 = clock::now();
    const auto sfd =
        cpu::gpw::internal::gpw_stress_fd_impl(atoms, method, 2e-3);
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

    // ── (3) MBD must actually reach the stress ────────────────────────────
    cpu::gpw::GpwDft plain;
    plain.cell[0] = plain.cell[1] = plain.cell[2] = L * kAng2Bohr;
    plain.cutoff_ry = 200.0;
    plain.n_occ = 8;
    plain.max_iter = 120;
    plain.tol = 1e-6;
    plain.mix = 0.4;
    plain.xc = cpu::gpw::XcKind::PBE;
    plain.nonperiodic = false;
    plain.local_pp_on_grid = true;
    plain.mbd = false;
    const auto s_plain =
        cpu::gpw::internal::gpw_stress_analytic_impl(atoms, plain, 2e-3);

    double max_shift = 0.0;
    for (int a = 0; a < 3; ++a)
        max_shift = std::max(max_shift,
                             std::abs(san[static_cast<std::size_t>(a)]
                                      - s_plain[static_cast<std::size_t>(a)]));
    if (!(max_shift > kMbdShiftFloor)) {
        std::printf("FAIL: sigma(mbd) == sigma(plain) to %.3e — MBD is not "
                    "entering the stress Lagrangian\n",
                    max_shift);
        return 1;
    }

    // ── (4) Frame kStress unit conversion (eV/Å³) ─────────────────────────
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

    std::printf("PASS: gpw-mbd-stress  |dE_disp|=%.2e Ha  "
                "max|sigma_an-sigma_fd|=%.3e  max|sigma|=%.3e  "
                "MBD shift=%.3e Ha/Bohr3  iters=%d  "
                "wall_product=%.2f s  wall_fd_twin=%.2f s\n",
                std::abs(e_disp - kEDispRef), max_diag_err, max_diag, max_shift,
                eval.last_result().iters, wall_product_s, wall_fd_s);
    return 0;
}
