/// @file regressions/gpw/gpw-d4-stress/reg_gpw-d4-stress.cpp
/// @brief END-TO-END: periodic water-dimer GPW PBE-D4 analytic cell stress.
///
/// Dual-gate: product analytic fixed-P stress vs multi-SCF strain-FD on the
/// same d4=true method. Soft E_disp pin reuses offline dftd4 3.7.0 from
/// gpw-d4-water. Public API only. See REFERENCE.md.

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
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

// Offline dftd4 3.7.0 two-body pin (same geometry as gpw-d4-water).
constexpr double kEDispRef = -0.00099458842745846774; // Ha
constexpr double kEDispTol = 1.0e-8;                  // Ha
// Analytic fixed-P vs multi-SCF strain-FD (geometry-only D4 residual).
constexpr double kAnFdTol = 2.0e-5;    // Ha/Bohr³
constexpr double kOffDiagTol = 1.0e-8; // Ha/Bohr³
constexpr double kDiagFloor = 1.0e-10; // Ha/Bohr³

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    // Periodic water dimer, 12 Å cubic cell (REFERENCE.md / gpw-d4-water).
    constexpr double L = 12.0; // Å
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double x[6] = {0.000000, 0.758602, 0.758602,
                         3.000000, 3.758602, 3.758602};
    const double y[6] = {0.000000, 0.000000, 0.000000,
                         0.000000, 0.000000, 0.000000};
    const double zc[6] = {0.000000, 0.504284, -0.504284,
                          0.000000, 0.504284, -0.504284};

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
        std::printf("FAIL: |E_disp-pin|=%.6e Ha  got=%.14e  want=%.14e\n",
                    std::abs(e_disp - kEDispRef), e_disp, kEDispRef);
        return 1;
    }
    if (e_disp >= 0.0) {
        std::printf("FAIL: expected attractive E_disp, got %.6e\n", e_disp);
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

    // Multi-SCF strain-FD oracle twin (same geometry + d4).
    constexpr double kAng2Bohr = 1.8897259885789;
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
    method.build(atoms);

    const auto t2 = clock::now();
    const auto sfd = cpu::gpw::gpw_stress_fd(
        atoms, method, /*h=*/2e-3, /*axes*/ {true, true, true},
        eval.last_result().C.size() > 0 ? &eval.last_result().C : nullptr);
    const auto t3 = clock::now();
    const double wall_fd_s = std::chrono::duration<double>(t3 - t2).count();

    double max_diag_err = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int a = 0; a < 3; ++a) {
        const double an = san[static_cast<std::size_t>(a)];
        const double fd = sfd[static_cast<std::size_t>(a)];
        const double err = std::abs(an - fd);
        max_diag_err = std::max(max_diag_err, err);
        std::printf("  σ_%s  analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[a], an, fd, err);
        if (!std::isfinite(an) || !std::isfinite(fd)) {
            std::printf("FAIL: non-finite stress component %s\n", names[a]);
            return 1;
        }
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

    const double e_ha = eval.last_energy_hartree();
    std::printf("E_atv=%.12f Ha  E_disp=%.14e Ha (pin %.14e)\n", e_ha, e_disp,
                kEDispRef);
    std::printf("max|σ_diag_an-σ_fd|=%.3e Ha/Bohr3  iters=%d\n", max_diag_err,
                eval.last_result().iters);
    std::printf("wall_product=%.3f s  wall_fd_twin=%.3f s\n", wall_product_s,
                wall_fd_s);

    if (max_diag_err > kAnFdTol) {
        std::printf(
            "FAIL: analytic vs FD max diag err %.6e > tol %g Ha/Bohr3\n",
            max_diag_err, kAnFdTol);
        return 1;
    }

    std::printf("PASS: gpw-d4-stress  max|dσ_an-fd|=%.3e  |dE_disp|=%.3e  "
                "wall_prod=%.3f s\n",
                max_diag_err, std::abs(e_disp - kEDispRef), wall_product_s);
    return 0;
}
