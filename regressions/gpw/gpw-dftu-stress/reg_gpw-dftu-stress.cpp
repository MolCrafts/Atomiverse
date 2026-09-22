/// @file regressions/gpw/gpw-dftu-stress/reg_gpw-dftu-stress.cpp
/// @brief END-TO-END: Γ-UKS GPW + Dudarev DFT+U analytic cell stress.
///
/// NiO molecule (same twin as gpw-dftu-nio). Dual-gate: product analytic
/// fixed-P vs multi-SCF strain-FD. Soft energy pin reuses offline CP2K
/// PLUS_U ENERGY from gpw-dftu-nio. See REFERENCE.md.
///
/// Public API only. Requires ATV_BUILD_GPW (+ HF + PW).

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

// Offline CP2K 2026.1 ENERGY pin (same twin as gpw-dftu-nio).
constexpr double kE_cp2k_ha = -184.92466076474312;
constexpr double kEnergyTol = 5.0e-2; // Ha — single-grid vs multigrid+OT
// Analytic fixed-P vs multi-SCF strain-FD (UKS + U residual).
constexpr double kAnFdTol = 2.0e-4;    // Ha/Bohr³
constexpr double kOffDiagTol = 1.0e-8; // Ha/Bohr³
constexpr double kDiagFloor = 1.0e-8;  // Ha/Bohr³

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    // NiO molecule, R(Ni–O)=1.668 Å, centered in 10 Å cubic cell (MT).
    constexpr double L = 10.0;
    constexpr double R = 1.668;
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 28;
    frame.span(kAtom_type)[1] = 8;
    frame.span(kAtom_mass)[0] = 58.693;
    frame.span(kAtom_mass)[1] = 15.999;
    frame.span(kAtom_x)[0] = 0.5 * L;
    frame.span(kAtom_y)[0] = 0.5 * L;
    frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
    frame.span(kAtom_x)[1] = 0.5 * L;
    frame.span(kAtom_y)[1] = 0.5 * L;
    frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
    for (int i = 0; i < 2; ++i)
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    frame.box = cpu::Box::cubic(L);

    const cpu::gpw::Kind Ni{
        molrs::Element::Ni,
        cpu::gpw::GaussianBasis(molrs::Element::Ni, "SZV-GTH-PADE-q18"),
        cpu::gpw::GthPotential(molrs::Element::Ni, "GTH-PADE-q18")};
    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "SZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PADE-q6")};

    // Product path: analytic fixed-P UKS + DFT+U stress.
    cpu::gpw::GPWEvaluator eval({Ni, O});
    eval.cutoff_ry = 250.0;
    eval.max_iter = 300;
    eval.tol = 1e-5;
    eval.mix = 0.35;
    eval.xc = cpu::gpw::XcKind::PADE;
    eval.nonperiodic = true;
    eval.n_alpha = 13;
    eval.n_beta = 11;
    eval.plus_u.push_back({molrs::Element::Ni, /*l=*/2, /*u_ev=*/6.0});
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
        std::printf("FAIL: GPW UKS+DFT+U SCF did not converge (iters=%d)\n",
                    eval.last_result().iters);
        return 1;
    }
    const double e_ha = eval.last_energy_hartree();
    if (!std::isfinite(e_ha)) {
        std::printf("FAIL: non-finite energy\n");
        return 1;
    }
    const double de = std::abs(e_ha - kE_cp2k_ha);
    if (de > kEnergyTol) {
        std::printf("FAIL: |E-E_cp2k|=%.6e Ha  got=%.12f  want=%.12f  tol=%g\n",
                    de, e_ha, kE_cp2k_ha, kEnergyTol);
        return 1;
    }
    if (!(eval.last_result().e_dftu > 1e-4)) {
        std::printf("FAIL: e_dftu=%.3e expected > 1e-4 Ha\n",
                    eval.last_result().e_dftu);
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
        std::printf("FAIL: max|σ_diag|=%.3e below floor %g (degenerate?)\n",
                    max_diag, kDiagFloor);
        return 1;
    }

    // Multi-SCF strain-FD oracle twin (public free function).
    constexpr double kAng2Bohr = 1.8897259885789;
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(2);
    const double x[2] = {0.5 * L, 0.5 * L};
    const double y[2] = {0.5 * L, 0.5 * L};
    const double zc[2] = {0.5 * L - 0.5 * R, 0.5 * L + 0.5 * R};
    const cpu::gpw::Kind kinds[2] = {Ni, O};
    const int Z[2] = {28, 8};
    for (int i = 0; i < 2; ++i) {
        const double pos[3] = {x[i] * kAng2Bohr, y[i] * kAng2Bohr,
                               zc[i] * kAng2Bohr};
        atoms.push_back(
            cpu::gpw::GpwAtom(pos, kinds[i].basis, kinds[i].pot, Z[i]));
    }
    cpu::gpw::GpwDft method;
    method.cell[0] = method.cell[1] = method.cell[2] = L * kAng2Bohr;
    method.cutoff_ry = 250.0;
    method.n_alpha = 13;
    method.n_beta = 11;
    method.max_iter = 300;
    method.tol = 1e-5;
    method.mix = 0.35;
    method.xc = cpu::gpw::XcKind::PADE;
    method.nonperiodic = true;
    method.local_pp_on_grid = true;
    method.plus_u.push_back({molrs::Element::Ni, /*l=*/2, /*u_ev=*/6.0});
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

    std::printf("E_atv=%.12f Ha  E_cp2k=%.12f Ha  |dE|=%.3e  e_dftu=%.6f Ha\n",
                e_ha, kE_cp2k_ha, de, eval.last_result().e_dftu);
    std::printf("max|σ_diag_an-σ_fd|=%.3e Ha/Bohr3  <S2>=%.6f  iters=%d\n",
                max_diag_err, eval.last_s2(), eval.last_result().iters);
    std::printf("wall_product=%.3f s  wall_fd_twin=%.3f s\n", wall_product_s,
                wall_fd_s);

    if (max_diag_err > kAnFdTol) {
        std::printf(
            "FAIL: analytic vs FD max diag err %.6e > tol %g Ha/Bohr3\n",
            max_diag_err, kAnFdTol);
        return 1;
    }

    std::printf("PASS: gpw-dftu-stress  max|dσ_an-fd|=%.3e  |dE|=%.3e  "
                "wall_prod=%.3f s\n",
                max_diag_err, de, wall_product_s);
    return 0;
}
