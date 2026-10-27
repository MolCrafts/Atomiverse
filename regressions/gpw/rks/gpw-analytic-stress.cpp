/// @file regressions/gpw/gpw-analytic-stress/reg_gpw-analytic-stress.cpp
/// @brief END-TO-END: GPW analytic (fixed-P) cell stress vs multi-SCF strain-FD.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW (+ HF + PW).
/// Frame stress eV/Å³; pin and twin compare in Ha/Bohr³.

#include <array>
#include <cmath>
#include <cstdio>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_analytic_stress {

using namespace ATV;

namespace {

// Same offline CP2K energy pin as gpw-stress-water / gpw-forces-water.
constexpr double kE_cp2k_ha = -17.212337150586624;
// Soft energy twin (multigrid residual) — same as sibling water regs.
constexpr double kEnergyTol = 1.0e-3; // Ha
// Analytic fixed-P vs multi-SCF strain-FD (Hellmann–Feynman residual).
constexpr double kAnFdTol = 1.0e-5; // Ha/Bohr³
// Soft off-diag: product zeros shear; gate residual noise.
constexpr double kOffDiagTol = 1.0e-8; // Ha/Bohr³

} // namespace

int run() {
    cpu::Frame frame(3);
    const int z[3] = {8, 1, 1};
    const double x[3] = {5.0000, 5.7586, 4.2414};
    const double y[3] = {5.0000, 5.0000, 5.0000};
    const double zc[3] = {5.0000, 5.5043, 5.5043};
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = x[i];
        frame.span(kAtom_y)[i] = y[i];
        frame.span(kAtom_z)[i] = zc[i];
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    }
    frame.box = cpu::Box::cubic(10.0);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    // Product path: analytic fixed-P stress (default stress_use_fd=false).
    cpu::gpw::GpwDft dft{{O, H}, 400.0, cpu::xc::XCType::PBE};
    dft.max_iter = 200;
    dft.tol = 1e-7;
    dft.mix = 0.3;
    dft.nonperiodic = true;
    cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces*/ false);
    eval.compute_stress = true;
    eval.stress_use_fd = false;
    eval.method().stress_strain_h = 2e-3;

    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);

    if (!eval.last_result().converged) {
        std::printf("FAIL: GPW SCF did not converge (iters=%d)\n",
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

    const auto& san = eval.last_stress_ha_bohr3();
    for (int k = 3; k < 6; ++k) {
        if (std::abs(san[static_cast<std::size_t>(k)]) > kOffDiagTol) {
            std::printf("FAIL: analytic off-diag σ[%d]=%.3e (expect ~0)\n", k,
                        san[static_cast<std::size_t>(k)]);
            return 1;
        }
    }

    // Multi-SCF strain-FD oracle twin (public free function) on the same
    // Frame geometry and method parameters as the evaluator.
    cpu::gpw::GpwDft method{{O, H}, 400.0, cpu::xc::XCType::PBE};
    method.n_occ = 4; // H2O GTH: O q6 + 2×H q1 = 8 e → 4 doubly occupied
    method.max_iter = 200;
    method.tol = 1e-7;
    method.mix = 0.3;
    method.nonperiodic = true;
    method.local_pp_on_grid = true;
    method.build(frame);

    // gpw_stress_fd returns eV/Å³; the twin compares in Ha/Bohr³.
    const auto sfd_ev_a3 = cpu::gpw::gpw_stress_fd(frame, method, /*h=*/2e-3,
                                                   /*axes*/ {true, true, true});
    std::array<double, 6> sfd{};
    for (std::size_t k = 0; k < 6; ++k)
        sfd[k] = sfd_ev_a3[k] / units::kHartreePerBohr3InEvPerAngstrom3;

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

    std::printf("E_atv=%.12f Ha  E_cp2k=%.12f Ha  |dE|=%.3e\n", e_ha,
                kE_cp2k_ha, de);
    std::printf("max|σ_diag_an-σ_fd|=%.3e Ha/Bohr3\n", max_diag_err);

    if (max_diag_err > kAnFdTol) {
        std::printf(
            "FAIL: analytic vs FD max diag err %.6e > tol %g Ha/Bohr3\n",
            max_diag_err, kAnFdTol);
        return 1;
    }

    std::printf(
        "PASS: gpw-analytic-stress  max|dσ_an-fd|=%.3e  |dE|=%.3e  iters=%d\n",
        max_diag_err, de, eval.last_result().iters);
    return 0;
}

} // namespace atv_reg_gpw_gpw_analytic_stress

ATV_REGRESSION("gpw/rks", "gpw-analytic-stress",
               atv_reg_gpw_gpw_analytic_stress::run);
