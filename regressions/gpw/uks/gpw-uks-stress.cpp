/// @file regressions/gpw/gpw-uks-stress/reg_gpw-uks-stress.cpp
/// @brief END-TO-END: Γ-UKS GPW analytic O₂ cell stress vs multi-SCF FD + energy pin.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW (+ HF + PW).
/// Frame stress eV/Å³; pin and twin compare in Ha/Bohr³.

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
#include "atomiverse/molpy.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_uks_stress {

using namespace ATV;

namespace {

// Offline CP2K 2026.1 ENERGY pin (same twin as gpw-uks-forces).
constexpr double kE_cp2k_ha = -31.928797634435675;
constexpr double kEnergyTol = 1.0e-3; // Ha
// Analytic fixed-P vs multi-SCF strain-FD (UKS O₂ residual).
constexpr double kAnFdTol = 1.0e-4;    // Ha/Bohr³
constexpr double kOffDiagTol = 1.0e-8; // Ha/Bohr³
constexpr double kDiagFloor = 1.0e-8;  // Ha/Bohr³

} // namespace

int run() {
    using clock = std::chrono::steady_clock;

    // O₂, R = 1.2075 Å, centered in 12 Å cubic cell (same as gpw-uks-forces).
    cpu::Frame frame(2);
    constexpr double L = 12.0;
    constexpr double R = 1.2075;
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 8;
        frame.span(kAtom_mass)[i] = 15.999;
        frame.span(kAtom_x)[i] = 0.5 * L;
        frame.span(kAtom_y)[i] = 0.5 * L;
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    }
    frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
    frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
    frame.box = cpu::Box::cubic(L);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

    // One method recipe for the product evaluator and the FD twin.
    auto make_method = [&O]() {
        cpu::gpw::GpwDft m({O}, 400.0, cpu::gpw::XcKind::PBE);
        m.n_alpha = 7;
        m.n_beta = 5;
        m.max_iter = 200;
        m.tol = 1e-7;
        m.mix = 0.4;
        m.nonperiodic = true;
        m.local_pp_on_grid = true;
        return m;
    };

    // Product path: analytic fixed-P UKS stress.
    cpu::gpw::GPWEvaluator eval(make_method());
    eval.compute_forces = false;
    eval.compute_stress = true;
    eval.stress_use_fd = false;
    eval.method().stress_strain_h = 2e-3;

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
        std::printf("FAIL: GPW UKS SCF did not converge (iters=%d)\n",
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
    const double max_diag =
        std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
    if (max_diag < kDiagFloor) {
        std::printf("FAIL: max|σ_diag|=%.3e below floor %g (degenerate?)\n",
                    max_diag, kDiagFloor);
        return 1;
    }

    // Multi-SCF strain-FD oracle twin (public free function).
    cpu::gpw::GpwDft method = make_method();
    method.build(frame);

    const auto t2 = clock::now();
    auto sfd = cpu::gpw::gpw_stress_fd(frame, method, /*h=*/2e-3,
                                       /*axes*/ {true, true, true});
    // gpw_stress_fd returns eV/Å³; the gate compares in Ha/Bohr³.
    for (auto& v : sfd)
        v /= units::kHartreePerBohr3InEvPerAngstrom3;
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

    std::printf("E_atv=%.12f Ha  E_cp2k=%.12f Ha  |dE|=%.3e\n", e_ha,
                kE_cp2k_ha, de);
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

    std::printf("PASS: gpw-uks-stress  max|dσ_an-fd|=%.3e  |dE|=%.3e  "
                "wall_prod=%.3f s\n",
                max_diag_err, de, wall_product_s);
    return 0;
}

} // namespace atv_reg_gpw_gpw_uks_stress

ATV_REGRESSION("gpw/uks", "gpw-uks-stress", atv_reg_gpw_gpw_uks_stress::run);
