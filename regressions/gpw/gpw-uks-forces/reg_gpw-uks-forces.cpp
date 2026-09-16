/// @file regressions/gpw/gpw-uks-forces/reg_gpw-uks-forces.cpp
/// @brief END-TO-END: Γ-UKS GPW analytic O₂ forces vs offline CP2K FORCE_EVAL.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW (+ HF + PW).
/// Units on Frame: eV / eV·Å⁻¹; pin and prints in Ha / Ha·Bohr⁻¹.

#include <chrono>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

// Offline CP2K 2026.1 ENERGY_FORCE pin (Ha/Bohr). See REFERENCE.md.
constexpr double kF_cp2k[2][3] = {
    {2.02069184e-12, 2.05677955e-12, -3.50081576e-02},
    {2.43439049e-12, 2.40869221e-12, 3.50081358e-02},
};
// ENERGY| Total FORCE_EVAL (QS) — EPS_SCF 1e-7 ENERGY_FORCE twin.
constexpr double kE_cp2k_ha = -31.928797634435675;
// Force tol 5e-3: first-green single-grid vs CP2K multigrid (NGRIDS 4);
// unit analytic-vs-FD on UKS H2 is 5e-5 (ATV energy gradient correct).
// |F_z|~0.035 Ha/Bohr; observed twin residual ~2.7e-3.
constexpr double kForceTol = 5.0e-3;  // Ha/Bohr
constexpr double kEnergyTol = 1.0e-3; // Ha
constexpr double kSumFTol = 5.0e-4;   // Ha/Bohr

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    // O₂, R = 1.2075 Å, centered in 12 Å cubic cell (same as gpw-uks-pbe).
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

    cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

    cpu::gpw::GPWEvaluator eval({O});
    eval.cutoff_ry = 400.0;
    eval.max_iter = 200;
    eval.tol = 1e-7;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = true;
    eval.compute_forces = true;
    eval.n_alpha = 7;
    eval.n_beta = 5;

    const auto t0 = clock::now();
    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);
    const auto t1 = clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

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

    const double to_ha_bohr = units::kEvPerAngstromInHartreePerBohr;
    double max_err = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    double F_ha[2][3];
    for (int a = 0; a < 2; ++a) {
        F_ha[a][0] = static_cast<double>(frame.span(kAtom_fx)[a]) * to_ha_bohr;
        F_ha[a][1] = static_cast<double>(frame.span(kAtom_fy)[a]) * to_ha_bohr;
        F_ha[a][2] = static_cast<double>(frame.span(kAtom_fz)[a]) * to_ha_bohr;
        for (int k = 0; k < 3; ++k) {
            sum[k] += F_ha[a][k];
            const double err = std::abs(F_ha[a][k] - kF_cp2k[a][k]);
            max_err = std::max(max_err, err);
            std::printf("  atom %d k %d  F_atv=% .8e  F_cp2k=% .8e  |d|=%.3e\n",
                        a, k, F_ha[a][k], kF_cp2k[a][k], err);
        }
    }
    const double sum_mag =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);

    std::printf("E_atv=%.12f Ha  E_cp2k=%.12f Ha  |dE|=%.3e\n", e_ha,
                kE_cp2k_ha, de);
    std::printf("max|F_atv-F_cp2k|=%.3e Ha/Bohr  |sum F|=%.3e  wall=%.3f s\n",
                max_err, sum_mag, wall_s);
    std::printf("  <S2>=%.8f  iters=%d\n", eval.last_s2(),
                eval.last_result().iters);

    if (max_err > kForceTol) {
        std::printf("FAIL: force max err %.6e > tol %g Ha/Bohr\n", max_err,
                    kForceTol);
        return 1;
    }
    if (sum_mag > kSumFTol) {
        std::printf("FAIL: |sum F|=%.6e > tol %g Ha/Bohr\n", sum_mag, kSumFTol);
        return 1;
    }

    std::printf("PASS: gpw-uks-forces  max|dF|=%.3e  |sumF|=%.3e  |dE|=%.3e  "
                "wall=%.3f s\n",
                max_err, sum_mag, de, wall_s);
    return 0;
}
