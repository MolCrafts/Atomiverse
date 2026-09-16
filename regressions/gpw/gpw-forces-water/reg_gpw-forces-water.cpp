/// @file regressions/gpw/gpw-forces-water/reg_gpw-forces-water.cpp
/// @brief END-TO-END: GPW analytic water forces vs offline CP2K FORCE_EVAL.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW (+ HF + PW).
/// Units on Frame: eV / eV·Å⁻¹; pin and prints in Ha / Ha·Bohr⁻¹.

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
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

namespace {

// Offline CP2K 2026.1 ENERGY_FORCE pin (Ha/Bohr). See REFERENCE.md.
constexpr double kF_cp2k[3][3] = {
    {-6.62274644e-08, -1.03414433e-06, -8.96660013e-02},
    {5.16349669e-02, -2.88080683e-09, 4.49073908e-02},
    {-5.16349704e-02, -2.33864544e-10, 4.49073720e-02},
};
constexpr double kE_cp2k_ha = -17.21233715058673;
constexpr double kForceTol = 1.0e-3; // Ha/Bohr (spec A6)
// Energy soft check: same-method twin; multigrid/REL_CUTOFF residual ~ few 1e-4 Ha
// (example 09 reports the same CP2K E; ATV product path is within ~5e-4 Ha).
constexpr double kEnergyTol = 1.0e-3; // Ha
// Translational residual is numerical (CP2K print residual ~1.5e-4 on same twin).
constexpr double kSumFTol = 5.0e-4; // Ha/Bohr (spec A7)

} // namespace

int main() {
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

    cpu::gpw::GPWEvaluator eval(
        /*kinds*/ {O, H}, /*cutoff_ry*/ 400.0, /*max_iter*/ 200, /*tol*/ 1e-7,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
        /*compute_forces*/ true);

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

    // Frame forces are eV/Å → Ha/Bohr for pin compare.
    const double to_ha_bohr = units::kEvPerAngstromInHartreePerBohr;
    double max_err = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    double F_ha[3][3];
    for (int a = 0; a < 3; ++a) {
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
    std::printf("max|F_atv-F_cp2k|=%.3e Ha/Bohr  |sum F|=%.3e\n", max_err,
                sum_mag);

    if (max_err > kForceTol) {
        std::printf("FAIL: force max err %.6e > tol %g Ha/Bohr\n", max_err,
                    kForceTol);
        return 1;
    }
    if (sum_mag > kSumFTol) {
        std::printf("FAIL: |sum F|=%.6e > tol %g Ha/Bohr\n", sum_mag, kSumFTol);
        return 1;
    }

    std::printf("PASS: gpw-forces-water  max|dF|=%.3e  |sumF|=%.3e  |dE|=%.3e  "
                "iters=%d\n",
                max_err, sum_mag, de, eval.last_result().iters);
    return 0;
}
