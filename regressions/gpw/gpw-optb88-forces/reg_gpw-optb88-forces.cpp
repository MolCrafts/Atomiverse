/// @file regressions/gpw/gpw-optb88-forces/reg_gpw-optb88-forces.cpp
/// @brief END-TO-END: Γ GPW OptB88+DRSLL analytic ionic forces vs offline
///        CP2K 2026.1 ENERGY_FORCE twin + OptB88 plain dual-gate.
///
/// Public API only. Hard pins in REFERENCE.md (offline CP2K; never live in
/// ctest). Fair wall = OptB88+DRSLL SCF+force only vs CP2K twin TOTAL TIME.

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

using namespace ATV;

namespace {

// Offline CP2K 2026.1 ENERGY_FORCE pin (REFERENCE.md).
constexpr double kEDispCp2k = 0.15206184251169;   // Ha
constexpr double kETotCp2k = -34.487914413113714; // Ha FORCE_EVAL QS
constexpr double kCp2kWall = 17.509;              // s, CP2K timer TOTAL
constexpr double kEDispTol = 1.0e-4;              // Ha
constexpr double kETotTol = 2.5e-2;               // Ha (OptB88 multigrid class)
constexpr double kForceTol = 1.0e-2;              // Ha/Bohr
constexpr double kSumFTol = 2.0e-2;               // Ha/Bohr (oracle residual)
constexpr double kDualEFloor = 1.0e-4;            // Ha
constexpr double kForce[6][3] = {
    {-7.33537750e-02, 4.42763551e-08, -3.00361565e-08},
    {3.93194959e-02, -3.97828002e-09, 1.31800024e-01},
    {3.93194286e-02, -3.13656238e-08, -1.31799911e-01},
    {-6.85304811e-02, -2.97383508e-08, 1.84114929e-08},
    {3.65105370e-02, 3.88727769e-08, 1.35580659e-01},
    {3.65106052e-02, -1.71698035e-08, -1.35581298e-01},
};

cpu::Frame make_water_dimer() {
    cpu::Frame frame(6);
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double x[6] = {0.0, 0.758602, 0.758602, 3.0, 3.758602, 3.758602};
    const double y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double zc[6] = {0.0, 0.504284, -0.504284, 0.0, 0.504284, -0.504284};
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = x[i];
        frame.span(kAtom_y)[i] = y[i];
        frame.span(kAtom_z)[i] = zc[i];
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    }
    frame.box = cpu::Box::cubic(12.0);
    return frame;
}

cpu::gpw::GPWEvaluator make_eval(bool vdw_df) {
    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::gpw::GPWEvaluator eval(
        {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 80, /*tol*/ 1e-7,
        /*mix*/ 0.4, cpu::gpw::XcKind::OptB88VdW, /*nonperiodic*/ false,
        /*compute_forces*/ true);
    // Match CP2K MGRID (NGRIDS 4 / REL_CUTOFF 40) — fair twin for E/F/wall.
    eval.ngrids = 4;
    eval.rel_cutoff = 40.0;
    eval.vdw_df = vdw_df;
    return eval;
}

void run_eval(cpu::Frame& frame, cpu::gpw::GPWEvaluator& eval) {
    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);
}

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    // ── (1) OptB88 + DRSLL ENERGY_FORCE twin (timed for fair wall) ────────
    cpu::Frame frame = make_water_dimer();
    cpu::gpw::GPWEvaluator eval = make_eval(/*vdw_df=*/true);

    const auto t0 = clock::now();
    run_eval(frame, eval);
    const auto t1 = clock::now();
    const double wall_atv = std::chrono::duration<double>(t1 - t0).count();

    if (!eval.last_result().converged) {
        std::printf("FAIL: OptB88+DRSLL SCF did not converge (iters=%d)\n",
                    eval.last_result().iters);
        return 1;
    }

    const double e_disp = eval.last_result().e_disp;
    const double e_ha = eval.last_energy_hartree();
    if (!std::isfinite(e_disp) || !std::isfinite(e_ha)) {
        std::printf("FAIL: non-finite E_disp=%.14e E_tot=%.14e\n", e_disp,
                    e_ha);
        return 1;
    }
    if (!(std::abs(e_disp) > 0.0)) {
        std::printf("FAIL: E_disp is zero (vdw_df inactive?)\n");
        return 1;
    }
    if (std::abs(e_disp - kEDispCp2k) > kEDispTol) {
        std::printf("FAIL: |E_disp-cp2k|=%.6e Ha  got=%.14e  want=%.14e "
                    "tol=%.1e\n",
                    std::abs(e_disp - kEDispCp2k), e_disp, kEDispCp2k,
                    kEDispTol);
        return 1;
    }
    if (std::abs(e_ha - kETotCp2k) > kETotTol) {
        std::printf("FAIL: |E_tot-cp2k|=%.6e Ha  got=%.14e  want=%.14e "
                    "tol=%.1e\n",
                    std::abs(e_ha - kETotCp2k), e_ha, kETotCp2k, kETotTol);
        return 1;
    }

    const double to_ha_bohr = units::kEvPerAngstromInHartreePerBohr;
    double max_err = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    double F_ha[6][3];
    for (int a = 0; a < 6; ++a) {
        F_ha[a][0] = static_cast<double>(frame.span(kAtom_fx)[a]) * to_ha_bohr;
        F_ha[a][1] = static_cast<double>(frame.span(kAtom_fy)[a]) * to_ha_bohr;
        F_ha[a][2] = static_cast<double>(frame.span(kAtom_fz)[a]) * to_ha_bohr;
        for (int k = 0; k < 3; ++k) {
            sum[k] += F_ha[a][k];
            const double err = std::abs(F_ha[a][k] - kForce[a][k]);
            max_err = std::max(max_err, err);
            std::printf("  atom %d k %d  F_atv=% .8e  F_cp2k=% .8e  |d|=%.3e\n",
                        a, k, F_ha[a][k], kForce[a][k], err);
        }
    }
    const double sum_mag =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);

    std::printf("E_disp=%.14e Ha (|d|=%.2e)  E_tot=%.14e Ha (|d|=%.2e)\n",
                e_disp, std::abs(e_disp - kEDispCp2k), e_ha,
                std::abs(e_ha - kETotCp2k));
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

    // ── (2) Dual gate: OptB88 plain (no DRSLL) must shift energy ──────────
    cpu::Frame frame_plain = make_water_dimer();
    cpu::gpw::GPWEvaluator eval_plain = make_eval(/*vdw_df=*/false);
    run_eval(frame_plain, eval_plain);
    if (!eval_plain.last_result().converged) {
        std::printf("FAIL: OptB88 plain SCF did not converge (iters=%d)\n",
                    eval_plain.last_result().iters);
        return 1;
    }
    const double e_plain = eval_plain.last_energy_hartree();
    const double de_dual = std::abs(e_ha - e_plain);
    if (!(de_dual > kDualEFloor)) {
        std::printf("FAIL: OptB88+DRSLL vs plain dual |dE|=%.3e Ha "
                    "(need > %.1e); E_vdw=%.14e E_plain=%.14e\n",
                    de_dual, kDualEFloor, e_ha, e_plain);
        return 1;
    }

    const double ratio = wall_atv / kCp2kWall;
    std::printf(
        "PASS: gpw-optb88-forces  max|dF|=%.3e  |sumF|=%.3e  |dE_disp|=%.2e  "
        "|dE_tot|=%.2e  dual|dE|=%.3e  wall_atv=%.3f s  T_cp2k=%.3f s  "
        "ratio=%.3f  iters=%d\n",
        max_err, sum_mag, std::abs(e_disp - kEDispCp2k),
        std::abs(e_ha - kETotCp2k), de_dual, wall_atv, kCp2kWall, ratio,
        eval.last_result().iters);

    if (!(ratio <= 10.0)) {
        std::printf("FAIL: wall ratio T_atv/T_cp2k=%.3f > 10 "
                    "(atv=%.3f s cp2k=%.3f s)\n",
                    ratio, wall_atv, kCp2kWall);
        return 1;
    }
    return 0;
}
