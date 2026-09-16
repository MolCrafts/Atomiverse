/// @file regressions/gpw/gpw-blyp-water/reg_gpw-blyp-water.cpp
/// @brief END-TO-END: isolated H2O GPW BLYP / DZVP-MOLOPT-GTH (MT) dual-gate
///        vs PBE + absolute soft/CP2K pin. See REFERENCE.md.
///
/// Public API only. Requires ATV_BUILD_GPW (+ HF + PW). Exit non-zero on
/// mismatch. ≤ 60 s, single CPU. No live CP2K.

#include <chrono>
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

ATV::cpu::Frame make_water_mt() {
    // CP2K getting-started h2o.inp geometry [Å], 10 Å cubic cell.
    ATV::cpu::Frame frame(3);
    const int z[3] = {8, 1, 1};
    const double x[3] = {5.0000, 5.7586, 4.2414};
    const double y[3] = {5.0000, 5.0000, 5.0000};
    const double zc[3] = {5.0000, 5.5043, 5.5043};
    for (int i = 0; i < 3; ++i) {
        frame.span(ATV::kAtom_type)[i] = z[i];
        frame.span(ATV::kAtom_x)[i] = x[i];
        frame.span(ATV::kAtom_y)[i] = y[i];
        frame.span(ATV::kAtom_z)[i] = zc[i];
    }
    frame.box = ATV::cpu::Box::cubic(10.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    // Dual-gate floor [Ha] — BLYP vs PBE must activate a different functional.
    constexpr double kDeFloor = 1.0e-4;
    // Absolute first-green ATV pin (Ha). Offline CP2K twin
    // tools/cp2k_oracle/inputs/gpw-blyp-water.inp replaces this when available.
    // See REFERENCE.md.
    constexpr double kEnergyRef = -17.1982170663; // Ha
    constexpr double kTol = 1.0e-3;               // Ha

    cpu::Frame frame_blyp = make_water_mt();
    cpu::gpw::GPWEvaluator eval_blyp(
        /*kinds*/ {O, H}, /*cutoff_ry*/ 400.0, /*max_iter*/ 200, /*tol*/ 1e-7,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::BLYP, /*nonperiodic*/ true);

    const auto t0 = std::chrono::steady_clock::now();
    cpu::single_point(frame_blyp, eval_blyp);
    const auto t1 = std::chrono::steady_clock::now();

    cpu::Frame frame_pbe = make_water_mt();
    cpu::gpw::GPWEvaluator eval_pbe(
        /*kinds*/ {O, H}, /*cutoff_ry*/ 400.0, /*max_iter*/ 200, /*tol*/ 1e-7,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true);
    cpu::single_point(frame_pbe, eval_pbe);
    const auto t2 = std::chrono::steady_clock::now();

    const double e_blyp = eval_blyp.last_energy_hartree();
    const double e_pbe = eval_pbe.last_energy_hartree();
    const bool conv_blyp = eval_blyp.last_result().converged;
    const bool conv_pbe = eval_pbe.last_result().converged;
    const double wall_blyp = std::chrono::duration<double>(t1 - t0).count();
    const double wall_pbe = std::chrono::duration<double>(t2 - t1).count();

    if (!conv_blyp) {
        std::fprintf(stderr,
                     "FAIL: GPW H2O BLYP/MT did not converge (iters=%d)\n",
                     eval_blyp.last_result().iters);
        return 1;
    }
    if (!std::isfinite(e_blyp)) {
        std::fprintf(stderr, "FAIL: non-finite E_BLYP\n");
        return 1;
    }
    if (!conv_pbe || !std::isfinite(e_pbe)) {
        std::fprintf(stderr,
                     "FAIL: GPW H2O PBE twin did not converge/finite "
                     "(iters=%d E=%.8f)\n",
                     eval_pbe.last_result().iters, e_pbe);
        return 1;
    }

    const double de = std::abs(e_blyp - e_pbe);
    if (!(de > kDeFloor)) {
        std::fprintf(stderr,
                     "FAIL: |E_BLYP−E_PBE|=%.6e Ha  floor=%.1e "
                     "(BLYP inactive?)\n",
                     de, kDeFloor);
        return 1;
    }
    if (std::abs(e_blyp - kEnergyRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_BLYP got=%.10f want~%.6f +/- %.1e Ha "
                     "(update pin in REFERENCE after first green)\n",
                     e_blyp, kEnergyRef, kTol);
        return 1;
    }
    if (!std::isfinite(frame_blyp.meta.get(kEnergy))) {
        std::fprintf(stderr, "FAIL: non-finite frame.meta[kEnergy]\n");
        return 1;
    }

    std::printf("PASS: E_GPW(H2O,BLYP,DZVP-MOLOPT-GTH,MT,400Ry) = %.10f Ha  "
                "E_PBE = %.10f Ha  |dE|=%.3e Ha  iters_blyp=%d  "
                "wall_blyp=%.1fs wall_pbe=%.1fs  "
                "[dual-gate + soft absolute; offline CP2K twin deferred]\n",
                e_blyp, e_pbe, de, eval_blyp.last_result().iters, wall_blyp,
                wall_pbe);
    return 0;
}
