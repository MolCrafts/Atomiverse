/// @file regressions/gpw/ot-uks-o2/reg_ot-uks-o2.cpp
/// @brief END-TO-END: Γ-UKS OT SCF on O₂ vs offline CP2K OT pin + OT/DIIS parity.
///
/// Public API only. See REFERENCE.md. Requires ATV_BUILD_GPW (+ HF + PW).
/// Exit non-zero on failure. Target ≤ 60 s Release.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

cpu::Frame make_o2() {
    // Centered O₂ in 12 Å cube (MT); R = 1.2075 Å along z.
    cpu::Frame frame(2);
    constexpr double L = 12.0;
    constexpr double R = 1.2075;
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 8;
        frame.span(kAtom_mass)[i] = 15.999;
        frame.span(kAtom_x)[i] = 0.5 * L;
        frame.span(kAtom_y)[i] = 0.5 * L;
    }
    frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
    frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
    frame.box = cpu::Box::cubic(L);
    return frame;
}

cpu::gpw::GPWEvaluator make_eval(bool use_ot) {
    cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    // 250 Ry: CP2K OT twin + ≤60 s regression budget.
    cpu::gpw::GPWEvaluator eval(
        /*kinds*/ {O}, /*cutoff_ry*/ 250.0, /*max_iter*/ 200, /*tol*/ 1e-7,
        /*mix*/ 0.4, cpu::gpw::XcKind::PADE, /*nonperiodic*/ true,
        /*compute_forces*/ false, /*fd_step*/ 1e-3, /*d3*/ false,
        /*use_ot*/ use_ot);
    eval.n_alpha = 7;
    eval.n_beta = 5;
    return eval;
}

} // namespace

int main() {
    // Offline CP2K 2026.1 OT pin at 250 Ry — REFERENCE.md (not a self-golden).
    constexpr double kCp2kOtHa = -31.770029989852723;
    constexpr double kCp2kS2 = 2.004666;
    // 250 Ry single-grid ATV vs CP2K multigrid twin — milli-Ha class (see REFERENCE).
    constexpr double kPinTolHa = 2.5e-3;
    constexpr double kOtDiisTolHa = 1.0e-4;
    constexpr double kTolS2 = 0.15;

    unsetenv("ATV_GPW_OT");

    using clock = std::chrono::steady_clock;

    cpu::Frame frame_ot = make_o2();
    auto eval_ot = make_eval(/*use_ot=*/true);
    const auto t0 = clock::now();
    cpu::single_point(frame_ot, eval_ot);
    const auto t1 = clock::now();
    const double wall_ot = std::chrono::duration<double>(t1 - t0).count();

    if (!eval_ot.last_result().converged) {
        std::printf("FAIL: OT UKS did not converge in %d iters\n",
                    eval_ot.last_result().iters);
        return 1;
    }
    const double e_ot = eval_ot.last_energy_hartree();
    const double s2 = eval_ot.last_s2();
    if (!std::isfinite(e_ot) || !std::isfinite(s2)) {
        std::printf("FAIL: non-finite OT energy or S2\n");
        return 1;
    }
    const double d_cp2k = std::fabs(e_ot - kCp2kOtHa);
    if (!(d_cp2k <= kPinTolHa)) {
        std::printf("FAIL: |E_OT - E_CP2K|=%.6e Ha  tol=%.1e  E_OT=%.12f  "
                    "E_CP2K=%.12f\n",
                    d_cp2k, kPinTolHa, e_ot, kCp2kOtHa);
        return 1;
    }
    if (!(std::fabs(s2 - kCp2kS2) <= kTolS2)) {
        std::printf("FAIL: |S2 - CP2K|=%.3e  tol=%.2f  S2=%.8f  CP2K=%.8f\n",
                    std::fabs(s2 - kCp2kS2), kTolS2, s2, kCp2kS2);
        return 1;
    }

    cpu::Frame frame_diis = make_o2();
    auto eval_diis = make_eval(/*use_ot=*/false);
    const auto t2 = clock::now();
    cpu::single_point(frame_diis, eval_diis);
    const auto t3 = clock::now();
    const double wall_diis = std::chrono::duration<double>(t3 - t2).count();
    if (!eval_diis.last_result().converged) {
        std::printf("FAIL: DIIS UKS did not converge in %d iters\n",
                    eval_diis.last_result().iters);
        return 1;
    }
    const double e_diis = eval_diis.last_energy_hartree();
    const double d_ot_diis = std::fabs(e_ot - e_diis);
    if (!(d_ot_diis <= kOtDiisTolHa)) {
        std::printf("FAIL: |E_OT - E_DIIS|=%.6e Ha  tol=%.1e  E_OT=%.12f  "
                    "E_DIIS=%.12f\n",
                    d_ot_diis, kOtDiisTolHa, e_ot, e_diis);
        return 1;
    }

    const double e_frame = frame_ot.meta.get(kEnergy) / units::kHartreeInEv;
    if (!(std::isfinite(e_frame) && std::fabs(e_frame - e_ot) < 1e-9)) {
        std::printf("FAIL: Frame energy Ha mismatch e_frame=%.12f e_ot=%.12f\n",
                    e_frame, e_ot);
        return 1;
    }

    std::printf("ok: E_OT=%.12f Ha (%d it, %.2fs)  E_DIIS=%.12f Ha (%d it, "
                "%.2fs)\n",
                e_ot, eval_ot.last_result().iters, wall_ot, e_diis,
                eval_diis.last_result().iters, wall_diis);
    std::printf("     |dE|_CP2K=%.3e  |dE|_OT-DIIS=%.3e  S2=%.8f  pin=%.12f "
                "Ha\n",
                d_cp2k, d_ot_diis, s2, kCp2kOtHa);
    std::printf("PASS: gpw ot-uks-o2 (OT vs CP2K + OT/DIIS parity)\n");
    return 0;
}
