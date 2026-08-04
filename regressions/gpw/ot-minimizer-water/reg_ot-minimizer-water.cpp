/// @file regressions/gpw/ot-minimizer-water/reg_ot-minimizer-water.cpp
/// @brief END-TO-END: GPW OT SCF on isolated water vs offline CP2K OT pin.
///
/// Public API only. See REFERENCE.md. Requires ATV_BUILD_GPW (+ HF + PW).

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

namespace {

/// CP2K getting-started h2o geometry [Å] in a 10 Å cubic cell (MT Poisson).
cpu::Frame make_water() {
    cpu::Frame f(3);
    const int z[3] = {8, 1, 1};
    const double x[3] = {5.0000, 5.7586, 4.2414};
    const double y[3] = {5.0000, 5.0000, 5.0000};
    const double zc[3] = {5.0000, 5.5043, 5.5043};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_type)[i] = z[i];
        f.span(kAtom_x)[i] = x[i];
        f.span(kAtom_y)[i] = y[i];
        f.span(kAtom_z)[i] = zc[i];
    }
    f.box = cpu::Box::cubic(10.0);
    return f;
}

cpu::gpw::GPWEvaluator make_eval(bool use_ot) {
    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};
    // Cutoff 250 Ry. Single-grid (ngrids=1): product multigrid is a net loss
    // on this 3-atom / 10 Å cell (product list overhead > collocate win).
    // CP2K twin uses NGRIDS 4 — energy pin tol absorbs the residual.
    cpu::gpw::GPWEvaluator eval(
        /*kinds*/ {O, H}, /*cutoff_ry*/ 250.0, /*max_iter*/ 200, /*tol*/ 1e-7,
        /*mix*/ 0.3, cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
        /*compute_forces*/ false, /*fd_step*/ 1e-3, /*d3*/ false,
        /*use_ot*/ use_ot);
    return eval;
}

} // namespace

int main() {
    // Offline CP2K 2026.1 OT pin at 250 Ry / NGRIDS 4 / REL_CUTOFF 40.
    // See REFERENCE.md (not a self-golden).
    constexpr double kCp2kOtHa = -17.212848291913666;
    constexpr double kPinTolHa = 1.5e-3;
    constexpr double kOtDiisTolHa = 1.0e-4;

    unsetenv("ATV_GPW_OT"); // product path is the member, not the env hack

    cpu::Frame frame_ot = make_water();
    auto eval_ot = make_eval(/*use_ot=*/true);
    cpu::single_point(frame_ot, eval_ot);
    if (!eval_ot.last_result().converged) {
        std::printf("FAIL: OT SCF did not converge in %d iters\n",
                    eval_ot.last_result().iters);
        return 1;
    }
    const double e_ot = eval_ot.last_energy_hartree();
    if (!std::isfinite(e_ot)) {
        std::printf("FAIL: non-finite OT energy\n");
        return 1;
    }
    const double d_cp2k = std::abs(e_ot - kCp2kOtHa);
    if (!(d_cp2k <= kPinTolHa)) {
        std::printf("FAIL: |E_OT - E_CP2K|=%.6e Ha  tol=%.1e  E_OT=%.10f  "
                    "E_CP2K=%.10f\n",
                    d_cp2k, kPinTolHa, e_ot, kCp2kOtHa);
        return 1;
    }

    cpu::Frame frame_diis = make_water();
    auto eval_diis = make_eval(/*use_ot=*/false);
    cpu::single_point(frame_diis, eval_diis);
    if (!eval_diis.last_result().converged) {
        std::printf("FAIL: DIIS SCF did not converge in %d iters\n",
                    eval_diis.last_result().iters);
        return 1;
    }
    const double e_diis = eval_diis.last_energy_hartree();
    const double d_ot_diis = std::abs(e_ot - e_diis);
    if (!(d_ot_diis <= kOtDiisTolHa)) {
        std::printf("FAIL: |E_OT - E_DIIS|=%.6e Ha  tol=%.1e  E_OT=%.10f  "
                    "E_DIIS=%.10f\n",
                    d_ot_diis, kOtDiisTolHa, e_ot, e_diis);
        return 1;
    }

    std::printf("ok: E_OT=%.10f Ha (%d it)  E_DIIS=%.10f Ha (%d it)\n", e_ot,
                eval_ot.last_result().iters, e_diis,
                eval_diis.last_result().iters);
    std::printf("     |dE|_CP2K=%.3e  |dE|_OT-DIIS=%.3e  pin=%.10f Ha\n",
                d_cp2k, d_ot_diis, kCp2kOtHa);
    std::printf("PASS: gpw ot-minimizer-water (OT vs CP2K + OT/DIIS parity)\n");
    return 0;
}
