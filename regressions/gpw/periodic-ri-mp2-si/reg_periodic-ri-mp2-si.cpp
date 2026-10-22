/// @file regressions/gpw/periodic-ri-mp2-si/reg_periodic-ri-mp2-si.cpp
/// @brief END-TO-END: Γ-point RI-GPW MP2 product path on periodic Si₂
///        (DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE reference). See REFERENCE.md.
///
/// Public API only (`GpwDft::run` + `GpwDft::mp2` with `mp2_ri`). Requires
/// ATV_BUILD_GPW (+ HF + PW). Exit non-zero on mismatch. <= 60 s, single
/// process. No live CP2K — offline-pin contract per tools/cp2k_oracle/README.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/mp2.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV::cpu::gpw;
    constexpr double a_ang = 5.43;

    // Offline CP2K 2026.1 MP2_GPW pin (REFERENCE.md). Identity RI≡canonical
    // is 1e-10 Ha (same Poisson kernel); CP2K band is 1e-3 Ha (grid/SCF twin).
    constexpr double kEcorrRef = -0.27608774065939; // Ha (CP2K MP2_GPW)
    constexpr double kEcorrTol = 1.0e-3;            // Ha
    constexpr double kRiCanonTol = 1.0e-10;         // Ha  RI vs Poisson path
    constexpr double kIdTol = 1.0e-12;

    ATV::cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        const double q = (i == 0) ? 0.0 : a_ang / 4.0;
        frame.span(ATV::kAtom_type)[i] = 14;
        frame.span(ATV::kAtom_x)[i] = q;
        frame.span(ATV::kAtom_y)[i] = q;
        frame.span(ATV::kAtom_z)[i] = q;
    }
    frame.box = ATV::cpu::Box::cubic(a_ang);

    GpwDft m{{Kind{molrs::Element::Si,
                   GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
                   GthPotential(molrs::Element::Si, "GTH-PBE-q4")}},
             30.0,
             XcKind::PBE};
    m.n_occ = 4;
    // Closed-shell Si₂ π² has an exactly degenerate HOMO in the core-H
    // guess; last-bit noise picks one of two stationary points
    // (ΔE 4e-5 Ha, both inside every gate below) reached in ~60 or
    // ~110-135 level-shifted iterations. See REFERENCE.md.
    m.max_iter = 200;
    m.tol = 1e-6;
    m.mix = 0.3;
    m.level_shift = 0.3;
    m.mp2_ri = true; // product default
    m.build(frame);

    const auto t0 = std::chrono::steady_clock::now();
    const GpwResult& scf = m.run(frame);
    const auto t1 = std::chrono::steady_clock::now();
    if (!scf.converged || !std::isfinite(scf.e_total)) {
        std::fprintf(stderr,
                     "FAIL: periodic Γ GPW PBE SCF did not converge "
                     "(iters=%d E=%.8f)\n",
                     scf.iters, scf.e_total);
        return 1;
    }

    // Product RI-GPW path (full virtual space).
    m.mp2_n_virt = 0;
    m.mp2_ri = true;
    const Mp2Result r_ri = m.mp2();
    const auto t2 = std::chrono::steady_clock::now();

    // Foothold Poisson path on the same orbitals — must match to 1e-6 Ha.
    m.mp2_ri = false;
    const Mp2Result r_can = m.mp2();
    const auto t3 = std::chrono::steady_clock::now();

    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();
    const double wall_ri = std::chrono::duration<double>(t2 - t1).count();
    const double wall_can = std::chrono::duration<double>(t3 - t2).count();

    if (!r_ri.ri) {
        std::fprintf(stderr, "FAIL: Mp2Result.ri not set on product path\n");
        return 1;
    }
    if (r_can.ri) {
        std::fprintf(stderr,
                     "FAIL: Mp2Result.ri set on Poisson foothold path\n");
        return 1;
    }
    if (!std::isfinite(r_ri.e_corr) || !(r_ri.e_corr < 0.0)) {
        std::fprintf(stderr,
                     "FAIL: E_corr(RI-MP2)=%.10f Ha not finite/negative\n",
                     r_ri.e_corr);
        return 1;
    }
    if (std::abs(r_ri.e_corr_os + r_ri.e_corr_ss - r_ri.e_corr) > kIdTol) {
        std::fprintf(stderr,
                     "FAIL: OS+SS decomposition broken: os=%.12f ss=%.12f "
                     "corr=%.12f Ha\n",
                     r_ri.e_corr_os, r_ri.e_corr_ss, r_ri.e_corr);
        return 1;
    }
    if (std::abs(r_ri.e_total - (r_ri.e_scf + r_ri.e_corr)) > kIdTol) {
        std::fprintf(stderr,
                     "FAIL: e_total != e_scf + e_corr (%.12f vs %.12f)\n",
                     r_ri.e_total, r_ri.e_scf + r_ri.e_corr);
        return 1;
    }
    const double d_ri_can = std::abs(r_ri.e_corr - r_can.e_corr);
    if (d_ri_can > kRiCanonTol) {
        std::fprintf(stderr,
                     "FAIL: RI vs canonical |dE_corr|=%.3e Ha > %.1e "
                     "(RI=%.12f can=%.12f)\n",
                     d_ri_can, kRiCanonTol, r_ri.e_corr, r_can.e_corr);
        return 1;
    }
    if (std::abs(r_ri.e_corr - kEcorrRef) > kEcorrTol) {
        std::fprintf(stderr,
                     "FAIL: E_corr got=%.10f want=%.10f +/- %.1e Ha "
                     "(offline pin band; see REFERENCE.md)\n",
                     r_ri.e_corr, kEcorrRef, kEcorrTol);
        return 1;
    }

    std::printf(
        "PASS: Γ RI-GPW MP2 Si2 (DZVP-MOLOPT-GTH, GTH-PBE-q4, PBE, 30 Ry)\n"
        "  E_SCF   = %.10f Ha (iters=%d, wall=%.1f s)\n"
        "  E_corr  = %.10f Ha (OS=%.10f SS=%.10f; n_occ=%d n_virt=%d, "
        "RI wall=%.1f s)\n"
        "  E_total = %.10f Ha   |RI−can|=%.3e Ha (can wall=%.1f s)\n"
        "  [RI≡Poisson + first-green/offline pin; see REFERENCE.md]\n",
        r_ri.e_scf, scf.iters, wall_scf, r_ri.e_corr, r_ri.e_corr_os,
        r_ri.e_corr_ss, r_ri.n_occ, r_ri.n_virt, wall_ri, r_ri.e_total,
        d_ri_can, wall_can);
    return 0;
}
