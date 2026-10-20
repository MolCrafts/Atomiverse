/// @file regressions/gpw/periodic-ugw-si/reg_periodic-ugw-si.cpp
/// @brief END-TO-END: Γ-point unrestricted spectral G0W0 (UGW@UKS) on
///        periodic Si₂ (DZVP-MOLOPT-GTH + GTH-PBE-q4, UKS PBE reference).
///        See REFERENCE.md.
///
/// Public API only (`GpwDft::run` + `GpwDft::g0w0`). Requires ATV_BUILD_GPW
/// (+ HF + PW). Exit non-zero on mismatch. <= 60 s, single process. No live
/// CP2K — offline-pin contract per tools/cp2k_oracle/README.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/g0w0.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV::cpu::gpw;
    constexpr double a_ang = 5.43;

    // Dual-gate vs closed-shell G0W0 first-green on the same deck
    // (UKS nα=nβ → UGW ≡ RGW when orbitals match). REFERENCE.md.
    constexpr double kGapRefEv = 5.719545; // eV (periodic-gw-si first-green)
    constexpr double kGapTolEv = 0.05;     // eV (setup-integrity band)

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
                   GthPotential(molrs::Element::Si, "GTH-PBE-q4")}}};
    m.cutoff_ry = 30.0;
    m.n_alpha = 4;
    m.n_beta = 4;
    // Closed-shell Si₂ π² has an exactly degenerate HOMO in the core-H
    // guess; last-bit noise picks one of two stationary points
    // (ΔE 4e-5 Ha, both inside every gate below) reached in ~60 or
    // ~110-135 level-shifted iterations. See REFERENCE.md.
    m.max_iter = 200;
    m.tol = 1e-6;
    m.mix = 0.3;
    m.level_shift = 0.3;
    m.xc = XcKind::PBE;
    m.mp2_ri = true;
    m.build(frame);

    const auto t0 = std::chrono::steady_clock::now();
    const GpwResult& scf = m.run(frame);
    const auto t1 = std::chrono::steady_clock::now();
    if (!scf.converged || !std::isfinite(scf.e_total)) {
        std::fprintf(stderr,
                     "FAIL: periodic Γ UKS-PBE SCF did not converge "
                     "(iters=%d E=%.8f)\n",
                     scf.iters, scf.e_total);
        return 1;
    }
    if (scf.C_beta.size() == 0 || scf.eps_beta.empty()) {
        std::fprintf(stderr, "FAIL: UKS result missing β orbitals\n");
        return 1;
    }

    m.mp2_n_virt = 0;
    m.mp2_ri = true;
    const G0W0Result& gw = m.g0w0();
    const auto t2 = std::chrono::steady_clock::now();

    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();
    const double wall_gw = std::chrono::duration<double>(t2 - t1).count();

    if (!gw.ri) {
        std::fprintf(stderr, "FAIL: G0W0Result.ri not set on product path\n");
        return 1;
    }
    if (!gw.converged) {
        std::fprintf(stderr, "FAIL: UGW Newton QP solve did not converge\n");
        return 1;
    }
    if (gw.n_occ != 4) {
        std::fprintf(stderr, "FAIL: n_occ=%d want 4 (n_α primary report)\n",
                     gw.n_occ);
        return 1;
    }
    if (static_cast<int>(gw.qp_energy_ha.size())
        != static_cast<int>(gw.mf_energy_ha.size())) {
        std::fprintf(stderr, "FAIL: qp/mf length mismatch\n");
        return 1;
    }
    if (!(gw.gap_ev > 0.0) || !std::isfinite(gw.gap_ev)) {
        std::fprintf(stderr, "FAIL: QP gap=%.6f eV not finite/positive\n",
                     gw.gap_ev);
        return 1;
    }
    if (!(gw.mf_gap_ev > 0.0) || !std::isfinite(gw.mf_gap_ev)) {
        std::fprintf(stderr, "FAIL: KS gap=%.6f eV not finite/positive\n",
                     gw.mf_gap_ev);
        return 1;
    }

    const int homo = gw.n_occ - 1;
    const int lumo = gw.n_occ;
    const double e_h = gw.qp_energy_ha[static_cast<std::size_t>(homo)];
    const double e_l = gw.qp_energy_ha[static_cast<std::size_t>(lumo)];
    if (!(e_h < e_l)) {
        std::fprintf(stderr,
                     "FAIL: ε_HOMO^QP=%.8f not below ε_LUMO^QP=%.8f Ha\n", e_h,
                     e_l);
        return 1;
    }
    const double d_h =
        std::abs(e_h - gw.mf_energy_ha[static_cast<std::size_t>(homo)]);
    const double d_l =
        std::abs(e_l - gw.mf_energy_ha[static_cast<std::size_t>(lumo)]);
    if (!std::isfinite(d_h) || !std::isfinite(d_l)) {
        std::fprintf(stderr, "FAIL: non-finite QP shifts\n");
        return 1;
    }

    if (std::abs(gw.gap_ev - kGapRefEv) > kGapTolEv) {
        std::fprintf(stderr,
                     "FAIL: QP gap got=%.6f want~%.6f +/- %.3f eV "
                     "(UGW dual-gate vs periodic-gw-si; see REFERENCE.md)\n",
                     gw.gap_ev, kGapRefEv, kGapTolEv);
        return 1;
    }

    std::printf(
        "PASS: Γ UGW@UKS-PBE Si2 (DZVP-MOLOPT-GTH, GTH-PBE-q4, 30 Ry)\n"
        "  E_SCF     = %.10f Ha (iters=%d, <S2>=%.6f, wall=%.1f s)\n"
        "  KS gap α  = %.6f eV\n"
        "  QP gap α  = %.6f eV  (IP=%.4f eV, EA=%.4f eV)\n"
        "  ε_HOMO^QP = %.8f Ha  (Δ=%.6f Ha vs KS)\n"
        "  ε_LUMO^QP = %.8f Ha  (Δ=%.6f Ha vs KS)\n"
        "  n_occ(α)=%d n_virt(α+β)=%d  UGW wall=%.1f s  total=%.1f s\n"
        "  [dual-gate |gap−%.6f|≤%.3f eV vs RKS G0W0; see REFERENCE.md]\n",
        scf.e_total, scf.iters, scf.s2, wall_scf, gw.mf_gap_ev, gw.gap_ev,
        gw.ip_ev, gw.ea_ev, e_h, d_h, e_l, d_l, gw.n_occ, gw.n_virt, wall_gw,
        wall_scf + wall_gw, kGapRefEv, kGapTolEv);
    return 0;
}
