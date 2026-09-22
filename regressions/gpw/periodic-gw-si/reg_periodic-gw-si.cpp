/// @file regressions/gpw/periodic-gw-si/reg_periodic-gw-si.cpp
/// @brief END-TO-END: Γ-point GPW spectral G0W0 on periodic Si₂
///        (DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE reference). See REFERENCE.md.
///
/// Public API only (`GpwDft::run` + `GpwDft::g0w0`). Requires
/// ATV_BUILD_GPW (+ HF + PW). Exit non-zero on mismatch. <= 60 s, single
/// process. No live CP2K — offline-pin contract per tools/cp2k_oracle/README.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/g0w0.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV::cpu::gpw;
    constexpr double kA2B = 1.8897259885789;
    constexpr double a_bohr = 5.43 * kA2B;

    // First-green Atomiverse pin (REFERENCE.md). Setup-integrity band 0.05 eV
    // until an offline CP2K spectral twin is available.
    constexpr double kGapRefEv = 5.719545; // eV (first-green 2026-07-25)
    constexpr double kGapTolEv = 0.05;     // eV

    const double pos0[3] = {0.0, 0.0, 0.0};
    const double pos1[3] = {a_bohr / 4.0, a_bohr / 4.0, a_bohr / 4.0};
    const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
    const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
    std::vector<GpwAtom> atoms;
    atoms.emplace_back(pos0, bas, pot, 14);
    atoms.emplace_back(pos1, bas, pot, 14);

    GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = a_bohr;
    m.cutoff_ry = 30.0;
    m.n_occ = 4;
    m.max_iter = 80;
    m.tol = 1e-6;
    m.mix = 0.3;
    m.level_shift = 0.3;
    m.xc = XcKind::PBE;
    m.mp2_ri = true;
    m.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const GpwResult& scf = m.run(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    if (!scf.converged || !std::isfinite(scf.e_total)) {
        std::fprintf(stderr,
                     "FAIL: periodic Γ GPW PBE SCF did not converge "
                     "(iters=%d E=%.8f)\n",
                     scf.iters, scf.e_total);
        return 1;
    }

    m.mp2_n_virt = 0;
    m.mp2_ri = true;
    const G0W0Result& gw = m.g0w0(atoms);
    const auto t2 = std::chrono::steady_clock::now();

    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();
    const double wall_gw = std::chrono::duration<double>(t2 - t1).count();

    if (!gw.ri) {
        std::fprintf(stderr, "FAIL: G0W0Result.ri not set on product path\n");
        return 1;
    }
    if (!gw.converged) {
        std::fprintf(stderr, "FAIL: G0W0 Newton QP solve did not converge\n");
        return 1;
    }
    if (gw.n_occ != 4) {
        std::fprintf(stderr, "FAIL: n_occ=%d want 4\n", gw.n_occ);
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
                     "(see REFERENCE.md; update pin after first green)\n",
                     gw.gap_ev, kGapRefEv, kGapTolEv);
        return 1;
    }

    std::printf(
        "PASS: Γ G0W0@PBE Si2 (DZVP-MOLOPT-GTH, GTH-PBE-q4, PBE, 30 Ry)\n"
        "  E_SCF     = %.10f Ha (iters=%d, wall=%.1f s)\n"
        "  KS gap    = %.6f eV\n"
        "  QP gap    = %.6f eV  (IP=%.4f eV, EA=%.4f eV)\n"
        "  ε_HOMO^QP = %.8f Ha  (Δ=%.6f Ha vs KS)\n"
        "  ε_LUMO^QP = %.8f Ha  (Δ=%.6f Ha vs KS)\n"
        "  n_occ=%d n_virt=%d  G0W0 wall=%.1f s  total=%.1f s\n"
        "  [pin band |gap−%.6f|≤%.3f eV; see REFERENCE.md]\n",
        scf.e_total, scf.iters, wall_scf, gw.mf_gap_ev, gw.gap_ev, gw.ip_ev,
        gw.ea_ev, e_h, d_h, e_l, d_l, gw.n_occ, gw.n_virt, wall_gw,
        wall_scf + wall_gw, kGapRefEv, kGapTolEv);
    return 0;
}
