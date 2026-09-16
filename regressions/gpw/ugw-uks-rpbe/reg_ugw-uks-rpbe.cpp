/// @file regressions/gpw/ugw-uks-rpbe/reg_ugw-uks-rpbe.cpp
/// @brief END-TO-END: Γ-point unrestricted spectral G0W0 (UGW@UKS) on
///        periodic Si₂ with **RPBE** spin XC (DZVP-MOLOPT-GTH +
///        GTH-PBE-q4). See REFERENCE.md.
///
/// Public API only (`GpwDft::run` + `GpwDft::g0w0`). Requires ATV_BUILD_GPW
/// (+ HF + PW). Exit non-zero on mismatch. <= 60 s, single process. No live
/// CP2K — offline-pin contract per tools/cp2k_oracle/README.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/g0w0.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::G0W0Result;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GpwResult;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::XcKind;

std::vector<GpwAtom> make_si2(double a_bohr) {
    const double pos0[3] = {0.0, 0.0, 0.0};
    const double pos1[3] = {a_bohr / 4.0, a_bohr / 4.0, a_bohr / 4.0};
    const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
    const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
    std::vector<GpwAtom> atoms;
    atoms.emplace_back(pos0, bas, pot, 14);
    atoms.emplace_back(pos1, bas, pot, 14);
    return atoms;
}

void configure_common(GpwDft& m, double a_bohr) {
    m.cell[0] = m.cell[1] = m.cell[2] = a_bohr;
    m.cutoff_ry = 30.0;
    m.max_iter = 80;
    m.tol = 1e-6;
    m.mix = 0.3;
    m.level_shift = 0.3;
    m.xc = XcKind::RPBE;
    m.mp2_ri = true;
}

} // namespace

int main() {
    constexpr double kA2B = 1.8897259885789;
    constexpr double a_bohr = 5.43 * kA2B;

    // Dual-gate: UGW@UKS-RPBE vs RKS G0W0@RPBE on the same deck when
    // nα=nβ (UGW ≡ RGW). REFERENCE.md records first-green digits.
    constexpr double kGapTolEv = 0.05; // eV
    // PBE UGW first-green of reg_periodic-ugw-si — RPBE must differ.
    constexpr double kPbeUgwGapEv = 5.719545;
    constexpr double kDeFloorEv = 1.0e-3; // eV

    auto atoms = make_si2(a_bohr);

    // ── UKS RPBE + UGW ────────────────────────────────────────────────
    GpwDft m_u;
    configure_common(m_u, a_bohr);
    m_u.n_alpha = 4;
    m_u.n_beta = 4;
    m_u.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const GpwResult& scf_u = m_u.run(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    if (!scf_u.converged || !std::isfinite(scf_u.e_total)) {
        std::fprintf(stderr,
                     "FAIL: Γ UKS-RPBE SCF did not converge "
                     "(iters=%d E=%.8f)\n",
                     scf_u.iters, scf_u.e_total);
        return 1;
    }
    if (scf_u.C_beta.size() == 0 || scf_u.eps_beta.empty()) {
        std::fprintf(stderr, "FAIL: UKS result missing β orbitals\n");
        return 1;
    }

    m_u.mp2_n_virt = 0;
    m_u.mp2_ri = true;
    const G0W0Result& gw_u = m_u.g0w0(atoms);
    const auto t2 = std::chrono::steady_clock::now();

    const double wall_scf_u = std::chrono::duration<double>(t1 - t0).count();
    const double wall_gw_u = std::chrono::duration<double>(t2 - t1).count();

    if (!gw_u.ri) {
        std::fprintf(stderr, "FAIL: G0W0Result.ri not set on product path\n");
        return 1;
    }
    if (!gw_u.converged) {
        std::fprintf(stderr, "FAIL: UGW Newton QP solve did not converge\n");
        return 1;
    }
    if (gw_u.n_occ != 4) {
        std::fprintf(stderr, "FAIL: n_occ=%d want 4 (n_α primary report)\n",
                     gw_u.n_occ);
        return 1;
    }
    if (!(gw_u.gap_ev > 0.0) || !std::isfinite(gw_u.gap_ev)) {
        std::fprintf(stderr, "FAIL: QP gap=%.6f eV not finite/positive\n",
                     gw_u.gap_ev);
        return 1;
    }
    if (!(gw_u.mf_gap_ev > 0.0) || !std::isfinite(gw_u.mf_gap_ev)) {
        std::fprintf(stderr, "FAIL: KS gap=%.6f eV not finite/positive\n",
                     gw_u.mf_gap_ev);
        return 1;
    }

    const int homo = gw_u.n_occ - 1;
    const int lumo = gw_u.n_occ;
    const double e_h = gw_u.qp_energy_ha[static_cast<std::size_t>(homo)];
    const double e_l = gw_u.qp_energy_ha[static_cast<std::size_t>(lumo)];
    if (!(e_h < e_l)) {
        std::fprintf(stderr,
                     "FAIL: ε_HOMO^QP=%.8f not below ε_LUMO^QP=%.8f Ha\n", e_h,
                     e_l);
        return 1;
    }
    const double d_h =
        std::abs(e_h - gw_u.mf_energy_ha[static_cast<std::size_t>(homo)]);
    const double d_l =
        std::abs(e_l - gw_u.mf_energy_ha[static_cast<std::size_t>(lumo)]);
    if (!std::isfinite(d_h) || !std::isfinite(d_l)) {
        std::fprintf(stderr, "FAIL: non-finite QP shifts\n");
        return 1;
    }

    // RPBE v_xc must not silently equal the PBE UGW first-green pin.
    if (std::abs(gw_u.gap_ev - kPbeUgwGapEv) < kDeFloorEv) {
        std::fprintf(stderr,
                     "FAIL: UGW@RPBE gap=%.6f eV is within %.1e eV of PBE "
                     "UGW pin %.6f — RPBE spin XC may be inactive\n",
                     gw_u.gap_ev, kDeFloorEv, kPbeUgwGapEv);
        return 1;
    }

    // ── RKS RPBE + G0W0 dual-gate twin ────────────────────────────────
    GpwDft m_r;
    configure_common(m_r, a_bohr);
    m_r.n_occ = 4; // closed-shell RKS
    m_r.build(atoms);

    const auto t3 = std::chrono::steady_clock::now();
    const GpwResult& scf_r = m_r.run(atoms);
    const auto t4 = std::chrono::steady_clock::now();
    if (!scf_r.converged || !std::isfinite(scf_r.e_total)) {
        std::fprintf(stderr,
                     "FAIL: Γ RKS-RPBE SCF did not converge "
                     "(iters=%d E=%.8f)\n",
                     scf_r.iters, scf_r.e_total);
        return 1;
    }

    m_r.mp2_n_virt = 0;
    m_r.mp2_ri = true;
    const G0W0Result& gw_r = m_r.g0w0(atoms);
    const auto t5 = std::chrono::steady_clock::now();
    const double wall_scf_r = std::chrono::duration<double>(t4 - t3).count();
    const double wall_gw_r = std::chrono::duration<double>(t5 - t4).count();

    if (!gw_r.converged || !(gw_r.gap_ev > 0.0) || !std::isfinite(gw_r.gap_ev)) {
        std::fprintf(stderr,
                     "FAIL: RKS G0W0@RPBE not usable for dual-gate "
                     "(conv=%d gap=%.6f)\n",
                     static_cast<int>(gw_r.converged), gw_r.gap_ev);
        return 1;
    }

    const double d_gap = std::abs(gw_u.gap_ev - gw_r.gap_ev);
    if (d_gap > kGapTolEv) {
        std::fprintf(stderr,
                     "FAIL: |gap_UGW−gap_RKS|=%.6f eV > tol %.3f eV "
                     "(UGW=%.6f RKS=%.6f; see REFERENCE.md)\n",
                     d_gap, kGapTolEv, gw_u.gap_ev, gw_r.gap_ev);
        return 1;
    }

    const double wall_total =
        wall_scf_u + wall_gw_u + wall_scf_r + wall_gw_r;

    std::printf(
        "PASS: Γ UGW@UKS-RPBE Si2 (DZVP-MOLOPT-GTH, GTH-PBE-q4, 30 Ry)\n"
        "  E_SCF(U)  = %.10f Ha (iters=%d, <S2>=%.6f, wall=%.1f s)\n"
        "  E_SCF(R)  = %.10f Ha (iters=%d, wall=%.1f s)\n"
        "  KS gap α  = %.6f eV\n"
        "  QP gap U  = %.6f eV  (IP=%.4f eV, EA=%.4f eV)\n"
        "  QP gap R  = %.6f eV  |U−R|=%.4e eV (tol %.3f)\n"
        "  |gap−PBE UGW pin %.6f|=%.4e eV (floor %.1e)\n"
        "  ε_HOMO^QP = %.8f Ha  (Δ=%.6f Ha vs KS)\n"
        "  ε_LUMO^QP = %.8f Ha  (Δ=%.6f Ha vs KS)\n"
        "  n_occ(α)=%d n_virt(α+β)=%d  UGW wall=%.1f s  RKS-GW wall=%.1f s\n"
        "  total=%.1f s  [dual-gate UGW≡RGW@RPBE; see REFERENCE.md]\n",
        scf_u.e_total, scf_u.iters, scf_u.s2, wall_scf_u, scf_r.e_total,
        scf_r.iters, wall_scf_r, gw_u.mf_gap_ev, gw_u.gap_ev, gw_u.ip_ev,
        gw_u.ea_ev, gw_r.gap_ev, d_gap, kGapTolEv, kPbeUgwGapEv,
        std::abs(gw_u.gap_ev - kPbeUgwGapEv), kDeFloorEv, e_h, d_h, e_l, d_l,
        gw_u.n_occ, gw_u.n_virt, wall_gw_u, wall_gw_r, wall_total);
    return 0;
}
