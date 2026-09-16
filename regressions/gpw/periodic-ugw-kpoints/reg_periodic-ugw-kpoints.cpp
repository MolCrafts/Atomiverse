/// @file regressions/gpw/periodic-ugw-kpoints/reg_periodic-ugw-kpoints.cpp
/// @brief END-TO-END: multi-k unrestricted spectral G0W0 (UGW@UKS) on
///        diamond Si (bct 4-atom, MP 2×1×1). Dual-gates closed-shell
///        multi-k G0W0 first-green. See REFERENCE.md. Public API only.
///        ATV_BUILD_GPW (+ HF + PW). No live CP2K. <= 60 s.

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

std::vector<ATV::cpu::gpw::GpwAtom> si_bct(double a_bohr) {
    using namespace ATV::cpu::gpw;
    const double at = a_bohr / std::sqrt(2.0);
    const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
    const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    std::vector<GpwAtom> atoms;
    for (int i = 0; i < 4; ++i) {
        const double pos[3] = {frac[i][0] * at, frac[i][1] * at,
                               frac[i][2] * a_bohr};
        atoms.emplace_back(pos, bas, pot, 14);
    }
    return atoms;
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;
    constexpr double kA2B = 1.0 / 0.52917720859;
    const double a_bohr = 5.43 * kA2B;

    // Dual-gate vs closed-shell multi-k finite-q G0W0 first-green
    // (periodic-gw-kpoints). UKS nα=nβ → UGW ≡ RGW when orbitals match.
    constexpr double kGapRefEv = 7.529834; // eV (periodic-gw-kpoints)
    constexpr double kGapTolEv = 0.5;      // eV setup band

    auto atoms = si_bct(a_bohr);

    GpwDft m;
    m.cell[0] = m.cell[1] = a_bohr / std::sqrt(2.0);
    m.cell[2] = a_bohr;
    m.cutoff_ry = 80.0;
    m.n_alpha = 8;
    m.n_beta = 8;
    m.max_iter = 60;
    m.tol = 1e-7;
    m.mix = 0.3;
    m.level_shift = 0.0;
    m.eps_default = 1e-7;
    m.xc = XcKind::PBE;
    m.mp2_ri = true;
    m.mp2_n_virt = 2;
    m.k_mesh[0] = 2;
    m.k_mesh[1] = 1;
    m.k_mesh[2] = 1;
    m.k_time_reversal = false;
    m.diis_depth = 6;
    m.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const GpwResult& scf = m.run(atoms);
    if (!scf.converged || scf.n_kpts != 2 || scf.C_k.size() != 2
        || scf.C_k_beta.size() != 2) {
        std::fprintf(stderr,
                     "FAIL: 2k UKS SCF (conv=%d n_kpts=%d C_k=%zu "
                     "C_k_beta=%zu iters=%d E=%.8f)\n",
                     static_cast<int>(scf.converged), scf.n_kpts,
                     scf.C_k.size(), scf.C_k_beta.size(), scf.iters,
                     scf.e_total);
        return 1;
    }
    if (scf.eps_k_beta.size() != 2) {
        std::fprintf(stderr, "FAIL: missing eps_k_beta (size=%zu)\n",
                     scf.eps_k_beta.size());
        return 1;
    }

    const G0W0Result& gw = m.g0w0(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();

    if (gw.n_kpts != 2 || !gw.ri) {
        std::fprintf(stderr, "FAIL: G0W0 flags (n_kpts=%d ri=%d)\n", gw.n_kpts,
                     static_cast<int>(gw.ri));
        return 1;
    }
    if (!gw.converged) {
        std::fprintf(stderr, "FAIL: UGW Newton QP solve did not converge\n");
        return 1;
    }
    // n_occ reports n_α; n_virt = n_va + n_vb with cap 2 per spin → 4.
    if (gw.n_occ != 8 || gw.n_virt != 4) {
        std::fprintf(stderr, "FAIL: n_occ=%d n_virt=%d (want 8/4)\n", gw.n_occ,
                     gw.n_virt);
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
    if (static_cast<int>(gw.qp_energy_ha.size()) <= lumo) {
        std::fprintf(stderr, "FAIL: qp_energy_ha too short\n");
        return 1;
    }
    const double e_h = gw.qp_energy_ha[static_cast<std::size_t>(homo)];
    const double e_l = gw.qp_energy_ha[static_cast<std::size_t>(lumo)];
    if (!(e_h < e_l)) {
        std::fprintf(stderr,
                     "FAIL: ε_HOMO^QP=%.8f not below ε_LUMO^QP=%.8f Ha\n", e_h,
                     e_l);
        return 1;
    }
    if (std::abs(gw.gap_ev - kGapRefEv) > kGapTolEv) {
        std::fprintf(stderr,
                     "FAIL: QP gap got=%.6f want~%.6f +/- %.3f eV "
                     "(UGW dual-gate vs periodic-gw-kpoints; see "
                     "REFERENCE.md)\n",
                     gw.gap_ev, kGapRefEv, kGapTolEv);
        return 1;
    }
    if (wall > 60.0) {
        std::fprintf(stderr, "FAIL: wall=%.1f s exceeds 60 s budget\n", wall);
        return 1;
    }

    std::printf(
        "PASS: multi-k UGW@UKS-PBE Si4-bct (MP 2x1x1, n_virt/spin=%d, "
        "%.0f Ry)\n"
        "  E_SCF     = %.10f Ha (iters=%d, n_kpts=%d, <S2>=%.6f)\n"
        "  KS gap α  = %.6f eV  (min direct over mesh)\n"
        "  QP gap α  = %.6f eV  (IP=%.4f eV, EA=%.4f eV)\n"
        "  ε_HOMO^QP = %.8f Ha\n"
        "  ε_LUMO^QP = %.8f Ha\n"
        "  n_occ(α)=%d n_virt(α+β)=%d n_kpts=%d  wall=%.1f s\n"
        "  [dual-gate |gap−%.6f|≤%.3f eV vs RKS multi-k G0W0; FIRST_GREEN "
        "gap_ev=%.12f]\n",
        m.mp2_n_virt, m.cutoff_ry, scf.e_total, scf.iters, scf.n_kpts, scf.s2,
        gw.mf_gap_ev, gw.gap_ev, gw.ip_ev, gw.ea_ev, e_h, e_l, gw.n_occ,
        gw.n_virt, gw.n_kpts, wall, kGapRefEv, kGapTolEv, gw.gap_ev);
    return 0;
}
