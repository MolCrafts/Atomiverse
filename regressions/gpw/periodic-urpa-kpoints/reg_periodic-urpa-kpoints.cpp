/// @file regressions/gpw/periodic-urpa-kpoints/reg_periodic-urpa-kpoints.cpp
/// @brief END-TO-END: multi-k unrestricted GPW dRPA (URPA@UKS) on diamond Si
///        (bct 4-atom, MP 2×1×1). See REFERENCE.md. Public API only.
///        ATV_BUILD_GPW (+ HF + PW). No live CP2K. <= 60 s.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/rpa.h"
#include "atomiverse/molrs.hpp"

namespace {

// Diamond Si body-centred-tetragonal 4-atom cell (same as
// periodic-rpa-kpoints / periodic-ugw-kpoints).
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

    // Dual-gate vs closed-shell multi-k dRPA first-green on the same deck
    // (UKS nα=nβ → multi-k URPA ≡ multi-k RKS dRPA when orbitals match).
    // Offline CP2K re-pin when available — REFERENCE.md.
    constexpr double kEcorr2kRef = -0.0736273682; // Ha (RKS multi-k twin)
    constexpr double kEcorr2kTol = 5.0e-3;
    constexpr double kIdTol = 1.0e-12;

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
        || scf.C_k_beta.size() != 2 || scf.eps_k_beta.size() != 2) {
        std::fprintf(stderr,
                     "FAIL: 2k UKS SCF (conv=%d n_kpts=%d C_k=%zu "
                     "C_k_beta=%zu iters=%d E=%.8f)\n",
                     static_cast<int>(scf.converged), scf.n_kpts,
                     scf.C_k.size(), scf.C_k_beta.size(), scf.iters,
                     scf.e_total);
        return 1;
    }

    const RpaResult r = m.direct_rpa(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();

    if (r.n_kpts != 2 || !r.ri) {
        std::fprintf(stderr, "FAIL: RPA flags (n_kpts=%d ri=%d)\n", r.n_kpts,
                     static_cast<int>(r.ri));
        return 1;
    }
    if (!std::isfinite(r.e_corr) || !(r.e_corr < 0.0)) {
        std::fprintf(stderr, "FAIL: E_corr=%.10f not finite/negative\n",
                     r.e_corr);
        return 1;
    }
    if (std::abs(r.e_total - (r.e_scf + r.e_corr)) > kIdTol) {
        std::fprintf(stderr, "FAIL: e_total != e_scf+e_corr\n");
        return 1;
    }
    // Spin-orbital counts: nα+nβ = 16; n_virtα+n_virtβ = 4 (mp2_n_virt=2).
    if (r.n_occ != 16 || r.n_virt != 4) {
        std::fprintf(stderr, "FAIL: n_occ=%d n_virt=%d (want 16/4)\n", r.n_occ,
                     r.n_virt);
        return 1;
    }
    // Finite-q product: n_omega = N_q * N_k * nov_composite with N_q = N_k
    // and nov = 2 * nα * n_va (α⊕β) = 2 * 8 * 2 = 32 → n_Ω = 2*2*32 = 128.
    const int n_omega_expect = 2 * 2 * (8 * 2 + 8 * 2);
    if (static_cast<int>(r.omega_ha.size()) != n_omega_expect) {
        std::fprintf(stderr, "FAIL: omega size %zu != %d\n", r.omega_ha.size(),
                     n_omega_expect);
        return 1;
    }
    for (double om : r.omega_ha) {
        if (!(om > 0.0) || !std::isfinite(om)) {
            std::fprintf(stderr, "FAIL: non-positive Ω=%.6e\n", om);
            return 1;
        }
    }
    if (std::abs(r.e_corr - kEcorr2kRef) > kEcorr2kTol) {
        std::fprintf(stderr,
                     "FAIL: E_corr=%.10f outside dual-gate band %.6f +/- %.3f "
                     "(RKS multi-k twin; see REFERENCE.md)\n",
                     r.e_corr, kEcorr2kRef, kEcorr2kTol);
        return 1;
    }
    if (wall > 60.0) {
        std::fprintf(stderr, "FAIL: wall=%.1f s exceeds 60 s budget\n", wall);
        return 1;
    }

    std::printf(
        "PASS: multi-k finite-q RI-GPW URPA Si4-bct (UKS-PBE, MP 2x1x1, "
        "n_virt/spin=%d, %.0f Ry)\n"
        "  E_SCF   = %.10f Ha (iters=%d, n_kpts=%d, <S2>=%.6f)\n"
        "  E_corr  = %.10f Ha (Tr A=%.6f; n_occ=%d n_virt=%d)\n"
        "  E_total = %.10f Ha   wall=%.1f s\n"
        "  Ω_min   = %.6f Ha    Ω_max=%.6f Ha  (n_om=%zu)\n"
        "  [URPA@UKS dual-gate vs RKS multi-k first-green; FIRST_GREEN_PIN "
        "e_corr=%.12f]\n",
        m.mp2_n_virt, m.cutoff_ry, r.e_scf, scf.iters, r.n_kpts, scf.s2,
        r.e_corr, r.tr_A, r.n_occ, r.n_virt, r.e_total, wall,
        r.omega_ha.front(), r.omega_ha.back(), r.omega_ha.size(), r.e_corr);
    return 0;
}
