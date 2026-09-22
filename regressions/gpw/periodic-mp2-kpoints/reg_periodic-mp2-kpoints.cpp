/// @file regressions/gpw/periodic-mp2-kpoints/reg_periodic-mp2-kpoints.cpp
/// @brief END-TO-END: multi-k RI-GPW MP2 on diamond Si (bct 4-atom, MP 2×1×1).
///        See REFERENCE.md. Public API only. ATV_BUILD_GPW (+ HF + PW).
///        No live CP2K. <= 60 s.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/mp2.h"
#include "atomiverse/molrs.hpp"

namespace {

// Diamond Si body-centred-tetragonal 4-atom cell (same as gpw-kpoints-si).
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

    // Offline pin strategy (no CP2K binary on worker host):
    //  - multi-k 2×1×1 hard gates: C_k stored, e_corr < 0, OS+SS, e_total
    //  - unit tests pin complex≡real OS/SS and mo_on_grid_k@Γ≡mo_on_grid
    //  - first-green absolute band (loose floor until freeze)
    // First-green ATV pin (REFERENCE.md, 2026-07-25). Replace with CP2K when
    // offline binary exists. Tol 1e-4 Ha = setup integrity band.
    constexpr double kEcorr2kRef = -0.0657637511;
    constexpr double kEcorr2kTol = 1.0e-4;
    constexpr double kIdTol = 1.0e-12;

    auto atoms = si_bct(a_bohr);

    GpwDft m;
    m.cell[0] = m.cell[1] = a_bohr / std::sqrt(2.0);
    m.cell[2] = a_bohr;
    m.cutoff_ry = 80.0; // e2e budget (gpw-kpoints-si uses 150 for absolute pin)
    m.n_occ = 8;        // 4 Si × 4 valence e⁻ / 2
    m.max_iter = 60;
    m.tol = 1e-7;
    m.mix = 0.3;
    m.level_shift = 0.0; // k-path rejects nonzero
    m.eps_default = 1e-7;
    m.xc = XcKind::PBE;
    m.mp2_ri = true;
    m.mp2_n_virt = 2; // tight virt cap for ≤60 s
    m.k_mesh[0] = 2;
    m.k_mesh[1] = 1;
    m.k_mesh[2] = 1;
    m.k_time_reversal = false; // equal weights
    m.diis_depth = 6;
    m.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const GpwResult& scf = m.run(atoms);
    if (!scf.converged || scf.n_kpts != 2 || scf.C_k.size() != 2) {
        std::fprintf(stderr,
                     "FAIL: 2k SCF (conv=%d n_kpts=%d C_k=%zu iters=%d "
                     "E=%.8f)\n",
                     static_cast<int>(scf.converged), scf.n_kpts,
                     scf.C_k.size(), scf.iters, scf.e_total);
        return 1;
    }

    const Mp2Result r = m.mp2(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();

    if (r.n_kpts != 2 || !r.ri) {
        std::fprintf(stderr, "FAIL: MP2 flags (n_kpts=%d ri=%d)\n", r.n_kpts,
                     static_cast<int>(r.ri));
        return 1;
    }
    if (!std::isfinite(r.e_corr) || !(r.e_corr < 0.0)) {
        std::fprintf(stderr, "FAIL: E_corr=%.10f not finite/negative\n",
                     r.e_corr);
        return 1;
    }
    if (std::abs(r.e_corr_os + r.e_corr_ss - r.e_corr) > kIdTol) {
        std::fprintf(stderr,
                     "FAIL: OS+SS broken: os=%.12f ss=%.12f corr=%.12f\n",
                     r.e_corr_os, r.e_corr_ss, r.e_corr);
        return 1;
    }
    if (std::abs(r.e_total - (r.e_scf + r.e_corr)) > kIdTol) {
        std::fprintf(stderr, "FAIL: e_total != e_scf+e_corr\n");
        return 1;
    }
    if (r.n_occ != 8 || r.n_virt != 2) {
        std::fprintf(stderr, "FAIL: n_occ=%d n_virt=%d (want 8/2)\n", r.n_occ,
                     r.n_virt);
        return 1;
    }
    if (std::abs(r.e_corr - kEcorr2kRef) > kEcorr2kTol) {
        std::fprintf(stderr,
                     "FAIL: E_corr=%.10f outside loose band %.3f +/- %.3f\n",
                     r.e_corr, kEcorr2kRef, kEcorr2kTol);
        return 1;
    }
    if (wall > 60.0) {
        std::fprintf(stderr, "FAIL: wall=%.1f s exceeds 60 s budget\n", wall);
        return 1;
    }

    std::printf(
        "PASS: multi-k RI-GPW MP2 Si4-bct (MP 2x1x1, n_virt=%d, %.0f Ry)\n"
        "  E_SCF   = %.10f Ha (iters=%d, n_kpts=%d)\n"
        "  E_corr  = %.10f Ha (OS=%.10f SS=%.10f)\n"
        "  E_total = %.10f Ha   wall=%.1f s\n"
        "  [multi-k hard gates; kernel Γ recovery in unit tests; "
        "see REFERENCE.md]\n",
        r.n_virt, m.cutoff_ry, r.e_scf, scf.iters, r.n_kpts, r.e_corr,
        r.e_corr_os, r.e_corr_ss, r.e_total, wall);
    return 0;
}
