/// @file regressions/gpw/periodic-scs-mp2-kpoints/reg_periodic-scs-mp2-kpoints.cpp
/// @brief END-TO-END: multi-k RI-GPW SCS-MP2 on diamond Si (bct 4-atom, MP 2×1×1)
///        (Grimme c_OS=1.2, c_SS=1/3; DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE).
///        See REFERENCE.md.
///
/// Public API only (`GpwDft::run` + `GpwDft::mp2` with `mp2_ri`, multi-k mesh,
/// and `mp2_os_scale` / `mp2_ss_scale`). Requires ATV_BUILD_GPW (+ HF + PW).
/// Exit non-zero on mismatch. ≤ 60 s, single process. No live CP2K —
/// offline-pin contract per tools/cp2k_oracle/README.md.

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

// Diamond Si body-centred-tetragonal 4-atom cell (same as gpw-kpoints-si /
// periodic-mp2-kpoints).
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

    // Literature dual gate (REFERENCE.md): Grimme SCS on multi-k RI-GPW OS/SS.
    // CP2K 2026.1 git:5e54ba2 aborts MP2_GPW under DFT KPOINTS (no algorithm
    // twin). Absolute band freezes first-green product digits (setup integrity).
    constexpr double kEcorrRef =
        -0.0737958960;                      // Ha (ATV first green, 2026-08-01)
    constexpr double kEcorrTol = 1.0e-4;    // Ha
    constexpr double kScaleIdTol = 1.0e-12; // Ha algebraic identity
    constexpr double kOsSsPlainTol = 1.0e-10; // Ha unscaled vs plain

    auto atoms = si_bct(a_bohr);

    GpwDft m;
    m.cell[0] = m.cell[1] = a_bohr / std::sqrt(2.0);
    m.cell[2] = a_bohr;
    m.cutoff_ry = 80.0;
    m.n_occ = 8; // 4 Si × 4 valence e⁻ / 2
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
    const auto t1 = std::chrono::steady_clock::now();
    if (!scf.converged || scf.n_kpts != 2 || scf.C_k.size() != 2) {
        std::fprintf(stderr,
                     "FAIL: 2k SCF (conv=%d n_kpts=%d C_k=%zu iters=%d "
                     "E=%.8f)\n",
                     static_cast<int>(scf.converged), scf.n_kpts,
                     scf.C_k.size(), scf.iters, scf.e_total);
        return 1;
    }

    // Plain RI-MP2 on the same multi-k orbitals (default scales 1/1).
    m.mp2_os_scale = 1.0;
    m.mp2_ss_scale = 1.0;
    const Mp2Result r_plain = m.mp2(atoms);
    const auto t2 = std::chrono::steady_clock::now();

    // Grimme SCS-MP2: c_OS = 1.2, c_SS = 1/3 (same members as Γ / molecular).
    m.mp2_os_scale = 1.2;
    m.mp2_ss_scale = 1.0 / 3.0;
    const Mp2Result r_scs = m.mp2(atoms);
    const auto t3 = std::chrono::steady_clock::now();

    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();
    const double wall_plain = std::chrono::duration<double>(t2 - t1).count();
    const double wall_scs = std::chrono::duration<double>(t3 - t2).count();
    const double wall_twin = wall_scf + wall_scs; // fair twin = SCF + SCS-MP2
    const double wall_all = std::chrono::duration<double>(t3 - t0).count();

    if (!r_scs.ri || !r_plain.ri) {
        std::fprintf(stderr, "FAIL: Mp2Result.ri not set on RI-GPW path\n");
        return 1;
    }
    if (r_scs.n_kpts != 2 || r_plain.n_kpts != 2) {
        std::fprintf(stderr, "FAIL: MP2 n_kpts plain=%d scs=%d (want 2)\n",
                     r_plain.n_kpts, r_scs.n_kpts);
        return 1;
    }
    if (!std::isfinite(r_scs.e_corr) || !(r_scs.e_corr < 0.0)) {
        std::fprintf(stderr,
                     "FAIL: E_corr(SCS-MP2)=%.10f Ha not finite/negative\n",
                     r_scs.e_corr);
        return 1;
    }
    if (r_scs.n_occ != 8 || r_scs.n_virt != 2) {
        std::fprintf(stderr, "FAIL: n_occ=%d n_virt=%d (want 8/2)\n",
                     r_scs.n_occ, r_scs.n_virt);
        return 1;
    }

    // Unscaled OS/SS are independent of the scale members.
    if (std::abs(r_scs.e_corr_os - r_plain.e_corr_os) > kOsSsPlainTol
        || std::abs(r_scs.e_corr_ss - r_plain.e_corr_ss) > kOsSsPlainTol) {
        std::fprintf(stderr,
                     "FAIL: unscaled OS/SS changed under SCS scales "
                     "(plain os=%.12f ss=%.12f; scs os=%.12f ss=%.12f)\n",
                     r_plain.e_corr_os, r_plain.e_corr_ss, r_scs.e_corr_os,
                     r_scs.e_corr_ss);
        return 1;
    }
    if (std::abs(r_plain.e_corr - (r_plain.e_corr_os + r_plain.e_corr_ss))
        > kScaleIdTol) {
        std::fprintf(stderr,
                     "FAIL: plain OS+SS != e_corr: os=%.12f ss=%.12f "
                     "corr=%.12f\n",
                     r_plain.e_corr_os, r_plain.e_corr_ss, r_plain.e_corr);
        return 1;
    }

    // Product identity: e_corr = c_OS * OS + c_SS * SS.
    const double e_scs_from_parts =
        1.2 * r_scs.e_corr_os + (1.0 / 3.0) * r_scs.e_corr_ss;
    if (std::abs(r_scs.e_corr - e_scs_from_parts) > kScaleIdTol) {
        std::fprintf(stderr,
                     "FAIL: SCS algebra broken: e_corr=%.12f vs "
                     "1.2*os+(1/3)*ss=%.12f\n",
                     r_scs.e_corr, e_scs_from_parts);
        return 1;
    }
    if (std::abs(r_scs.e_total - (r_scs.e_scf + r_scs.e_corr)) > kScaleIdTol) {
        std::fprintf(stderr,
                     "FAIL: e_total != e_scf + e_corr (%.12f vs %.12f)\n",
                     r_scs.e_total, r_scs.e_scf + r_scs.e_corr);
        return 1;
    }

    // Offline CP2K pin band.
    if (std::abs(r_scs.e_corr - kEcorrRef) > kEcorrTol) {
        std::fprintf(stderr,
                     "FAIL: E_corr(SCS) got=%.10f want=%.10f +/- %.1e Ha "
                     "(offline pin band; see REFERENCE.md)\n",
                     r_scs.e_corr, kEcorrRef, kEcorrTol);
        return 1;
    }
    if (wall_all > 60.0) {
        std::fprintf(stderr, "FAIL: wall=%.1f s exceeds 60 s budget\n",
                     wall_all);
        return 1;
    }

    std::printf(
        "PASS: multi-k RI-GPW SCS-MP2 Si4-bct (MP 2x1x1, n_virt=%d, %.0f Ry)\n"
        "  E_SCF    = %.10f Ha (iters=%d, n_kpts=%d, wall=%.2f s)\n"
        "  plain    = %.10f Ha (OS=%.10f SS=%.10f; wall=%.2f s)\n"
        "  E_corr   = %.10f Ha (SCS; 1.2*OS+(1/3)*SS; wall=%.2f s)\n"
        "  E_total  = %.10f Ha   twin_wall(SCF+SCS)=%.2f s\n"
        "  [literature SCS pin %.10f Ha; CP2K multi-k MP2_GPW absent; "
        "see REFERENCE.md]\n",
        r_scs.n_virt, m.cutoff_ry, r_scs.e_scf, scf.iters, r_scs.n_kpts,
        wall_scf, r_plain.e_corr, r_plain.e_corr_os, r_plain.e_corr_ss,
        wall_plain, r_scs.e_corr, wall_scs, r_scs.e_total, wall_twin,
        kEcorrRef);
    return 0;
}
