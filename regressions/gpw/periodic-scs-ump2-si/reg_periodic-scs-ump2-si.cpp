/// @file regressions/gpw/periodic-scs-ump2-si/reg_periodic-scs-ump2-si.cpp
/// @brief END-TO-END: Γ-point RI-GPW SCS-UMP2 on periodic Si₂
///        (UKS PBE nα=nβ; Grimme c_OS=1.2, c_SS=1/3; DZVP-MOLOPT-GTH +
///        GTH-PBE-q4). See REFERENCE.md.
///
/// Public API only (`GpwDft::run` + `GpwDft::mp2` with `mp2_ri` and
/// `mp2_os_scale` / `mp2_ss_scale` on the UKS path). Requires ATV_BUILD_GPW
/// (+ HF + PW). Exit non-zero on mismatch. ≤ 60 s, single process. No live
/// CP2K — offline-pin contract per tools/cp2k_oracle/README.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/mp2.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV::cpu::gpw;
    constexpr double kA2B = 1.8897259885789;
    constexpr double a_bohr = 5.43 * kA2B;

    // Absolute band: closed-shell SCS product twin (periodic-scs-mp2-si /
    // CP2K RKS SCALE_S/SCALE_T). UKS nα=nβ with low spin contamination
    // recovers the same correlation to within UKS numerical drift.
    // CP2K UKS on this deck spin-breaks (S²≈1) and lands elsewhere — cited
    // in REFERENCE.md for provenance + wall only (not the absolute gate).
    constexpr double kEcorrRef = -0.29794120639418; // Ha (RKS SCS CP2K)
    constexpr double kEcorrTol = 2.0e-3;             // Ha (UKS drift)
    constexpr double kScaleIdTol = 1.0e-12;          // Ha  algebraic identity
    constexpr double kOsSsPlainTol = 1.0e-10; // Ha  unscaled channels vs plain

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
    m.n_alpha = 4;
    m.n_beta = 4;
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
                     "FAIL: periodic Γ UKS-PBE SCF did not converge "
                     "(iters=%d E=%.8f)\n",
                     scf.iters, scf.e_total);
        return 1;
    }
    if (scf.C_beta.size() == 0 || scf.eps_beta.empty()) {
        std::fprintf(stderr, "FAIL: UKS result missing β orbitals\n");
        return 1;
    }

    // Plain RI-UMP2 on the same orbitals (default scales 1/1).
    m.mp2_n_virt = 0;
    m.mp2_ri = true;
    m.mp2_os_scale = 1.0;
    m.mp2_ss_scale = 1.0;
    const Mp2Result r_plain = m.mp2(atoms);
    const auto t2 = std::chrono::steady_clock::now();

    // Grimme SCS-UMP2: c_OS = 1.2, c_SS = 1/3.
    m.mp2_os_scale = 1.2;
    m.mp2_ss_scale = 1.0 / 3.0;
    const Mp2Result r_scs = m.mp2(atoms);
    const auto t3 = std::chrono::steady_clock::now();

    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();
    const double wall_plain = std::chrono::duration<double>(t2 - t1).count();
    const double wall_scs = std::chrono::duration<double>(t3 - t2).count();
    const double wall_twin = wall_scf + wall_scs; // fair twin = SCF + SCS-UMP2

    if (!r_scs.ri || !r_plain.ri) {
        std::fprintf(stderr, "FAIL: Mp2Result.ri not set on RI-GPW path\n");
        return 1;
    }
    if (!std::isfinite(r_scs.e_corr) || !(r_scs.e_corr < 0.0)) {
        std::fprintf(stderr,
                     "FAIL: E_corr(SCS-UMP2)=%.10f Ha not finite/negative\n",
                     r_scs.e_corr);
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
    if (std::abs(r_plain.e_corr
                 - (r_plain.e_corr_os + r_plain.e_corr_ss))
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
    if (r_scs.n_occ != 8) {
        std::fprintf(stderr, "FAIL: n_occ=%d want 8 (nα+nβ)\n", r_scs.n_occ);
        return 1;
    }

    // Dual-gate absolute band (closed-shell SCS when UKS stays near RKS).
    if (std::abs(r_scs.e_corr - kEcorrRef) > kEcorrTol) {
        std::fprintf(stderr,
                     "FAIL: E_corr(SCS-UMP2) got=%.10f want=%.10f +/- %.1e Ha "
                     "(closed-shell SCS dual-gate; see REFERENCE.md)\n",
                     r_scs.e_corr, kEcorrRef, kEcorrTol);
        return 1;
    }

    std::printf(
        "PASS: Γ RI-GPW SCS-UMP2 Si2 (UKS-PBE nα=nβ, DZVP-MOLOPT-GTH, "
        "GTH-PBE-q4, 30 Ry)\n"
        "  E_SCF    = %.10f Ha (iters=%d, <S2>=%.6f, wall=%.2f s)\n"
        "  plain    = %.10f Ha (OS=%.10f SS=%.10f; wall=%.2f s)\n"
        "  E_corr   = %.10f Ha (SCS; 1.2*OS+(1/3)*SS; wall=%.2f s)\n"
        "  E_total  = %.10f Ha   twin_wall(SCF+SCS)=%.2f s\n"
        "  [dual-gate pin %.10f Ha; CP2K UKS wall twin in REFERENCE.md]\n",
        r_scs.e_scf, scf.iters, scf.s2, wall_scf, r_plain.e_corr,
        r_plain.e_corr_os, r_plain.e_corr_ss, wall_plain, r_scs.e_corr,
        wall_scs, r_scs.e_total, wall_twin, kEcorrRef);
    return 0;
}
