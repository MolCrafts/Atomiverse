/// @file regressions/gpw/periodic-scs-mp2-si/reg_periodic-scs-mp2-si.cpp
/// @brief END-TO-END: Γ-point RI-GPW SCS-MP2 on periodic Si₂
///        (Grimme c_OS=1.2, c_SS=1/3; DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE).
///        See REFERENCE.md.
///
/// Public API only (`GpwDft::run` + `GpwDft::mp2` with `mp2_ri` and
/// `mp2_os_scale` / `mp2_ss_scale`). Requires ATV_BUILD_GPW (+ HF + PW).
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

int main() {
    using namespace ATV::cpu::gpw;
    constexpr double kA2B = 1.8897259885789;
    constexpr double a_bohr = 5.43 * kA2B;

    // Offline CP2K 2026.1 git:5e54ba2 SCS-MP2_GPW pin (REFERENCE.md).
    // Unscaled SO/SS match plain RI-MP2 to ~1e-5 Ha; scaled E_corr band is
    // 1e-3 Ha (same grid/SCF twin tolerance as periodic-ri-mp2-si).
    constexpr double kEcorrRef = -0.29794120639418; // Ha (CP2K SCS-MP2)
    constexpr double kEcorrTol = 1.0e-3;            // Ha
    constexpr double kScaleIdTol = 1.0e-12;         // Ha  algebraic identity
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

    // Plain RI-MP2 on the same orbitals (default scales 1/1).
    m.mp2_n_virt = 0;
    m.mp2_ri = true;
    m.mp2_os_scale = 1.0;
    m.mp2_ss_scale = 1.0;
    const Mp2Result r_plain = m.mp2(atoms);
    const auto t2 = std::chrono::steady_clock::now();

    // Grimme SCS-MP2: c_OS = 1.2, c_SS = 1/3 (same members as molecular RHF).
    m.mp2_os_scale = 1.2;
    m.mp2_ss_scale = 1.0 / 3.0;
    const Mp2Result r_scs = m.mp2(atoms);
    const auto t3 = std::chrono::steady_clock::now();

    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();
    const double wall_plain = std::chrono::duration<double>(t2 - t1).count();
    const double wall_scs = std::chrono::duration<double>(t3 - t2).count();
    const double wall_twin = wall_scf + wall_scs; // fair twin = SCF + SCS-MP2

    if (!r_scs.ri || !r_plain.ri) {
        std::fprintf(stderr, "FAIL: Mp2Result.ri not set on RI-GPW path\n");
        return 1;
    }
    if (!std::isfinite(r_scs.e_corr) || !(r_scs.e_corr < 0.0)) {
        std::fprintf(stderr,
                     "FAIL: E_corr(SCS-MP2)=%.10f Ha not finite/negative\n",
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

    std::printf(
        "PASS: Γ RI-GPW SCS-MP2 Si2 (DZVP-MOLOPT-GTH, GTH-PBE-q4, PBE, 30 Ry)\n"
        "  E_SCF    = %.10f Ha (iters=%d, wall=%.2f s)\n"
        "  plain    = %.10f Ha (OS=%.10f SS=%.10f; wall=%.2f s)\n"
        "  E_corr   = %.10f Ha (SCS; 1.2*OS+(1/3)*SS; wall=%.2f s)\n"
        "  E_total  = %.10f Ha   twin_wall(SCF+SCS)=%.2f s\n"
        "  [CP2K pin %.10f Ha; see REFERENCE.md]\n",
        r_scs.e_scf, scf.iters, wall_scf, r_plain.e_corr, r_plain.e_corr_os,
        r_plain.e_corr_ss, wall_plain, r_scs.e_corr, wall_scs, r_scs.e_total,
        wall_twin, kEcorrRef);
    return 0;
}
