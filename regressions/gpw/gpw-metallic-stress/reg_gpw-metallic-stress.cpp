/// @file regressions/gpw/gpw-metallic-stress/reg_gpw-metallic-stress.cpp
/// @brief END-TO-END: multi-k GPW RKS free-energy analytic cell stress under
///        Fermi–Dirac metallic smearing (BCC Na).
///
/// Dual-gate: analytic fixed-{f_k} Hellmann–Feynman free-energy stress vs
/// multi-SCF strain-FD of the Mermin free energy F, plus offline CP2K 2026.1
/// STRESS_TENSOR NUMERICAL + SMEAR twin. See REFERENCE.md.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859; // Å → Bohr

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

/// BCC conventional cubic: 2 Na at (0,0,0) and (½,½,½).
std::vector<ATV::cpu::gpw::GpwAtom> na_bcc(double a_bohr) {
    using namespace ATV;
    const cpu::gpw::GaussianBasis bNa(molrs::Element::Na, "DZVP-MOLOPT-SR-GTH");
    const cpu::gpw::GthPotential pNa(molrs::Element::Na, "GTH-PBE-q9");
    const double frac[2][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
    std::vector<cpu::gpw::GpwAtom> atoms;
    for (int i = 0; i < 2; ++i) {
        const double pos[3] = {frac[i][0] * a_bohr, frac[i][1] * a_bohr,
                               frac[i][2] * a_bohr};
        atoms.emplace_back(pos, bNa, pNa, 11);
    }
    return atoms;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::gpw_stress;
    using cpu::gpw::gpw_stress_fd;
    using cpu::gpw::GpwDft;
    using cpu::gpw::Smearing;
    using cpu::gpw::XcKind;

    // Same system class as reg_gpw-metallic-smearing (BCC Na, FD 0.02 Ry).
    // Free-energy pin filled after first green; stress dual-gate is primary.
    const double f_pin_ha = -92.126330608378; // Ha free energy F (energy twin)
    const double f_tol_ha = 1e-3; // Ha (stress twin may differ grid)
    const double a_ang = 4.225;
    const double a = a_ang * kA2B;
    const double n_elec = 18.0; // 2 × GTH-PBE-q9

    auto atoms = na_bcc(a);

    auto make_cfg = [&]() {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = a;
        m.cutoff_ry = 150.0;
        m.n_occ = 9; // Ne/2
        m.max_iter = 80;
        m.tol = 1e-6;
        m.eps_default = 1e-7;
        m.xc = XcKind::PADE;
        m.diis_depth = 6;
        m.mix = 0.4;
        m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
        m.k_time_reversal = true;
        m.smearing = Smearing::fermi_dirac;
        m.degauss = 0.02; // Ry
        m.extra_bands = 4;
        m.local_pp_on_grid = true;
        return m;
    };

    GpwDft energy = make_cfg();
    energy.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const auto& r = energy.run(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();

    std::printf("F=%.10f Ha  e_fermi=%.6f Ha  n_elec=%.8f  e_entropy=%.6e Ha  "
                "iters=%d  wall_scf=%.3f s  nk=%d  converged=%d\n",
                r.e_total, r.e_fermi, r.n_elec_occ, r.e_entropy, r.iters,
                wall_scf, r.n_kpts, r.converged ? 1 : 0);

    check(r.converged, "R1 SCF converged");
    check(std::isfinite(r.e_total) && std::isfinite(r.e_fermi),
          "finite free energy and e_fermi");
    check(std::fabs(r.n_elec_occ - n_elec) < 1e-5, "R2 electron count");
    check(r.n_kpts > 1, "multi-k sampling active");
    check(!r.f_k.empty() && static_cast<int>(r.f_k.size()) == r.n_kpts,
          "R3 fractional f_k payload present");
    check(!r.C_k.empty() && static_cast<int>(r.C_k.size()) == r.n_kpts,
          "R4 full C_k payload present");
    check(std::fabs(r.e_total - f_pin_ha) < f_tol_ha, "R5 free energy pin");

    const double h = 2e-3;
    const auto t2 = std::chrono::steady_clock::now();
    const auto san = gpw_stress(atoms, make_cfg(), h);
    const auto t3 = std::chrono::steady_clock::now();
    const auto sfd = gpw_stress_fd(atoms, make_cfg(), h);
    const auto t4 = std::chrono::steady_clock::now();
    const double wall_an = std::chrono::duration<double>(t3 - t2).count();
    const double wall_fd = std::chrono::duration<double>(t4 - t3).count();

    double max_diag_err = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int ax = 0; ax < 3; ++ax) {
        const double an = san[static_cast<std::size_t>(ax)];
        const double fd = sfd[static_cast<std::size_t>(ax)];
        const double err = std::abs(an - fd);
        max_diag_err = std::max(max_diag_err, err);
        std::printf("  σ_%s  analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[ax], an, fd, err);
        check(std::isfinite(an) && std::isfinite(fd), "finite stress");
    }
    for (int k = 3; k < 6; ++k)
        check(std::abs(san[static_cast<std::size_t>(k)]) <= 1e-8,
              "off-diagonal ~ 0");

    // Free-energy analytic vs multi-SCF FD of F; residual SCF + O(h²).
    constexpr double kAnFdTol = 2.0e-4; // Ha/Bohr³
    std::printf(
        "max|σ_an-σ_fd|=%.3e Ha/Bohr3  wall_an=%.3f s  wall_fd=%.3f s\n",
        max_diag_err, wall_an, wall_fd);
    check(max_diag_err <= kAnFdTol, "A2 free-energy analytic vs FD max diag");

    // Absolute soft pins (first green 2026-08-01; REFERENCE.md).
    // Cubic → σ_xx = σ_yy = σ_zz within mesh residual under FD smear.
    constexpr double kSDiagPin = -1.91772220e-02; // Ha/Bohr³ mean diag
    constexpr double kSTol = 5.0e-4;              // Ha/Bohr³ soft absolute
    const double max_diag =
        std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
    std::printf("max|sigma_diag|=%.3e Ha/Bohr3\n", max_diag);
    check(max_diag > 1e-6, "stress magnitude floor");
    check(std::abs(san[0] - san[1]) < 5e-5, "cubic σ_xx≈σ_yy");
    check(std::abs(san[0] - san[2]) < 5e-4, "cubic σ_xx≈σ_zz soft");

    const double s_mean = (san[0] + san[1] + san[2]) / 3.0;
    std::printf("mean σ_diag analytic=%.8e Ha/Bohr3  |d pin|=%.3e\n", s_mean,
                std::abs(s_mean - kSDiagPin));
    check(std::abs(s_mean - kSDiagPin) <= kSTol, "A4 mean σ_diag absolute pin");

    // Offline CP2K NUMERICAL free-energy stress (REFERENCE.md): same sign /
    // order as product; absolute grid residual ~0.002 Ha/Bohr³ vs CP2K bar
    // conversion — dual-gate analytic↔FD is the hard product gate.
    constexpr double kCp2kSDiagHa = -1.7051e-02; // from −5.01678e6 bar
    constexpr double kCp2kSoftTol = 5.0e-3;      // Ha/Bohr³ soft class check
    check(std::abs(s_mean - kCp2kSDiagHa) <= kCp2kSoftTol,
          "A5 CP2K NUMERICAL stress class (soft absolute)");

    // Fair wall: SCF-only twin wall reported for ratio (not multi-gate reg wall).
    std::printf("T_atv_scf=%.3f s (fair twin wall; stress assembly separate)\n",
                wall_scf);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-metallic-stress  F=%.10f  max|dσ|=%.3e  meanσ=%.3e  "
                "T_scf=%.3f s\n",
                r.e_total, max_diag_err, s_mean, wall_scf);
    return 0;
}
