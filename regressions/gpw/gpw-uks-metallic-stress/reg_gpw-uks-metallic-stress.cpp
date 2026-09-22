/// @file regressions/gpw/gpw-uks-metallic-stress/reg_gpw-uks-metallic-stress.cpp
/// @brief END-TO-END: multi-k UKS free-energy analytic cell stress under FD.
///
/// BCC Na collinear UKS + Fermi–Dirac smear. Dual-gate analytic vs multi-SCF
/// FD of F; closed-shell UKS+smear ≡ RKS+smear stress identity; soft CP2K
/// NUMERICAL class check. See REFERENCE.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859;

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

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

    const double f_pin_ha =
        -92.126330608004; // same class as uks-metallic-forces
    const double f_tol_ha = 1e-3;
    const double a_ang = 4.225;
    const double a = a_ang * kA2B;
    const double n_elec = 18.0;

    auto atoms = na_bcc(a);

    auto make_uks = [&]() {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = a;
        m.cutoff_ry = 150.0;
        m.n_alpha = 9;
        m.n_beta = 9;
        m.max_iter = 80;
        m.tol = 1e-6;
        m.eps_default = 1e-7;
        m.xc = XcKind::PADE;
        m.diis_depth = 6;
        m.mix = 0.4;
        m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
        m.k_time_reversal = true;
        m.smearing = Smearing::fermi_dirac;
        m.degauss = 0.02;
        m.extra_bands = 4;
        m.local_pp_on_grid = true;
        return m;
    };

    auto make_rks = [&]() {
        GpwDft m = make_uks();
        m.n_alpha = 0;
        m.n_beta = 0;
        m.n_occ = 9;
        return m;
    };

    GpwDft energy = make_uks();
    energy.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const auto& r = energy.run(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();

    std::printf("F=%.10f Ha  e_fermi=%.6f Ha  n_elec=%.8f  e_entropy=%.6e Ha  "
                "iters=%d  wall_scf=%.3f s  nk=%d  conv=%d\n",
                r.e_total, r.e_fermi, r.n_elec_occ, r.e_entropy, r.iters,
                wall_scf, r.n_kpts, r.converged ? 1 : 0);

    check(r.converged, "R1 SCF converged");
    check(std::isfinite(r.e_total) && std::isfinite(r.e_fermi),
          "finite free energy and e_fermi");
    check(std::fabs(r.n_elec_occ - n_elec) < 1e-5, "R2 electron count");
    check(r.n_kpts > 1, "multi-k sampling active");
    check(!r.f_k.empty() && !r.f_k_beta.empty(), "dual fractional f_k payload");
    check(std::fabs(r.e_total - f_pin_ha) < f_tol_ha, "R5 free energy pin");

    const double h = 2e-3;
    const auto t2 = std::chrono::steady_clock::now();
    const auto san = gpw_stress(atoms, make_uks(), h);
    const auto t3 = std::chrono::steady_clock::now();
    const auto sfd = gpw_stress_fd(atoms, make_uks(), h);
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

    constexpr double kAnFdTol = 2.0e-4; // Ha/Bohr³ (sibling metallic-stress)
    std::printf(
        "max|σ_an-σ_fd|=%.3e Ha/Bohr3  wall_an=%.3f s  wall_fd=%.3f s\n",
        max_diag_err, wall_an, wall_fd);
    check(max_diag_err <= kAnFdTol, "A2 free-energy analytic vs FD max diag");

    // Closed-shell UKS+smear ≡ RKS+smear stress identity.
    const auto s_rks = gpw_stress(atoms, make_rks(), h);
    double max_cs = 0.0;
    for (int ax = 0; ax < 3; ++ax)
        max_cs =
            std::max(max_cs, std::abs(san[static_cast<std::size_t>(ax)]
                                      - s_rks[static_cast<std::size_t>(ax)]));
    std::printf("max|σ_uks−σ_rks|=%.3e Ha/Bohr3\n", max_cs);
    check(max_cs <= 5e-5, "closed-shell UKS+smear ≡ RKS stress");

    // Soft absolute class (sibling metallic-stress pin).
    constexpr double kSDiagPin = -1.91772220e-02;
    constexpr double kSTol = 5.0e-4;
    const double s_mean = (san[0] + san[1] + san[2]) / 3.0;
    std::printf("mean σ_diag=%.8e Ha/Bohr3  |d pin|=%.3e\n", s_mean,
                std::abs(s_mean - kSDiagPin));
    check(std::abs(s_mean - kSDiagPin) <= kSTol, "mean σ_diag absolute pin");

    constexpr double kCp2kSDiagHa = -1.7051e-02;
    constexpr double kCp2kSoftTol = 5.0e-3;
    check(std::abs(s_mean - kCp2kSDiagHa) <= kCp2kSoftTol,
          "CP2K NUMERICAL stress class (soft)");

    std::printf("T_atv_scf=%.3f s\n", wall_scf);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf(
        "PASS: gpw-uks-metallic-stress  F=%.10f  max|dσ|=%.3e  meanσ=%.3e  "
        "T_scf=%.3f s\n",
        r.e_total, max_diag_err, s_mean, wall_scf);
    return 0;
}
