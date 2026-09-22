/// @file regressions/gpw/gpw-kpoint-stress/reg_gpw-kpoint-stress.cpp
/// @brief END-TO-END: GPW analytic cell stress on a Monkhorst-Pack k mesh.
///
/// Diamond Si bct 4-atom cell (DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE), mesh
/// 1x1x2. Dual-gate analytic fixed-P_k vs multi-SCF strain-FD. See REFERENCE.md.

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859; // Angstrom -> Bohr

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

std::vector<ATV::cpu::gpw::GpwAtom> si_bct(double a_bohr) {
    const double at = a_bohr / std::sqrt(2.0);
    const ATV::cpu::gpw::GaussianBasis bSi(molrs::Element::Si,
                                           "DZVP-MOLOPT-GTH");
    const ATV::cpu::gpw::GthPotential pSi(molrs::Element::Si, "GTH-PBE-q4");
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    std::vector<ATV::cpu::gpw::GpwAtom> atoms;
    for (int i = 0; i < 4; ++i) {
        const double pos[3] = {frac[i][0] * at, frac[i][1] * at,
                               frac[i][2] * a_bohr};
        atoms.emplace_back(pos, bSi, pSi, 14);
    }
    return atoms;
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;

    const double a = 5.43 * kA2B; // 10.26120 Bohr
    const auto atoms = si_bct(a);

    auto make_cfg = [&]() {
        GpwDft m;
        m.cell[0] = m.cell[1] = a / std::sqrt(2.0);
        m.cell[2] = a;
        m.cutoff_ry = 100.0;
        m.n_occ = 8;
        m.max_iter = 60;
        m.tol = 1e-8;
        m.xc = XcKind::PBE;
        m.eps_default = 1e-7;
        m.local_pp_on_grid = true;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2; // MP → κ_z = ±1/4, TR → one k
        return m;
    };

    // A1: energy + SCF
    GpwDft energy = make_cfg();
    energy.build(atoms);
    const auto& er = energy.run(atoms);
    check(er.converged, "A1 SCF converged");
    check(std::isfinite(er.e_total), "A1 finite e_total");
    check(er.n_kpts >= 1 && !er.C_occ_k.empty(), "A1 C_occ_k payload present");
    std::printf("E(mesh 1x1x2, 100 Ry) = %.12f Ha  iters=%d  nk=%d\n",
                er.e_total, er.iters, er.n_kpts);

    // Frozen first-green energy band (REFERENCE.md; replaced when CP2K pin
    // exists). Filled after first green run — placeholder until measured.
    // Gate A4 uses a soft absolute band once set.

    const double h = 2e-3;
    const auto san = gpw_stress(atoms, make_cfg(), h);
    const auto sfd = gpw_stress_fd(atoms, make_cfg(), h);

    double max_diag_err = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int a = 0; a < 3; ++a) {
        const double an = san[static_cast<std::size_t>(a)];
        const double fd = sfd[static_cast<std::size_t>(a)];
        const double err = std::abs(an - fd);
        max_diag_err = std::max(max_diag_err, err);
        std::printf("  σ_%s  analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[a], an, fd, err);
        check(std::isfinite(an) && std::isfinite(fd), "A1 finite stress");
    }
    for (int k = 3; k < 6; ++k)
        check(std::abs(san[static_cast<std::size_t>(k)]) <= 1e-8,
              "A3 off-diagonal ~ 0");

    constexpr double kAnFdTol = 1.0e-4; // Ha/Bohr3 (mesh residual + O(h^2))
    std::printf("max|σ_an-σ_fd|=%.3e Ha/Bohr3\n", max_diag_err);
    check(max_diag_err <= kAnFdTol, "A2 analytic vs FD max diag");

    // A4: frozen first-green absolute pins (filled after first green).
    // Energy pin from this setup; stress diag mean as integrity band.
    // First-green frozen pins (REFERENCE.md; 100 Ry mesh 1x1x2).
    constexpr double kEPin = -15.3483039265; // Ha
    constexpr double kETol = 1.0e-3;
    const double dE = std::abs(er.e_total - kEPin);
    std::printf("A4: |E - pin|=%.3e Ha (pin %.10f)\n", dE, kEPin);
    check(dE <= kETol, "A4 absolute e_tot within 1e-3 Ha of pin");

    constexpr double kSxxPin = 8.83135454e-04;
    constexpr double kSzzPin = 2.60970168e-04;
    constexpr double kSTol = 1.0e-4; // Ha/Bohr3 soft absolute band
    check(std::abs(san[0] - kSxxPin) <= kSTol, "A4 |sigma_xx - pin|");
    check(std::abs(san[1] - kSxxPin) <= kSTol, "A4 |sigma_yy - pin|");
    check(std::abs(san[2] - kSzzPin) <= kSTol, "A4 |sigma_zz - pin|");

    const double max_diag =
        std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
    std::printf("max|sigma_diag|=%.3e Ha/Bohr3\n", max_diag);
    check(max_diag > 1e-6, "A4 stress magnitude floor");

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-kpoint-stress  E=%.10f  max|dσ|=%.3e  max|σ|=%.3e\n",
                er.e_total, max_diag_err, max_diag);
    return 0;
}
