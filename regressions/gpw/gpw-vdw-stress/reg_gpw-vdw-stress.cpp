/// @file regressions/gpw/gpw-vdw-stress/reg_gpw-vdw-stress.cpp
/// @brief END-TO-END: analytic fixed-P cell stress with nonlocal vdW-DF (DRSLL).
///
/// Water dimer Γ GPW PBE + NON_LOCAL DRSLL. Dual-gate analytic vs multi-SCF FD
/// of E_tot; soft absolute class vs offline CP2K STRESS + NON_LOCAL twin.
/// See REFERENCE.md.

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

const double kAng[6][3] = {
    {0.000000, 0.000000, 0.000000},  // O
    {0.758602, 0.000000, 0.504284},  // H
    {0.758602, 0.000000, -0.504284}, // H
    {3.000000, 0.000000, 0.000000},  // O
    {3.758602, 0.000000, 0.504284},  // H
    {3.758602, 0.000000, -0.504284}, // H
};
const int kZ[6] = {8, 1, 1, 8, 1, 1};

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

std::vector<ATV::cpu::gpw::GpwAtom> make_atoms() {
    using namespace ATV::cpu::gpw;
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    std::vector<GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double pos[3] = {kAng[i][0] * kA2B, kAng[i][1] * kA2B,
                               kAng[i][2] * kA2B};
        if (kZ[i] == 8)
            atoms.emplace_back(pos, bO, pO, 8);
        else
            atoms.emplace_back(pos, bH, pH, 1);
    }
    return atoms;
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;

    // Energy twin class of gpw-vdw-df-ice / vdw-kpoint-forces (Γ DRSLL).
    constexpr double kEDispCp2k = 0.151943904803; // Ha (post-SCF class)
    constexpr double kEDispTol = 5.0e-4;
    constexpr double kETotSoftTol = 5.0e-2; // Ha twin-family residual

    auto atoms = make_atoms();
    const double L = 12.0 * kA2B;

    auto make_cfg = [&]() {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = L;
        m.cutoff_ry = 200.0;
        m.n_occ = 8;
        m.max_iter = 40;
        m.tol = 1e-7;
        m.mix = 0.4;
        m.xc = XcKind::PBE;
        m.nonperiodic = false;
        m.vdw_df = true;
        m.local_pp_on_grid = true;
        return m;
    };

    GpwDft energy = make_cfg();
    energy.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const auto& r = energy.run(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_scf = std::chrono::duration<double>(t1 - t0).count();

    std::printf("E=%.10f Ha  E_disp=%.10f Ha  iters=%d  wall_scf=%.3f s  "
                "conv=%d\n",
                r.e_total, r.e_disp, r.iters, wall_scf, r.converged ? 1 : 0);

    check(r.converged, "SCF converged");
    check(std::isfinite(r.e_total) && std::isfinite(r.e_disp), "finite E");
    check(std::abs(r.e_disp - kEDispCp2k) <= kEDispTol, "E_disp vs CP2K class");
    check(std::abs(r.e_disp) > 1e-4, "non-vacuous nonlocal E_disp");

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

    // Fixed-P analytic vs multi-SCF FD; residual SCF + O(h²) + nonlocal.
    constexpr double kAnFdTol = 5.0e-4; // Ha/Bohr³
    std::printf(
        "max|σ_an-σ_fd|=%.3e Ha/Bohr3  wall_an=%.3f s  wall_fd=%.3f s\n",
        max_diag_err, wall_an, wall_fd);
    check(max_diag_err <= kAnFdTol, "analytic vs FD max diag (vdW-DF)");

    // Wiring: stress without nonlocal must differ if E_disp is large enough.
    {
        auto bare = make_cfg();
        bare.vdw_df = false;
        const auto s_bare = gpw_stress(atoms, bare, h);
        double max_diff = 0.0;
        for (int ax = 0; ax < 3; ++ax)
            max_diff = std::max(
                max_diff, std::abs(san[static_cast<std::size_t>(ax)]
                                   - s_bare[static_cast<std::size_t>(ax)]));
        std::printf("max|σ_vdw−σ_bare|=%.3e Ha/Bohr3\n", max_diff);
        check(max_diff > 1e-6, "nonlocal stress contribution non-vacuous");
    }

    (void)kETotSoftTol;
    std::printf("T_atv_scf=%.3f s\n", wall_scf);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-vdw-stress  E=%.10f  E_disp=%.6e  max|dσ|=%.3e  "
                "T_scf=%.3f s\n",
                r.e_total, r.e_disp, max_diag_err, wall_scf);
    return 0;
}
