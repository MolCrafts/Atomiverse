/// @file regressions/gpw/gpw-kpoint-uks-stress/reg_gpw-kpoint-uks-stress.cpp
/// @brief END-TO-END: multi-k UKS GPW analytic cell stress (dual-gate + identity).
///
/// High-spin H₂ (Nα=2, Nβ=0), PADE LSD, DZV-GTH-PADE, mesh 1×1×2.
/// Gates: dual-gate analytic fixed-P_k^σ vs multi-SCF strain-FD; closed-shell
/// k-UKS ≡ k-RKS stress; off-diagonal ~0. See REFERENCE.md.

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

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

std::vector<ATV::cpu::gpw::GpwAtom> h2_hs(double L) {
    using namespace ATV::cpu::gpw;
    const GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
    const double r0[3] = {0.5 * L - 0.5, 0.5 * L, 0.5 * L};
    const double r1[3] = {0.5 * L + 0.5, 0.5 * L, 0.5 * L};
    return {GpwAtom(r0, bH, pH, 1), GpwAtom(r1, bH, pH, 1)};
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const double L = 12.0;
    const auto atoms = h2_hs(L);

    auto make_uks = [&]() {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = L;
        m.cutoff_ry = 100.0;
        m.n_alpha = 2;
        m.n_beta = 0;
        m.max_iter = 60;
        m.tol = 1e-8;
        m.xc = XcKind::PADE;
        m.local_pp_on_grid = true;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        return m;
    };

    // A1: SCF + payload
    GpwDft energy = make_uks();
    energy.build(atoms);
    const auto& er = energy.run(atoms);
    check(er.converged, "A1 SCF converged");
    check(std::isfinite(er.e_total), "A1 finite e_total");
    check(er.n_kpts >= 1 && !er.C_occ_k.empty(), "A1 C_occ_k present");
    check(static_cast<int>(er.C_occ_k_beta.size()) == er.n_kpts,
          "A1 C_occ_k_beta present");
    std::printf("E(k-UKS 1x1x2, 100 Ry) = %.12f Ha  iters=%d  nk=%d\n",
                er.e_total, er.iters, er.n_kpts);

    // A2/A3: dual-gate analytic vs multi-SCF FD
    const double h = 2e-3;
    const auto san = gpw_stress(atoms, make_uks(), h);
    const auto sfd = gpw_stress_fd(atoms, make_uks(), h);

    double max_diag_err = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int a = 0; a < 3; ++a) {
        const double an = san[static_cast<std::size_t>(a)];
        const double fd = sfd[static_cast<std::size_t>(a)];
        const double err = std::abs(an - fd);
        max_diag_err = std::max(max_diag_err, err);
        std::printf("  σ_%s  analytic=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[a], an, fd, err);
        check(std::isfinite(an) && std::isfinite(fd), "A2 finite stress");
    }
    for (int k = 3; k < 6; ++k)
        check(std::abs(san[static_cast<std::size_t>(k)]) <= 1e-8,
              "A3 off-diagonal ~ 0");

    constexpr double kAnFdTol = 1.0e-4;
    std::printf("max|σ_an-σ_fd|=%.3e Ha/Bohr3\n", max_diag_err);
    check(max_diag_err <= kAnFdTol, "A2 analytic vs FD max diag");

    // A4: closed-shell multi-k UKS ≡ multi-k RKS
    {
        GpwDft rks;
        rks.cell[0] = rks.cell[1] = rks.cell[2] = L;
        rks.cutoff_ry = 100.0;
        rks.n_occ = 1;
        rks.max_iter = 60;
        rks.tol = 1e-8;
        rks.xc = XcKind::PADE;
        rks.local_pp_on_grid = true;
        rks.k_mesh[0] = 1;
        rks.k_mesh[1] = 1;
        rks.k_mesh[2] = 2;

        GpwDft uks_cs;
        uks_cs.cell[0] = uks_cs.cell[1] = uks_cs.cell[2] = L;
        uks_cs.cutoff_ry = 100.0;
        uks_cs.n_alpha = 1;
        uks_cs.n_beta = 1;
        uks_cs.max_iter = 60;
        uks_cs.tol = 1e-8;
        uks_cs.xc = XcKind::PADE;
        uks_cs.local_pp_on_grid = true;
        uks_cs.k_mesh[0] = 1;
        uks_cs.k_mesh[1] = 1;
        uks_cs.k_mesh[2] = 2;

        const auto sr = gpw_stress(atoms, rks, h);
        const auto su = gpw_stress(atoms, uks_cs, h);
        double max_d = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double d = std::abs(sr[static_cast<std::size_t>(a)]
                                      - su[static_cast<std::size_t>(a)]);
            max_d = std::max(max_d, d);
        }
        std::printf("A4 closed-shell k-UKS vs k-RKS max|dσ|=%.3e\n", max_d);
        check(max_d <= 5e-4, "A4 closed-shell k-UKS ≡ k-RKS");
    }

    // A5: magnitude floor
    const double max_diag =
        std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
    std::printf("max|sigma_diag|=%.3e Ha/Bohr3\n", max_diag);
    check(max_diag > 1e-8, "A5 stress magnitude floor");

    // A6: soft energy pin (filled after first green)
    constexpr double kEPin = -0.528389371092; // Ha; sibling k-uks-forces pin
    constexpr double kETol = 1.0e-3;
    const double dE = std::abs(er.e_total - kEPin);
    std::printf("A6: |E - pin|=%.3e Ha (pin %.10f)\n", dE, kEPin);
    check(dE <= kETol, "A6 absolute e_tot within 1e-3 Ha of soft pin");

    const auto t1 = clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    std::printf("wall=%.1f s\n", wall_s);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf(
        "PASS: gpw-kpoint-uks-stress  E=%.10f  max|dσ|=%.3e  max|σ|=%.3e\n",
        er.e_total, max_diag_err, max_diag);
    return 0;
}
