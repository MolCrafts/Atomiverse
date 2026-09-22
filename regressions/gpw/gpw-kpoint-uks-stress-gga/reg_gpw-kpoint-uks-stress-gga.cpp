/// @file regressions/gpw/gpw-kpoint-uks-stress-gga/reg_gpw-kpoint-uks-stress-gga.cpp
/// @brief END-TO-END: multi-k UKS GPW analytic cell stress for pure GGA
///        (RevPBE / RPBE / BLYP). Dual-gate + activation vs PBE.
///
/// High-spin H₂ (Nα=2, Nβ=0), DZV-GTH-PADE, mesh 1×1×2. See REFERENCE.md.

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

ATV::cpu::gpw::GpwDft make_uks(ATV::cpu::gpw::XcKind xc, double L) {
    using namespace ATV::cpu::gpw;
    GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = L;
    m.cutoff_ry = 100.0;
    m.n_alpha = 2;
    m.n_beta = 0;
    m.max_iter = 60;
    m.tol = 1e-8;
    m.xc = xc;
    m.local_pp_on_grid = true;
    m.k_mesh[0] = 1;
    m.k_mesh[1] = 1;
    m.k_mesh[2] = 2;
    return m;
}

struct GgaCase {
    ATV::cpu::gpw::XcKind xc;
    const char* name;
    double e_pin; // soft ATV first-green pin [Ha]; 0 = fill after run
};

} // namespace

int main() {
    using namespace ATV::cpu::gpw;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const double L = 12.0;
    const auto atoms = h2_hs(L);
    const double h = 2e-3;
    constexpr double kAnFdTol = 1.0e-4;
    constexpr double kETol = 1.0e-3;
    constexpr double kDeFloor = 1.0e-4;

    // Soft energy pins: ATV first-green self-pins (CP2K same-basis twin
    // pending — DZV-GTH-PADE not in BASIS_MOLOPT).
    GgaCase cases[] = {
        {XcKind::RevPBE, "RevPBE", -0.574211377589},
        {XcKind::RPBE, "RPBE", -0.575044480519},
        {XcKind::BLYP, "BLYP", -0.558884091671},
    };

    // Activation: RevPBE energy must differ from PBE on the same geometry.
    double e_pbe = 0.0;
    {
        GpwDft pbe = make_uks(XcKind::PBE, L);
        pbe.build(atoms);
        const auto& er = pbe.run(atoms);
        check(er.converged, "PBE SCF converged");
        e_pbe = er.e_total;
        std::printf("E(k-UKS PBE 1x1x2) = %.12f Ha\n", e_pbe);
    }

    double global_max_diag_err = 0.0;
    for (auto& c : cases) {
        std::printf("\n=== %s ===\n", c.name);
        GpwDft energy = make_uks(c.xc, L);
        energy.build(atoms);
        const auto& er = energy.run(atoms);
        check(er.converged, "SCF converged");
        check(std::isfinite(er.e_total), "finite e_total");
        check(er.n_kpts >= 1 && !er.C_occ_k.empty(), "C_occ_k present");
        check(static_cast<int>(er.C_occ_k_beta.size()) == er.n_kpts,
              "C_occ_k_beta present");
        std::printf("E(k-UKS %s 1x1x2, 100 Ry) = %.12f Ha  iters=%d  nk=%d\n",
                    c.name, er.e_total, er.iters, er.n_kpts);

        if (c.xc == XcKind::RevPBE) {
            const double dE = std::abs(er.e_total - e_pbe);
            std::printf("|E_RevPBE - E_PBE| = %.3e Ha\n", dE);
            check(dE >= kDeFloor, "activation |E_RevPBE-E_PBE| floor");
        }

        const auto san = gpw_stress(atoms, make_uks(c.xc, L), h);
        const auto sfd = gpw_stress_fd(atoms, make_uks(c.xc, L), h);

        double max_diag_err = 0.0;
        const char* names[3] = {"xx", "yy", "zz"};
        for (int a = 0; a < 3; ++a) {
            const double an = san[static_cast<std::size_t>(a)];
            const double fd = sfd[static_cast<std::size_t>(a)];
            const double err = std::abs(an - fd);
            max_diag_err = std::max(max_diag_err, err);
            std::printf("  σ_%s  analytic=% .8e  fd=% .8e  |d|=%.3e\n",
                        names[a], an, fd, err);
            check(std::isfinite(an) && std::isfinite(fd), "finite stress");
        }
        for (int k = 3; k < 6; ++k)
            check(std::abs(san[static_cast<std::size_t>(k)]) <= 1e-8,
                  "off-diagonal ~ 0");

        std::printf("max|σ_an-σ_fd|=%.3e Ha/Bohr3\n", max_diag_err);
        check(max_diag_err <= kAnFdTol, "analytic vs FD max diag");
        global_max_diag_err = std::max(global_max_diag_err, max_diag_err);

        const double max_diag =
            std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
        check(max_diag > 1e-8, "stress magnitude floor");

        // Soft pin: after first green, pins are written into REFERENCE and
        // recompiled; 0.0 means "record only" on bootstrap.
        if (c.e_pin != 0.0) {
            const double dE = std::abs(er.e_total - c.e_pin);
            std::printf("|E - pin|=%.3e Ha (pin %.10f)\n", dE, c.e_pin);
            check(dE <= kETol, "absolute e_tot soft pin");
        } else {
            std::printf("soft pin (record): E_%s = %.12f Ha\n", c.name,
                        er.e_total);
        }
    }

    const auto t1 = clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    std::printf("\nwall=%.1f s  global_max|dσ|=%.3e\n", wall_s,
                global_max_diag_err);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-kpoint-uks-stress-gga  max|dσ|=%.3e\n",
                global_max_diag_err);
    return 0;
}
