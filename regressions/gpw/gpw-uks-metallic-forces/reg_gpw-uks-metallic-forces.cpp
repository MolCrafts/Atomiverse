/// @file regressions/gpw/gpw-uks-metallic-forces/reg_gpw-uks-metallic-forces.cpp
/// @brief END-TO-END: multi-k UKS free-energy analytic ionic forces under FD.
///
/// BCC Na (GTH-PBE-q9 / DZVP-MOLOPT-SR-GTH), collinear UKS + Fermi–Dirac
/// metallic smear, Monkhorst–Pack 2×2×2. Gates free-energy pin, electron
/// count, fractional f_k payload, finite free-energy forces, Newton-3,
/// closed-shell UKS+smear ≡ RKS+smear force identity. See REFERENCE.md.
/// Public: `GpwDft` + `gpw_forces_analytic` / `gpw_forces_from_scf`.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859; // Å → Bohr

// Frozen ATV pins (Release, OMP=8) — filled after first green; see REFERENCE.
// Free energy matches gpw-uks-smearing (same electronic path).
constexpr double kFPin = -92.126330608004; // Ha (Mermin free energy)
constexpr double kFTol = 1e-4;             // Ha
constexpr double kNeTol = 1e-5;
// Force magnitude floor: BCC Na is nearly force-free by symmetry at
// equilibrium; the dual gate checks |sum F| and closed-shell identity.
// A mild bond stretch is not used — the product pin is equilibrium Na.
constexpr double kSumFTol = 1e-5;             // Ha/Bohr
constexpr double kClosedShellForceTol = 5e-4; // Ha/Bohr

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

/// Slightly off-equilibrium H₂ for a nonzero free-energy force identity gate.
std::vector<ATV::cpu::gpw::GpwAtom> h2_toy(double L_bohr) {
    using namespace ATV;
    const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
    const double r0[3] = {0.5 * L_bohr - 0.55, 0.5 * L_bohr, 0.5 * L_bohr};
    const double r1[3] = {0.5 * L_bohr + 0.55, 0.5 * L_bohr, 0.5 * L_bohr};
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.emplace_back(r0, bH, pH, 1);
    atoms.emplace_back(r1, bH, pH, 1);
    return atoms;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::gpw_forces_analytic;
    using cpu::gpw::gpw_forces_from_scf;
    using cpu::gpw::GpwDft;
    using cpu::gpw::Smearing;
    using cpu::gpw::XcKind;

    const auto t0 = std::chrono::steady_clock::now();

    // ── A0: closed-shell multi-k UKS+smear forces ≡ RKS+smear (H₂) ────────
    {
        const double L = 10.0;
        auto atoms = h2_toy(L);

        GpwDft rks;
        rks.cell[0] = rks.cell[1] = rks.cell[2] = L;
        rks.cutoff_ry = 100.0;
        rks.n_occ = 1;
        rks.max_iter = 80;
        rks.tol = 1e-7;
        rks.xc = XcKind::PADE;
        rks.local_pp_on_grid = true;
        rks.k_mesh[0] = rks.k_mesh[1] = 1;
        rks.k_mesh[2] = 2;
        rks.smearing = Smearing::fermi_dirac;
        rks.degauss = 0.02;

        GpwDft uks;
        uks.cell[0] = uks.cell[1] = uks.cell[2] = L;
        uks.cutoff_ry = 100.0;
        uks.n_alpha = 1;
        uks.n_beta = 1;
        uks.max_iter = 80;
        uks.tol = 1e-7;
        uks.xc = XcKind::PADE;
        uks.local_pp_on_grid = true;
        uks.k_mesh[0] = uks.k_mesh[1] = 1;
        uks.k_mesh[2] = 2;
        uks.smearing = Smearing::fermi_dirac;
        uks.degauss = 0.02;

        const auto Fr = gpw_forces_analytic(atoms, rks);
        const auto Fu = gpw_forces_analytic(atoms, uks);
        check(Fr.size() == 2 && Fu.size() == 2, "A0 two force vectors each");
        double max_d = 0.0;
        for (int a = 0; a < 2; ++a)
            for (int k = 0; k < 3; ++k)
                max_d = std::max(
                    max_d, std::abs(Fr[static_cast<std::size_t>(a)][k]
                                    - Fu[static_cast<std::size_t>(a)][k]));
        std::printf("A0 closed-shell max|dF| = %.3e Ha/Bohr\n", max_d);
        check(max_d <= kClosedShellForceTol,
              "A0 closed-shell UKS+smear ≡ RKS+smear forces");
        check(Fu[0][0] < 0.0 && Fu[1][0] > 0.0,
              "A0 restoring bond force under free-energy smear");
        check(std::abs(Fu[0][0] + Fu[1][0]) <= 1e-5, "A0 Newton-3 F_x");
    }

    // ── A1: BCC Na multi-k UKS + FD free energy + analytic forces ─────────
    {
        const double a_ang = 4.225;
        const double a_bohr = a_ang * kA2B;
        auto atoms = na_bcc(a_bohr);

        auto make_cfg = [&]() {
            GpwDft m;
            m.cell[0] = m.cell[1] = m.cell[2] = a_bohr;
            m.cutoff_ry = 150.0;
            m.n_alpha = 10;
            m.n_beta = 8;
            m.max_iter = 80;
            m.tol = 1e-6;
            m.xc = XcKind::PADE;
            m.local_pp_on_grid = true;
            m.eps_default = 1e-7;
            m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
            m.smearing = Smearing::fermi_dirac;
            m.degauss = 0.02; // Ry
            return m;
        };

        const auto t_scf0 = std::chrono::steady_clock::now();
        GpwDft energy = make_cfg();
        energy.build(atoms);
        const auto er = energy.run(atoms);
        const auto t_scf1 = std::chrono::steady_clock::now();
        const double t_scf_s =
            std::chrono::duration<double>(t_scf1 - t_scf0).count();

        check(er.converged, "A1 Na UKS+smear SCF converged");
        check(std::isfinite(er.e_total), "A1 finite free energy");
        check(std::abs(er.e_total - kFPin) <= kFTol, "A1 free-energy pin");
        check(std::abs(er.n_elec_occ - 18.0) <= kNeTol, "A1 Ne = 18");
        check(std::isfinite(er.e_fermi) && std::abs(er.e_fermi) < 1e3,
              "A1 finite Fermi level");
        check(std::isfinite(er.e_entropy), "A1 finite entropy term");
        check(er.n_kpts >= 1, "A1 n_kpts >= 1");
        check(static_cast<int>(er.f_k.size()) == er.n_kpts, "A1 f_k payload");
        check(static_cast<int>(er.f_k_beta.size()) == er.n_kpts,
              "A1 f_k_beta payload");
        check(static_cast<int>(er.C_occ_k.size()) == er.n_kpts, "A1 C_occ_k");
        check(!er.f_k.empty() && !er.f_k[0].empty(),
              "A1 fractional occupations non-empty");

        std::printf("E(Na UKS FD free) = %.12f Ha  iters=%d  nk=%d  "
                    "EF=%.6f  -TS=%.3e  Ne=%.6f  T_scf=%.3f s\n",
                    er.e_total, er.iters, er.n_kpts, er.e_fermi, er.e_entropy,
                    er.n_elec_occ, t_scf_s);

        const auto t_f0 = std::chrono::steady_clock::now();
        const auto Fa = gpw_forces_from_scf(atoms, make_cfg(), er);
        const auto t_f1 = std::chrono::steady_clock::now();
        const double t_force_s =
            std::chrono::duration<double>(t_f1 - t_f0).count();

        check(Fa.size() == 2, "A1 two force vectors");
        for (int a = 0; a < 2; ++a)
            for (int k = 0; k < 3; ++k)
                check(std::isfinite(Fa[static_cast<std::size_t>(a)][k]),
                      "A1 finite force component");

        std::printf("  F0 = (% .12e, % .12e, % .12e)\n", Fa[0][0], Fa[0][1],
                    Fa[0][2]);
        std::printf("  F1 = (% .12e, % .12e, % .12e)\n", Fa[1][0], Fa[1][1],
                    Fa[1][2]);
        std::printf("  T_force_assembly = %.3f s\n", t_force_s);

        double sum[3] = {Fa[0][0] + Fa[1][0], Fa[0][1] + Fa[1][1],
                         Fa[0][2] + Fa[1][2]};
        const double sum_mag =
            std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
        std::printf("A1 |sum F| = %.3e Ha/Bohr\n", sum_mag);
        check(sum_mag <= kSumFTol, "A1 |sum F| residual (Newton-3)");

        // Fair wall class: SCF + force assembly (matched twin is ENERGY_FORCE).
        // One SCF + from_scf (no second cold SCF) — same density as energy pin.
        const double t_atv = t_scf_s + t_force_s;
        std::printf("ATV_WALL_SCF_FORCE %.6f s\n", t_atv);
    }

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("total wall %.3f s  fails=%d\n",
                std::chrono::duration<double>(t1 - t0).count(), g_fail);
    return g_fail == 0 ? 0 : 1;
}
