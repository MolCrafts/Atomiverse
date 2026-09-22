/// @file regressions/gpw/ot-smearing/reg_ot-smearing.cpp
/// @brief END-TO-END: multi-k GPW RKS OT + Fermi–Dirac metallic smearing (BCC Na).
///
/// See REFERENCE.md. Public headers only. Requires ATV_BUILD_GPW.
/// Gates: OT+FD vs DIIS+FD free-energy identity, electron count, frozen pin.
/// ctest never launches CP2K.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
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

ATV::cpu::gpw::GpwDft base_method(double a) {
    ATV::cpu::gpw::GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = a;
    m.cutoff_ry = 150.0;
    m.n_occ = 9; // Ne/2 for GTH-PBE-q9 × 2
    m.max_iter = 120;
    m.tol = 1e-6;
    m.eps_default = 1e-7;
    m.xc = ATV::cpu::gpw::XcKind::PADE;
    m.diis_depth = 6;
    m.mix = 0.4;
    m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
    m.k_time_reversal = true;
    m.smearing = ATV::cpu::gpw::Smearing::fermi_dirac;
    m.degauss = 0.02; // Ry
    m.extra_bands = 4;
    return m;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GpwDft;
    using cpu::gpw::Smearing;

    const double a_ang = 4.225;
    const double a = a_ang * kA2B;
    const double n_elec = 18.0;
    const auto atoms = na_bcc(a);

    // ── A1: build accepts RKS OT + FD ──────────────────────────────────────
    {
        GpwDft m = base_method(a);
        m.use_ot = true;
        try {
            m.build(atoms);
            check(true, "A1 build accepts RKS OT + FD smear");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "A1 threw: %s\n", e.what());
            check(false, "A1 build accepts RKS OT + FD smear");
        }
    }

    // ── A2/A3: OT+FD vs DIIS+FD free-energy identity ───────────────────────
    double f_ot = 0.0;
    double f_diis = 0.0;
    double wall_ot = 0.0;
    {
        GpwDft diis = base_method(a);
        diis.use_ot = false;
        diis.build(atoms);
        const auto t0 = std::chrono::steady_clock::now();
        const auto& rd = diis.run(atoms);
        const auto t1 = std::chrono::steady_clock::now();
        (void)t0;
        (void)t1;
        check(rd.converged, "A2 DIIS+FD converged");
        f_diis = rd.e_total;
        check(std::fabs(rd.n_elec_occ - n_elec) < 1e-5, "A2 DIIS Ne");

        GpwDft ot = base_method(a);
        ot.use_ot = true;
        ot.max_iter = 200;
        ot.build(atoms);
        const auto t2 = std::chrono::steady_clock::now();
        const auto& ro = ot.run(atoms);
        const auto t3 = std::chrono::steady_clock::now();
        wall_ot = std::chrono::duration<double>(t3 - t2).count();
        check(ro.converged, "A2 OT+FD converged");
        f_ot = ro.e_total;
        check(std::fabs(ro.n_elec_occ - n_elec) < 1e-5, "A4 OT Ne");
        check(std::isfinite(ro.e_fermi) && std::fabs(ro.e_fermi) < 1e3,
              "A2 finite e_fermi");
        check(std::isfinite(ro.e_entropy), "A2 finite e_entropy");
        check(ro.n_kpts > 1, "A2 multi-k active");

        const double d = std::fabs(f_ot - f_diis);
        std::printf("A3: F_OT=%.10f  F_DIIS=%.10f  |Δ|=%.3e Ha  "
                    "iters ot/diis=%d/%d  wall_ot=%.3f s\n",
                    f_ot, f_diis, d, ro.iters, rd.iters, wall_ot);
        check(d <= 1e-4, "A3 |F_OT - F_DIIS| <= 1e-4 Ha");
    }

    // ── A5: frozen first-green free-energy band ────────────────────────────
    // First green 2026-07-26 (REFERENCE.md); OT ≡ DIIS to ~1e-7 Ha.
    {
        constexpr double kFrozenFHa = -92.1263304935; // OT first green
        constexpr double kAbsTol = 1e-3;
        const double d = std::fabs(f_ot - kFrozenFHa);
        std::printf("A5: F_OT=%.10f  frozen=%.10f  |Δ|=%.3e\n", f_ot,
                    kFrozenFHa, d);
        check(d <= kAbsTol, "A5 |F_OT - frozen| <= 1e-3 Ha");
    }

    // ── A6: multi-k UKS OT + metallic smear accepted at build ─────────────
    // Product path is `ot-uks-smearing`; build must not reject the combo.
    {
        try {
            GpwDft m = base_method(a);
            m.n_occ = 0;
            m.n_alpha = 10;
            m.n_beta = 8;
            m.use_ot = true;
            m.smearing = Smearing::fermi_dirac;
            m.degauss = 0.02;
            m.build(atoms);
            check(true, "A6 multi-k UKS OT + FD smear build accepted");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "A6 threw: %s\n", e.what());
            check(false, "A6 multi-k UKS OT + FD smear build accepted");
        }
    }

    if (g_fail) {
        std::printf("FAIL: ot-smearing (%d checks)\n", g_fail);
        return 1;
    }
    std::printf("PASS: ot-smearing (BCC Na OT+FD ≡ DIIS+FD)\n");
    return 0;
}
