/// @file regressions/gpw/ot-uks-kpoints/reg_ot-uks-kpoints.cpp
/// @brief END-TO-END: multi-k collinear UKS GPW OT vs DIIS identities.
///
/// Public API only. See REFERENCE.md. Requires ATV_BUILD_GPW (+ HF + PW).
/// ctest never launches CP2K.

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
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

// Compact high-spin H₂ (DZV-GTH-PADE) — same class as gpw-kpoints-uks.
std::vector<ATV::cpu::gpw::GpwAtom> make_h2(double L) {
    using namespace ATV;
    const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
    const double r0[3] = {0.5 * L - 0.7, 0.5 * L, 0.5 * L};
    const double r1[3] = {0.5 * L + 0.7, 0.5 * L, 0.5 * L};
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.emplace_back(r0, bH, pH, 1);
    atoms.emplace_back(r1, bH, pH, 1);
    return atoms;
}

ATV::cpu::gpw::GpwDft base_uks(double L) {
    ATV::cpu::gpw::GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = L;
    m.cutoff_ry = 100.0;
    m.n_occ = 0;
    m.n_alpha = 2;
    m.n_beta = 0;
    m.max_iter = 120;
    m.tol = 1e-8;
    m.xc = ATV::cpu::gpw::XcKind::PADE;
    m.eps_default = 1e-7;
    return m;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GpwDft;

    const double L = 8.0; // Bohr
    const auto atoms = make_h2(L);

    // ── A1: explicit-Γ complex UKS OT ≡ real Γ UKS OT ──────────────────────
    {
        GpwDft real_ot = base_uks(L);
        real_ot.use_ot = true;
        real_ot.max_iter = 200;
        real_ot.build(atoms);
        const auto& rr = real_ot.run(atoms);
        check(rr.converged, "A1 real Γ UKS OT converged");

        GpwDft k_ot = base_uks(L);
        k_ot.use_ot = true;
        k_ot.k_points.push_back({0.0, 0.0, 0.0});
        k_ot.k_point_weights.push_back(1.0);
        k_ot.max_iter = 200;
        k_ot.build(atoms);
        const auto& rk = k_ot.run(atoms);
        check(rk.converged, "A1 k-path Γ UKS OT converged");
        const double d = std::abs(rr.e_total - rk.e_total);
        std::printf("A1: real Γ UKS OT = %.10f  k-Γ OT = %.10f  |Δ|=%.3e Ha\n",
                    rr.e_total, rk.e_total, d);
        check(d <= 1e-6, "A1 |E_realOT - E_kGammaOT| <= 1e-6 Ha");
    }

    // ── A2: multi-k UKS OT vs multi-k UKS DIIS (κ = (1/4,0,0)) ─────────────
    double e_ot = 0.0;
    double e_diis = 0.0;
    {
        GpwDft diis = base_uks(L);
        diis.k_points.push_back({0.25, 0.0, 0.0});
        diis.k_point_weights.push_back(1.0);
        diis.use_ot = false;
        diis.build(atoms);
        const auto& rd = diis.run(atoms);
        check(rd.converged, "A2 multi-k UKS DIIS converged");
        e_diis = rd.e_total;

        GpwDft ot = base_uks(L);
        ot.k_points.push_back({0.25, 0.0, 0.0});
        ot.k_point_weights.push_back(1.0);
        ot.use_ot = true;
        ot.max_iter = 200;
        ot.build(atoms);
        const auto& ro = ot.run(atoms);
        check(ro.converged, "A2 multi-k UKS OT converged");
        e_ot = ro.e_total;
        check(ro.n_kpts == 1, "A2 n_kpts == 1");
        check(!ro.k_frac.empty() && std::abs(ro.k_frac[0][0] - 0.25) < 1e-12,
              "A2 k_frac payload");
        const double d = std::abs(e_ot - e_diis);
        std::printf("A2: E_OT = %.10f  E_DIIS = %.10f  |Δ|=%.3e Ha  "
                    "iters ot/diis=%d/%d  s2=%.6f\n",
                    e_ot, e_diis, d, ro.iters, rd.iters, ro.s2);
        check(d <= 1e-4, "A2 |E_OT - E_DIIS| <= 1e-4 Ha");
        check(std::isfinite(e_ot), "A2 finite E_OT");
        check(std::abs(ro.s2 - rd.s2) <= 1e-6, "A2 S2 OT ≡ DIIS");
    }

    // ── A3: Monkhorst–Pack 2×1×1 TR mesh UKS OT vs DIIS ────────────────────
    {
        GpwDft diis = base_uks(L);
        diis.k_mesh[0] = 2;
        diis.k_mesh[1] = 1;
        diis.k_mesh[2] = 1;
        diis.k_time_reversal = true;
        diis.use_ot = false;
        diis.build(atoms);
        const auto& rd = diis.run(atoms);
        check(rd.converged, "A3 MP UKS DIIS converged");

        GpwDft ot = base_uks(L);
        ot.k_mesh[0] = 2;
        ot.k_mesh[1] = 1;
        ot.k_mesh[2] = 1;
        ot.k_time_reversal = true;
        ot.use_ot = true;
        ot.max_iter = 200;
        ot.build(atoms);
        const auto& ro = ot.run(atoms);
        check(ro.converged, "A3 MP UKS OT converged");
        const double d = std::abs(ro.e_total - rd.e_total);
        std::printf("A3: MP OT = %.10f  MP DIIS = %.10f  |Δ|=%.3e Ha  "
                    "n_kpts=%d\n",
                    ro.e_total, rd.e_total, d, ro.n_kpts);
        check(d <= 1e-4, "A3 |E_OT - E_DIIS| MP mesh <= 1e-4 Ha");
        check(ro.n_kpts >= 1, "A3 n_kpts >= 1");
    }

    // ── A4: frozen first-green absolute band (until offline CP2K pin) ──────
    // Measured 2026-07-26; replace with CP2K OT+KPOINTS+UKS when available
    // (tools/cp2k_oracle/inputs/ot-uks-kpoints/).
    {
        constexpr double kFrozenOtHa = -0.7061343842;
        constexpr double kAbsTol = 1e-3;
        const double d = std::abs(e_ot - kFrozenOtHa);
        std::printf("A4: multi-k UKS OT E = %.10f Ha  frozen = %.10f  "
                    "|Δ|=%.3e\n",
                    e_ot, kFrozenOtHa, d);
        check(d <= kAbsTol, "A4 |E_OT - frozen first-green| <= 1e-3 Ha");
    }

    if (g_fail) {
        std::fprintf(stderr, "FAIL ot-uks-kpoints (%d checks)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw ot-uks-kpoints (multi-k UKS OT identities)\n");
    return 0;
}
