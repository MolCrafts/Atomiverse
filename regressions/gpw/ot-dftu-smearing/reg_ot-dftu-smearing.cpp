/// @file regressions/gpw/ot-dftu-smearing/reg_ot-dftu-smearing.cpp
/// @brief END-TO-END: multi-k GPW RKS OT + Dudarev DFT+U + FD metallic smear.
///
/// Public `GpwDft` only. Offline CP2K twin under
/// `tools/cp2k_oracle/inputs/ot-dftu-smearing/` (REFERENCE.md). ctest never
/// spawns CP2K.
///
/// Gates:
///   A0 build accepts OT+U+FD
///   A1–A4 OT+U+FD ≡ DIIS+U+FD free energy, Ne, e_dftu (κ = 1/4)
///   A5 activation |F(U)−F(0)| on OT+FD
///   A6 frozen free-energy pin (first green)
///   A7 multi-k UKS OT+U+FD accepted at build (product: ot-uks-dftu-smearing)
///
/// Exit non-zero on failure. Target ≤ 60 s Release.

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

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

std::vector<ATV::cpu::gpw::GpwAtom> h2_toy(double L_bohr) {
    using namespace ATV;
    const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
    const double r0[3] = {0.5 * L_bohr - 0.7, 0.5 * L_bohr, 0.5 * L_bohr};
    const double r1[3] = {0.5 * L_bohr + 0.7, 0.5 * L_bohr, 0.5 * L_bohr};
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.emplace_back(r0, bH, pH, 1);
    atoms.emplace_back(r1, bH, pH, 1);
    return atoms;
}

ATV::cpu::gpw::GpwDft base_method(double L, double u_ev, bool with_u) {
    using namespace ATV;
    cpu::gpw::GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = L;
    m.cutoff_ry = 100.0;
    m.n_occ = 1;
    m.max_iter = 160;
    m.tol = 1e-7;
    m.mix = 0.35;
    m.xc = cpu::gpw::XcKind::PADE;
    m.k_points.push_back({0.25, 0.0, 0.0});
    m.k_point_weights.push_back(1.0);
    m.smearing = cpu::gpw::Smearing::fermi_dirac;
    m.degauss = 0.02; // Ry
    m.extra_bands = 2;
    if (with_u)
        m.plus_u.push_back(
            cpu::gpw::PlusU{molrs::Element::H, /*l=*/0, u_ev});
    return m;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GpwResult;
    using cpu::gpw::Smearing;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const double L = 8.0;
    auto atoms = h2_toy(L);
    constexpr double kU_ev = 4.0;
    constexpr double kNe = 2.0;

    // ── A0: build accepts RKS OT + U + FD ──────────────────────────────────
    {
        GpwDft m = base_method(L, kU_ev, true);
        m.use_ot = true;
        try {
            m.build(atoms);
            check(true, "A0 build accepts RKS OT + U + FD");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "A0 threw: %s\n", e.what());
            check(false, "A0 build accepts RKS OT + U + FD");
        }
    }

    // ── A1–A4: OT+U+FD ≡ DIIS+U+FD ─────────────────────────────────────────
    double f_ot = 0.0;
    double e_dftu_ot = 0.0;
    double wall_ot = 0.0;
    {
        GpwDft diis = base_method(L, kU_ev, true);
        diis.use_ot = false;
        diis.build(atoms);
        const GpwResult rd = diis.run(atoms);
        check(rd.converged, "A1 DIIS+U+FD converged");
        check(std::fabs(rd.n_elec_occ - kNe) < 1e-5, "A4 DIIS Ne");

        GpwDft ot = base_method(L, kU_ev, true);
        ot.use_ot = true;
        ot.max_iter = 220;
        ot.build(atoms);
        const auto t2 = clock::now();
        const GpwResult ro = ot.run(atoms);
        const auto t3 = clock::now();
        wall_ot = std::chrono::duration<double>(t3 - t2).count();
        check(ro.converged, "A1 OT+U+FD converged");
        f_ot = ro.e_total;
        e_dftu_ot = ro.e_dftu;
        check(std::fabs(ro.n_elec_occ - kNe) < 1e-5, "A4 OT Ne");
        check(std::isfinite(ro.e_fermi), "A1 finite e_fermi");
        check(std::isfinite(ro.e_entropy), "A1 finite e_entropy");
        check(std::fabs(ro.e_dftu) > 1e-8, "A3 e_dftu non-trivial");
        check(std::fabs(ro.e_dftu - rd.e_dftu) <= 1e-3,
              "A3 |e_dftu OT − DIIS| ≤ 1e-3 Ha");

        const double d = std::fabs(ro.e_total - rd.e_total);
        std::printf(
            "A2: F_OT=%.12f  F_DIIS=%.12f  |Δ|=%.3e Ha  "
            "e_dftu_ot=%.8f  iters ot/diis=%d/%d  wall_ot=%.3f s\n",
            ro.e_total, rd.e_total, d, ro.e_dftu, ro.iters, rd.iters,
            wall_ot);
        check(d <= 1e-4, "A2 |F_OT − F_DIIS| ≤ 1e-4 Ha");
    }

    // ── A5: activation |F(U)−F(0)| under OT+FD ─────────────────────────────
    {
        GpwDft m0 = base_method(L, kU_ev, false);
        m0.use_ot = true;
        m0.max_iter = 220;
        m0.build(atoms);
        const GpwResult r0 = m0.run(atoms);
        check(r0.converged, "A5 OT+FD (U=0) converged");
        const double dU = std::fabs(f_ot - r0.e_total);
        std::printf("A5: F(U=0)=%.12f  F(U=4)=%.12f  |Δ|=%.6e Ha\n",
                    r0.e_total, f_ot, dU);
        check(dU >= 1e-4, "A5 |F(U)−F(0)| activation floor");
    }

    // ── A6: frozen first-green free-energy pin ─────────────────────────────
    // First green 2026-07-26 (REFERENCE.md); OT ≡ DIIS to ~2e-7 Ha.
    {
        // re-pinned 2026-07-31: variational +U scatter (pre-fix -1.081998158198)
        constexpr double kFrozenFHa = -1.400026280953;
        constexpr double kAbsTol = 1e-3;
        const double d = std::fabs(f_ot - kFrozenFHa);
        std::printf("A6: F_OT=%.12f  frozen=%.12f  |Δ|=%.3e  e_dftu=%.8f\n",
                    f_ot, kFrozenFHa, d, e_dftu_ot);
        check(d <= kAbsTol, "A6 |F_OT − frozen| ≤ 1e-3 Ha");
    }

    // ── A7: multi-k UKS OT + U + FD accepted (product ot-uks-dftu-smearing) ─
    {
        try {
            GpwDft m = base_method(L, kU_ev, true);
            m.n_occ = 0;
            m.n_alpha = 1;
            m.n_beta = 1;
            m.use_ot = true;
            m.smearing = Smearing::fermi_dirac;
            m.degauss = 0.02;
            m.build(atoms);
            check(true, "A7 multi-k UKS OT + U + FD accepted at build");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "A7 threw: %s\n", e.what());
            check(false, "A7 multi-k UKS OT + U + FD accepted at build");
        }
    }

    const auto t1 = clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    std::printf("wall = %.3f s  wall_ot = %.3f s  fails=%d\n", wall, wall_ot,
                g_fail);
    if (g_fail != 0) {
        std::fprintf(stderr, "reg_ot-dftu-smearing: %d check(s) failed\n",
                     g_fail);
        return 1;
    }
    std::printf("reg_ot-dftu-smearing: PASS\n");
    return 0;
}
