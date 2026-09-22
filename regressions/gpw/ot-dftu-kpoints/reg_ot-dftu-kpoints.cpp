/// @file regressions/gpw/ot-dftu-kpoints/reg_ot-dftu-kpoints.cpp
/// @brief END-TO-END: multi-k closed-shell GPW OT + Dudarev DFT+U.
///
/// Public `GpwDft` only. Offline CP2K twin under
/// `tools/cp2k_oracle/inputs/ot-dftu-kpoints/` (REFERENCE.md). ctest never
/// spawns CP2K.
///
/// Gates:
///   A1 Γ-limit: explicit k=Γ OT+U ≡ multi-k DIIS+U at Γ (complex path)
///   A2 multi-k OT+U ≡ multi-k DIIS+U (κ = (1/4,0,0))
///   A3 MP 2×2×2 activation |E(U)−E(0)| + absolute pin + e_dftu
///
/// Exit non-zero on failure. Target ≤ 60 s Release.

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

ATV::cpu::gpw::GpwDft base_plus_u(double L, double u_ev) {
    using namespace ATV;
    cpu::gpw::GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = L;
    cfg.cutoff_ry = 100.0;
    cfg.n_occ = 1;
    cfg.max_iter = 120;
    cfg.tol = 1e-7;
    cfg.mix = 0.35;
    cfg.xc = cpu::gpw::XcKind::PADE;
    cfg.plus_u.push_back(cpu::gpw::PlusU{molrs::Element::H, /*l=*/0, u_ev});
    return cfg;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GpwResult;
    using cpu::gpw::PlusU;
    using cpu::gpw::XcKind;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const double L = 8.0;
    auto atoms = h2_toy(L);
    constexpr double kU_ev = 4.0;

    // ── A1: explicit Γ OT+U ≡ multi-k DIIS+U at Γ ──────────────────────────
    {
        GpwDft diis = base_plus_u(L, kU_ev);
        diis.use_ot = false;
        diis.k_points = {{0.0, 0.0, 0.0}};
        diis.k_point_weights = {1.0};
        diis.build(atoms);
        const GpwResult rd = diis.run(atoms);

        GpwDft ot = base_plus_u(L, kU_ev);
        ot.use_ot = true;
        ot.k_points = {{0.0, 0.0, 0.0}};
        ot.k_point_weights = {1.0};
        ot.max_iter = 200;
        ot.build(atoms);
        const GpwResult ro = ot.run(atoms);

        check(rd.converged && ro.converged, "A1 both converge");
        const double d = std::fabs(ro.e_total - rd.e_total);
        std::printf("  A1 E_DIIS+U=%.12f  E_OT+U=%.12f  |Δ|=%.3e  "
                    "e_dftu_ot=%.6e\n",
                    rd.e_total, ro.e_total, d, ro.e_dftu);
        check(d <= 1e-4, "A1 |E_OT+U − E_DIIS+U| ≤ 1e-4 Ha");
        check(std::fabs(ro.e_dftu) > 1e-8, "A1 e_dftu non-trivial");
        check(std::fabs(ro.e_dftu - rd.e_dftu) <= 1e-3,
              "A1 |e_dftu OT − DIIS| ≤ 1e-3 Ha");
    }

    // ── A2: multi-k OT+U ≡ multi-k DIIS+U (κ = 1/4) ────────────────────────
    {
        GpwDft diis = base_plus_u(L, kU_ev);
        diis.use_ot = false;
        diis.k_points.push_back({0.25, 0.0, 0.0});
        diis.k_point_weights.push_back(1.0);
        diis.build(atoms);
        const GpwResult rd = diis.run(atoms);
        check(rd.converged, "A2 multi-k DIIS+U converged");

        GpwDft ot = base_plus_u(L, kU_ev);
        ot.use_ot = true;
        ot.k_points.push_back({0.25, 0.0, 0.0});
        ot.k_point_weights.push_back(1.0);
        ot.max_iter = 200;
        ot.build(atoms);
        const GpwResult ro = ot.run(atoms);
        check(ro.converged, "A2 multi-k OT+U converged");
        check(ro.n_kpts == 1, "A2 n_kpts == 1");
        const double d = std::fabs(ro.e_total - rd.e_total);
        std::printf("A2: E_OT+U = %.12f  E_DIIS+U = %.12f  |Δ|=%.3e Ha  "
                    "e_dftu=%.8f  iters ot/diis=%d/%d\n",
                    ro.e_total, rd.e_total, d, ro.e_dftu, ro.iters, rd.iters);
        check(d <= 1e-4, "A2 |E_OT+U - E_DIIS+U| <= 1e-4 Ha");
        check(std::fabs(ro.e_dftu - rd.e_dftu) <= 1e-3,
              "A2 |e_dftu OT − DIIS| <= 1e-3 Ha");
        check(std::fabs(ro.e_dftu) > 1e-8, "A2 e_dftu non-trivial");
    }

    // ── A3: MP 2×2×2 activation + absolute pin ─────────────────────────────
    // First-green ATV absolute (REFERENCE.md, 2026-07-26).
    // re-pinned 2026-07-31: variational +U scatter (pre-fix -1.077477969428)
    constexpr double kE_pin = -1.396485808593;
    constexpr double kTolE = 1e-3; // Ha
    constexpr double kFloorAct = 1e-4;

    {
        auto make = [&](bool with_u) {
            GpwDft m;
            m.cell[0] = m.cell[1] = m.cell[2] = L;
            m.cutoff_ry = 100.0;
            m.n_occ = 1;
            m.max_iter = 200;
            m.tol = 1e-7;
            m.mix = 0.35;
            m.xc = XcKind::PADE;
            m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
            m.k_time_reversal = true;
            m.use_ot = true;
            if (with_u)
                m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, kU_ev});
            m.build(atoms);
            return m.run(atoms);
        };

        const GpwResult r0 = make(false);
        const GpwResult ru = make(true);

        check(r0.converged && ru.converged, "A3 both OT converge");
        check(ru.n_kpts == 4, "A3 TR-folded n_kpts == 4");
        check(std::isfinite(ru.e_total) && std::isfinite(ru.e_dftu),
              "A3 energies finite");
        check(std::fabs(ru.e_dftu) > 1e-8, "A3 e_dftu non-trivial");
        const double dU = std::fabs(ru.e_total - r0.e_total);
        check(dU >= kFloorAct, "A3 |E(U)−E(0)| activation floor");
        check(std::fabs(ru.e_total - kE_pin) <= kTolE,
              "A3 |E − pin| within tol");

        std::printf("  E(U=0, OT) = %.12f Ha  iters=%d\n", r0.e_total,
                    r0.iters);
        std::printf("  E(U=4, OT) = %.12f Ha  iters=%d  e_dftu=%.8f\n",
                    ru.e_total, ru.iters, ru.e_dftu);
        std::printf("  |E(U)−E(0)| = %.6e Ha  pin=%.12f tol=%.1e\n", dU, kE_pin,
                    kTolE);
    }

    const auto t1 = clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    std::printf("wall = %.3f s  fails=%d\n", wall, g_fail);
    if (g_fail != 0) {
        std::fprintf(stderr, "reg_ot-dftu-kpoints: %d check(s) failed\n",
                     g_fail);
        return 1;
    }
    std::printf("reg_ot-dftu-kpoints: PASS\n");
    return 0;
}
