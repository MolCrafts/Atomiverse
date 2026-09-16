/// @file regressions/gpw/gpw-dftu-kpoints/reg_gpw-dftu-kpoints.cpp
/// @brief END-TO-END: multi-k collinear UKS GPW + Dudarev DFT+U.
///
/// Public `GpwDft` only. Offline CP2K twin under
/// `tools/cp2k_oracle/inputs/gpw-dftu-kpoints/` (REFERENCE.md). ctest never
/// spawns CP2K.
///
/// Gates:
///   A1 Γ-limit: explicit k=Γ + PLUS_U ≡ real Γ-UKS + PLUS_U
///   A2 multi-k activation: |E(U) − E(U=0)| floor on MP 2×2×2
///   A3 multi-k converges; e_dftu finite / non-trivial; absolute pin
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

    // ── A1: Γ-limit identity with PLUS_U ──────────────────────────────────
    {
        auto base = [&]() {
            GpwDft cfg;
            cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = L;
            cfg.cutoff_ry = 100.0;
            cfg.n_alpha = 2;
            cfg.n_beta = 0;
            // Variational ½(V̂S+SV̂) +U scatter (2026-07-31) needs a larger
            // DIIS budget on this strong-overlap toy (~90-150 iters).
            cfg.max_iter = 400;
            cfg.tol = 1e-7;
            cfg.mix = 0.35;
            cfg.xc = XcKind::PADE;
            cfg.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, kU_ev});
            return cfg;
        };

        GpwDft g = base();
        g.build(atoms);
        const GpwResult rg = g.run(atoms);

        GpwDft k = base();
        k.k_points = {{0.0, 0.0, 0.0}};
        k.k_point_weights = {1.0};
        k.build(atoms);
        const GpwResult rk = k.run(atoms);

        check(rg.converged && rk.converged, "A1 both converge");
        check(std::fabs(rk.e_total - rg.e_total) <= 1e-6,
              "A1 |E_kΓ+U − E_Γ+U| ≤ 1e-6 Ha");
        // e_dftu decomposition can differ slightly (grid/Hermitian projection);
        // total energy is the decisive gate.
        check(std::fabs(rk.e_dftu - rg.e_dftu) <= 1e-4,
              "A1 |e_dftu kΓ − Γ| ≤ 1e-4 Ha");
        check(std::fabs(rg.e_dftu) > 1e-8, "A1 e_dftu non-trivial");
        std::printf("  A1 E_Γ=%.10f  E_kΓ=%.10f  dE=%.3e  e_dftu_Γ=%.6e "
                    "e_dftu_k=%.6e\n",
                    rg.e_total, rk.e_total, rk.e_total - rg.e_total, rg.e_dftu,
                    rk.e_dftu);
    }

    // ── A2/A3: MP 2×2×2 high-spin H₂ UKS ± PLUS_U ────────────────────────
    // Frozen ATV absolute (REFERENCE.md). Re-pinned 2026-07-31 for the
    // variational ½(V̂S+SV̂) +U scatter (the bare-V̂ fixed point was
    // non-stationary for E_U[n(P,S)]; pre-fix pin was −0.814170782662).
    constexpr double kE_pin = -0.997491857710;
    constexpr double kTolE = 1e-3; // Ha
    constexpr double kFloorAct = 1e-4;

    {
        auto make = [&](bool with_u) {
            GpwDft m;
            m.cell[0] = m.cell[1] = m.cell[2] = L;
            m.cutoff_ry = 120.0;
            m.n_alpha = 2;
            m.n_beta = 0;
            m.max_iter = 400; // variational +U scatter: larger DIIS budget
            m.tol = 1e-7;
            m.mix = 0.35;
            m.xc = XcKind::PADE;
            m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
            m.k_time_reversal = true;
            if (with_u)
                m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, kU_ev});
            m.build(atoms);
            return m.run(atoms);
        };

        const GpwResult r0 = make(false);
        const GpwResult ru = make(true);

        check(r0.converged && ru.converged, "A2/A3 both converge");
        check(ru.n_kpts == 4, "A3 TR-folded n_kpts == 4");
        check(std::isfinite(ru.e_total) && std::isfinite(ru.e_dftu),
              "A3 energies finite");
        check(std::fabs(ru.e_dftu) > 1e-8, "A3 e_dftu non-trivial");
        const double dU = std::fabs(ru.e_total - r0.e_total);
        check(dU >= kFloorAct, "A2 |E(U)−E(0)| activation floor");
        check(std::fabs(ru.e_total - kE_pin) <= kTolE,
              "A3 |E − pin| within tol");

        std::printf("  E(U=0) = %.12f Ha  iters=%d\n", r0.e_total, r0.iters);
        std::printf("  E(U=4) = %.12f Ha  iters=%d  e_dftu=%.8f\n", ru.e_total,
                    ru.iters, ru.e_dftu);
        std::printf("  |E(U)−E(0)| = %.6e Ha  pin=%.12f tol=%.1e\n", dU, kE_pin,
                    kTolE);
    }

    const auto t1 = clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    std::printf("wall = %.3f s\n", wall);
    if (g_fail) {
        std::fprintf(stderr, "FAIL gpw-dftu-kpoints (%d checks)\n", g_fail);
        return 1;
    }
    std::printf("PASS gpw-dftu-kpoints\n");
    return 0;
}
