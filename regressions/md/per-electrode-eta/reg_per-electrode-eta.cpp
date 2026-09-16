/// @file regressions/md/per-electrode-eta/reg_per-electrode-eta.cpp
/// @brief E2E: per-electrode Gaussian width η_e on the constant-potential BO
///        solve (MetalWalls multi-electrode V,η; literature pin).
///
/// Offline pin: closed-form Gaussian elastance (Reed–Lanning–Madden 2007;
/// Scalfi et al. 2020; MetalWalls JOSS 2020) — see REFERENCE.md.
/// Public headers + link atomiverse only. Exits non-zero on failure.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

namespace {

/// Two 2x2-atom square walls (a = 2 Å) at z = 8 and 13 in a 4x4x24 Å box.
cpu::Frame two_walls() {
    cpu::Frame f(8);
    int i = 0;
    for (int wall = 0; wall < 2; ++wall) {
        const double z = (wall == 0) ? 8.0 : 13.0;
        for (int ax = 0; ax < 2; ++ax) {
            for (int ay = 0; ay < 2; ++ay) {
                f.span(kAtom_x)[i] = 2.0 * ax;
                f.span(kAtom_y)[i] = 2.0 * ay;
                f.span(kAtom_z)[i] = z;
                f.span(kAtom_charge)[i] = 0.0;
                f.span(kAtom_type)[i] = wall + 1;
                f.span(kAtom_mass)[i] = 196.97;
                f.span(kAtom_tag)[i] = i + 1;
                f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] =
                    f.span(kAtom_vz)[i] = 0.0;
                ++i;
            }
        }
    }
    f.box = cpu::Box::orthorhombic(4.0, 4.0, 24.0);
    return f;
}

void solve_bo(cpu::md::ConstPotBO& conp, cpu::Frame& f) {
    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
    sim.set_evaluator(cpu::PairCoulCut{12.0});
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_observer_ref(conp);
    sim.build();
    sim.run(f, 1);
}

double solve_two_walls(double eta_bot, double eta_top, bool three_arg) {
    auto f = two_walls();
    cpu::md::ConstPotBO conp(eta_bot);
    if (three_arg) {
        conp.add_electrode(1, 0.0, eta_bot);
        conp.add_electrode(2, 1.0, eta_top);
    } else {
        // shared-η path (eta_top ignored; must equal eta_bot by contract)
        conp.add_electrode(1, 0.0);
        conp.add_electrode(2, 1.0);
    }
    conp.electroneutrality = true;
    conp.xy_image_range = 100;
    solve_bo(conp, f);
    if (conp.neutrality_residual() > 1e-10) {
        std::fprintf(stderr, "FAIL two-wall neutrality %g\n",
                     conp.neutrality_residual());
        std::exit(1);
    }
    return conp.total_charge(1);
}

} // namespace

int main() {
    int nfail = 0;

    // ── Case 1: two-site vacuum capacitor, dissimilar η — literature pin ──
    // Q = ΔΨ/(A11+A22−2A12) = 0.037722100829787 e (REFERENCE.md).
    constexpr double kQPin = 0.037722100829787; // [e], oracle literature
    {
        cpu::Frame f(2);
        const double z[2] = {18.0, 22.0};
        for (int i = 0; i < 2; ++i) {
            f.span(kAtom_x)[i] = 0.0;
            f.span(kAtom_y)[i] = 0.0;
            f.span(kAtom_z)[i] = z[i];
            f.span(kAtom_charge)[i] = 0.0;
            f.span(kAtom_type)[i] = i + 1;
            f.span(kAtom_mass)[i] = 196.97;
            f.span(kAtom_tag)[i] = i + 1;
            f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] =
                0.0;
        }
        f.box = cpu::Box::cubic(40.0);

        cpu::md::ConstPotBO conp(/*shared eta=*/2.0);
        conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0, /*eta [1/Å]=*/1.979);
        conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0, /*eta [1/Å]=*/0.955);
        conp.electroneutrality = true;
        solve_bo(conp, f);

        const double q_top = conp.total_charge(1);
        const double rel = std::abs(q_top - kQPin) / kQPin;
        std::printf("per-electrode-eta case1 (two-site pin)\n");
        std::printf("  Q_top   = % .15e e  (pin %+.15e)\n", q_top, kQPin);
        std::printf("  rel err = % .3e  |sumQ| = % .3e\n", rel,
                    conp.neutrality_residual());
        if (!(rel <= 1e-9)) {
            std::fprintf(stderr, "FAIL case1 rel %g > 1e-9\n", rel);
            ++nfail;
        }
        if (!(conp.neutrality_residual() <= 1e-12)) {
            std::fprintf(stderr, "FAIL case1 neutrality %g\n",
                         conp.neutrality_residual());
            ++nfail;
        }
    }

    // ── Case 2: multi-atom two-wall, mixed η, 2D xy Gaussian images ──────
    const double q_mixed = solve_two_walls(1.979, 0.955, /*three_arg=*/true);
    const double q_hi = solve_two_walls(1.979, 1.979, /*three_arg=*/true);
    const double q_lo = solve_two_walls(0.955, 0.955, /*three_arg=*/true);
    const double q_hi_shared =
        solve_two_walls(1.979, 1.979, /*three_arg=*/false);

    std::printf("per-electrode-eta case2 (2x2 walls, R=100 images)\n");
    std::printf("  Q_top mixed        = % .15e e\n", q_mixed);
    std::printf("  Q_top shared 1.979 = % .15e e\n", q_hi);
    std::printf("  Q_top shared 0.955 = % .15e e\n", q_lo);

    if (!(q_mixed > 0.0)) {
        std::fprintf(stderr, "FAIL case2 sign Q_top=%g\n", q_mixed);
        ++nfail;
    }
    // Monotone bracket: smaller η lowers the self elastance → more charge.
    if (!(q_hi < q_mixed && q_mixed < q_lo)) {
        std::fprintf(stderr, "FAIL case2 bracket %g < %g < %g violated\n",
                     q_hi, q_mixed, q_lo);
        ++nfail;
    }
    // 3-arg overload at equal η reproduces the 2-arg shared path exactly.
    if (!(std::abs(q_hi - q_hi_shared) <= 1e-14)) {
        std::fprintf(stderr, "FAIL case2 shared-eta twin mismatch %g vs %g\n",
                     q_hi, q_hi_shared);
        ++nfail;
    }

    if (nfail) {
        std::fprintf(stderr, "%d check(s) failed\n", nfail);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
