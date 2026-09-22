/// @file regressions/md/voronoi-volume-tf/reg_voronoi-volume-tf.cpp
/// @brief E2E: Voronoi site-volume companion for Thomas–Fermi metallicity.
///
/// Offline pin: MetalWalls TF model with per-species `voronoi_volume`
/// (Scalfi et al. 2020) — see REFERENCE.md. Public headers + link
/// atomiverse only. Exits non-zero on failure.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/voronoi_volume.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

namespace {

// Real MetalWalls c8ad175d pins (mpirun -n 1 mw, 2026-07-29):
// tools/metalwalls_oracle/inputs/metalwalls/voronoi-volume-tf/mw_twin/
// with @LTF@ / @VORONOI@ substituted — see total_charges.out archives.
// Supersedes the pin_voronoi_tf.py self-golden (same 6.13% defect as the
// pre-repin thomas-fermi-electrode numbers).
constexpr double kVfccPin = 16.854785750000; // [Å³] a0³/4, exact geometry
constexpr double kQTopBulkV = 3.707555065364343e-03; // [e] l_tf=2 Å, v = a0³/4
constexpr double kQTopHalfV = 3.321360199637453e-03; // [e] l_tf=2 Å, v = a0³/8
constexpr double kQTopPerfect = 4.354124119406114e-03; // [e] l_tf=0
// ATV truncated xy-image elastance vs MW 2D-Ewald: ~1e-5 residual at 1280.
constexpr double kRelTol = 0.01; // 1% dual-gate vs real MW pins

constexpr double kEta = 1.0 / 0.56;            // [1/Å] MetalWalls w = 0.56 Å
constexpr double kA0 = 4.07;                   // [Å] Au lattice constant
constexpr double kD = 4.0 / (kA0 * kA0 * kA0); // [1/Å³] fixed-density twin
constexpr double kPlane = kA0 / 2.0;
constexpr double kLGap = 20.0;
constexpr int kImageRange = 1280;

int g_nfail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_nfail;
    }
}

/// Bulk FCC gold, 2x2x2 conventional cells (32 sites), fully periodic.
cpu::Frame bulk_gold() {
    const double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
    cpu::Frame f(32);
    int i = 0;
    for (int cx = 0; cx < 2; ++cx)
        for (int cy = 0; cy < 2; ++cy)
            for (int cz = 0; cz < 2; ++cz)
                for (int b = 0; b < 4; ++b) {
                    f.span(kAtom_x)[i] = (cx + basis[b][0]) * kA0;
                    f.span(kAtom_y)[i] = (cy + basis[b][1]) * kA0;
                    f.span(kAtom_z)[i] = (cz + basis[b][2]) * kA0;
                    f.span(kAtom_charge)[i] = 0.0;
                    f.span(kAtom_type)[i] = 1;
                    f.span(kAtom_mass)[i] = 196.966553;
                    f.span(kAtom_tag)[i] = i + 1;
                    ++i;
                }
    f.box = cpu::Box::cubic(2.0 * kA0);
    return f;
}

/// Gold capacitor toy: 3+3 electrode planes (types 1/2), empty vacuum —
/// identical geometry to the wave-2 `thomas-fermi-electrode` regression.
cpu::Frame gold_capacitor() {
    cpu::Frame f(6);
    const double z_surf_bot = 5.0 + 2.0 * kPlane;
    const double z_surf_top = z_surf_bot + kLGap;
    const double z[6] = {
        z_surf_bot - 2.0 * kPlane,
        z_surf_bot - kPlane,
        z_surf_bot,
        z_surf_top,
        z_surf_top + kPlane,
        z_surf_top + 2.0 * kPlane,
    };
    const int typ[6] = {1, 1, 1, 2, 2, 2};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = 196.966553;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0.0;
    }
    f.box = cpu::Box::orthorhombic(kA0, kA0, 80.0);
    return f;
}

double run_tf(double l_tf, const std::vector<double>& site_volume,
              double site_density, double* c_out) {
    auto f = gold_capacitor();
    cpu::md::ConstPotBO conp(kEta);
    conp.add_electrode(/*type=*/1, /*Psi=*/0.0);
    conp.add_electrode(/*type=*/2, /*Psi=*/1.0);
    conp.electroneutrality = true;
    conp.xy_image_range = kImageRange;
    conp.l_tf = l_tf;
    conp.site_volume = site_volume;
    conp.site_density = site_density;
    {
        cpu::md::MD sim;
        sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
        sim.set_evaluator(cpu::PairCoulCut{12.0});
        sim.set_integrator(cpu::md::VelocityVerlet{0.0});
        sim.set_observer_ref(conp);
        sim.build();
        sim.run(f, 1);
    }
    check(conp.neutrality_residual() <= 1e-10, "electroneutrality <= 1e-10 e");
    if (c_out)
        *c_out = conp.capacitance();
    return conp.total_charge(1);
}

} // namespace

int main() {
    std::printf("voronoi-volume-tf: Voronoi site-volume TF companion "
                "(MetalWalls E7, Scalfi et al. 2020)\n");

    // ── Part 1: v_i from geometry on bulk FCC gold ─────────────────────
    auto bulk = bulk_gold();
    const auto v = cpu::VoronoiVolume{}.volumes(bulk);
    double vsum = 0.0, vmin = 1e300, vmax = 0.0;
    for (double vi : v) {
        vsum += vi;
        vmin = std::min(vmin, vi);
        vmax = std::max(vmax, vi);
    }
    const double vbox = std::pow(2.0 * kA0, 3);
    std::printf("  FCC v_i     = [%.12f, %.12f] Å³  (pin %.12f)\n", vmin, vmax,
                kVfccPin);
    std::printf("  sum v_i     = %.12f Å³  (box %.12f)\n", vsum, vbox);
    check(std::abs(vmin - kVfccPin) <= kVfccPin * 1e-9,
          "min FCC Voronoi volume == a0^3/4 (1e-9 rel)");
    check(std::abs(vmax - kVfccPin) <= kVfccPin * 1e-9,
          "max FCC Voronoi volume == a0^3/4 (1e-9 rel)");
    check(std::abs(vsum - vbox) <= vbox * 1e-10,
          "sum v_i == box volume (1e-10 rel)");

    // ── Part 2: Q with computed v_i vs MetalWalls-model pins ───────────
    const double v_site = v[0]; // bulk FCC volume feeds the capacitor sites
    double c_bulk = 0.0, c_half = 0.0, c_dens = 0.0;

    const double q_voro =
        run_tf(2.0, std::vector<double>(6, v_site), 0.0, &c_bulk);
    const double q_dens = run_tf(2.0, {}, kD, &c_dens);
    std::printf("  Q_top(v=a0³/4) = %+.12e e (pin %+.12e)\n", q_voro,
                kQTopBulkV);
    std::printf("  Q_top(d=4/a0³) = %+.12e e (fixed-density twin)\n", q_dens);
    check(std::abs(q_voro - kQTopBulkV) <= std::abs(kQTopBulkV) * kRelTol,
          "Q_top(voronoi bulk) vs MW-model pin (1%)");
    check(std::abs(q_voro - q_dens) <= std::abs(q_dens) * 1e-10,
          "Q_top(voronoi bulk) == Q_top(fixed density) (1e-10 rel)");
    check(std::abs(c_bulk - std::abs(kQTopBulkV))
              <= std::abs(kQTopBulkV) * kRelTol,
          "C(voronoi bulk) vs pin (1%)");

    // ── Part 3: metallicity gate — halved volume hardens the diagonal ──
    const double q_half =
        run_tf(2.0, std::vector<double>(6, 0.5 * v_site), 0.0, &c_half);
    std::printf("  Q_top(v/2)     = %+.12e e (pin %+.12e)\n", q_half,
                kQTopHalfV);
    check(std::abs(q_half - kQTopHalfV) <= std::abs(kQTopHalfV) * kRelTol,
          "Q_top(v/2) vs MW-model pin (1%)");
    check(c_half < c_bulk, "C(v/2) < C(v) metallicity ordering");

    // Perfect metal is untouched by volumes (l_tf = 0 ignores them).
    const double q_perfect =
        run_tf(0.0, std::vector<double>(6, v_site), 0.0, nullptr);
    std::printf("  Q_top(l_tf=0)  = %+.12e e (pin %+.12e)\n", q_perfect,
                kQTopPerfect);
    check(std::abs(q_perfect - kQTopPerfect)
              <= std::abs(kQTopPerfect) * kRelTol,
          "Q_top(perfect metal) vs wave-2 pin (1%)");
    check(c_bulk < std::abs(q_perfect),
          "C(TF bulk v) < C(perfect metal) metallicity ordering");

    if (g_nfail) {
        std::fprintf(stderr, "%d check(s) failed\n", g_nfail);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
