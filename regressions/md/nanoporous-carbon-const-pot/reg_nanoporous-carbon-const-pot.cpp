/// @file regressions/md/nanoporous-carbon-const-pot/reg_nanoporous-carbon-const-pot.cpp
/// @brief E2E product: slit-pore carbon electrodes at fixed Ψ + electrolyte.
///
/// Composition gate (map §9 nanoporous): honeycomb carbon slit walls held by
/// ConstPotBO, frozen, with mobile ions inside the pore. Checks polarity,
/// neutrality, freeze, occupancy (ions stay in the pore half-space), and a
/// soft |Q| pin against MetalWalls c8ad175d empty-slit twin.
/// Public headers only. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/fix_freeze.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

// Slit pore: two honeycomb walls (112 C each) at z=8 / 32 Å in Lz=40 Å
// → pore gap 24 Å, exterior return 16 Å (pore path dominates).
constexpr double kAcc = 1.42;
constexpr int kNx = 4, kNy = 7;
constexpr double kLz = 40.0;
constexpr double kZBot = 8.0;
constexpr double kZTop = 32.0;
constexpr double kPsiBot = -1.0; // [V]
constexpr double kPsiTop = +1.0;
constexpr double kEta = 1.805132; // MW graphite [1/Å]
constexpr int kImageRange = 40;
constexpr int kNWallPer = kNx * kNy * 4; // 112
constexpr int kNIons = 8;

// MetalWalls c8ad175d empty-slit twin (no ions): |Q| on one plate.
// tools/metalwalls_oracle/inputs/metalwalls/nanoporous-carbon-const-pot/
// Soft pin — image-sum vs Ewald residual.
constexpr double kMwEmptyAbsQ = 0.0; // filled after MW run; 0 → skip soft pin
constexpr double kMwQRelTol = 0.05;

cpu::Frame build_slit(bool with_ions) {
    const double cx = 3.0 * kAcc;
    const double cy = std::sqrt(3.0) * kAcc;
    const double Lx = kNx * cx;
    const double Ly = kNy * cy;
    const double bx[4] = {0.0, kAcc, 1.5 * kAcc, 2.5 * kAcc};
    const double by[4] = {0.0, 0.0, 0.5 * cy, 0.5 * cy};

    const int N = 2 * kNWallPer + (with_ions ? kNIons : 0);
    cpu::Frame f(N);
    int idx = 0;
    for (int wall = 0; wall < 2; ++wall) {
        const double z = (wall == 0) ? kZBot : kZTop;
        const int typ = (wall == 0) ? 2 : 3;
        for (int ix = 0; ix < kNx; ++ix)
            for (int iy = 0; iy < kNy; ++iy)
                for (int b = 0; b < 4; ++b) {
                    f.span(kAtom_x)[idx] = ix * cx + bx[b];
                    f.span(kAtom_y)[idx] = iy * cy + by[b];
                    f.span(kAtom_z)[idx] = z;
                    f.span(kAtom_type)[idx] = typ;
                    f.span(kAtom_charge)[idx] = 0.0;
                    f.span(kAtom_mass)[idx] = 12.011;
                    f.span(kAtom_tag)[idx] = idx + 1;
                    ++idx;
                }
    }
    if (with_ions) {
        // 4 Na + 4 Cl staggered in the pore mid-plane band.
        for (int i = 0; i < kNIons; ++i) {
            const bool na = (i % 2 == 0);
            f.span(kAtom_x)[idx] = (0.2 + 0.2 * (i % 4)) * Lx;
            f.span(kAtom_y)[idx] = (0.25 + 0.25 * (i / 4)) * Ly;
            f.span(kAtom_z)[idx] =
                0.5 * (kZBot + kZTop) + ((i < 4) ? -2.0 : 2.0);
            f.span(kAtom_type)[idx] = na ? 0 : 1;
            f.span(kAtom_charge)[idx] = na ? 1.0 : -1.0;
            f.span(kAtom_mass)[idx] = na ? 22.9898 : 35.453;
            f.span(kAtom_tag)[idx] = idx + 1;
            ++idx;
        }
    }
    f.box = cpu::Box::orthorhombic(Lx, Ly, kLz);
    return f;
}

double electrode_Q(const cpu::Frame& f, int typ) {
    double q = 0.0;
    for (int i = 0; i < f.nlocal(); ++i)
        if (f.span(kAtom_type)[i] == typ)
            q += f.span(kAtom_charge)[i];
    return q;
}

int occupancy_in_pore(const cpu::Frame& f) {
    int n = 0;
    for (int i = 0; i < f.nlocal(); ++i) {
        const int t = f.span(kAtom_type)[i];
        if (t != 0 && t != 1)
            continue;
        const double z = f.span(kAtom_z)[i];
        if (z > kZBot + 0.5 && z < kZTop - 0.5)
            ++n;
    }
    return n;
}

} // namespace

int main() {
    // ── Empty slit: BO pin vs continuum polarity + optional MW soft pin ─
    {
        auto f = build_slit(/*with_ions=*/false);
        cpu::md::ConstPotBO bo(kEta);
        bo.add_electrode(2, kPsiBot);
        bo.add_electrode(3, kPsiTop);
        bo.electroneutrality = true;
        bo.xy_image_range = kImageRange;
        {
            cpu::md::MD sim;
            sim.set_neighbor(cpu::md::NeighborList<>{10.0, 0.5});
            sim.set_evaluator(cpu::PairCoulCut{10.0});
            sim.set_integrator(cpu::md::VelocityVerlet{0.0});
            sim.set_observer_ref(bo);
            sim.build();
            sim.run(f, 1);
        }
        const double QL = bo.total_charge(0);
        const double QR = bo.total_charge(1);
        std::printf("empty slit: QL=%+.6e QR=%+.6e neut=%.3e\n", QL, QR,
                    bo.neutrality_residual());
        if (!(QL < 0.0 && QR > 0.0)) {
            std::printf("FAIL empty polarity\n");
            return 1;
        }
        if (!(std::fabs(QL + QR) <= 1e-8)) {
            std::printf("FAIL empty neutrality\n");
            return 1;
        }
        if (kMwEmptyAbsQ > 0.0) {
            if (!(std::fabs(std::fabs(QL) - kMwEmptyAbsQ)
                  <= kMwEmptyAbsQ * kMwQRelTol)) {
                std::printf("FAIL empty |Q| vs MW %.6e (got %.6e)\n",
                            kMwEmptyAbsQ, std::fabs(QL));
                return 1;
            }
        }
    }

    // ── Filled pore: short NVT product ─────────────────────────────────
    auto frame = build_slit(/*with_ions=*/true);
    std::vector<double> wall_z0;
    std::vector<int> wall_idx;
    for (int i = 0; i < frame.nlocal(); ++i) {
        const int t = frame.span(kAtom_type)[i];
        if (t == 2 || t == 3) {
            wall_idx.push_back(i);
            wall_z0.push_back(frame.span(kAtom_z)[i]);
        }
    }

    cpu::md::ConstPotBO cpot(kEta);
    cpot.add_electrode(2, kPsiBot);
    cpot.add_electrode(3, kPsiTop);
    cpot.electroneutrality = true;
    cpot.xy_image_range = kImageRange;
    cpot.use_cg = true;
    cpot.cg_tol = 1e-10;
    cpot.predictor_order = 2;

    cpu::md::MD sim;
    sim.set_threads(4);
    sim.set_observer(cpu::md::InitVelocity{/*T=*/300.0, /*seed=*/17u});
    sim.set_integrator(cpu::md::VelocityVerlet{/*dt=*/1.0});
    sim.set_thermostat(cpu::md::NoseHoover{300.0, /*tau=*/100.0, 1.0});
    // Soft LJ on ions only (no C–C diagonal).
    sim.set_evaluator(cpu::PairLJCut{{{0, 0, 0.01, 2.5},
                                      {1, 1, 0.01, 2.5},
                                      {0, 1, 0.01, 2.5},
                                      {0, 2, 0.005, 3.0},
                                      {0, 3, 0.005, 3.0},
                                      {1, 2, 0.005, 3.0},
                                      {1, 3, 0.005, 3.0}},
                                     /*cutoff=*/8.0});
    sim.set_neighbor(cpu::md::NeighborList<>{8.0, 0.5});
    sim.on(md::Phase::force).add(cpu::CoulLong{/*alpha=*/0.3, /*cutoff=*/8.0});
    sim.set_evaluator(cpu::KSpacePME{0.3, 0.5, 5});
    sim.set_observer(std::move(cpot));
    sim.set_observer(cpu::md::FixFreeze{2});
    sim.set_observer(cpu::md::FixFreeze{3});
    sim.enable_timing(false);
    sim.build();

    sim.run(frame, 1);
    double QL = electrode_Q(frame, 2);
    double QR = electrode_Q(frame, 3);
    std::printf("filled step1: QL=%+.6e QR=%+.6e occ=%d/%d\n", QL, QR,
                occupancy_in_pore(frame), kNIons);
    if (!(QL < 0.0 && QR > 0.0)) {
        std::printf("FAIL filled polarity\n");
        return 1;
    }
    if (!(std::fabs(QL + QR) <= 1e-6)) {
        std::printf("FAIL filled neutrality\n");
        return 1;
    }
    // Screening: filled |Q| > empty |Q| is expected but geometry differs;
    // require occupancy full at start.
    if (occupancy_in_pore(frame) != kNIons) {
        std::printf("FAIL initial occupancy\n");
        return 1;
    }

    for (int c = 0; c < 4; ++c) {
        sim.run(frame, 50);
        if (!std::isfinite(frame.meta.get(kEnergy))
            || !std::isfinite(frame.meta.get(kKinetic))) {
            std::printf("FAIL non-finite thermo chunk %d\n", c);
            return 1;
        }
        QL = electrode_Q(frame, 2);
        QR = electrode_Q(frame, 3);
        if (!(std::fabs(QL + QR) <= 1e-5)) {
            std::printf("FAIL neutrality drift chunk %d sum=%.3e\n", c,
                        QL + QR);
            return 1;
        }
    }

    // Freeze + occupancy proxy after short NVT.
    double max_dz = 0.0;
    for (std::size_t k = 0; k < wall_idx.size(); ++k)
        max_dz = std::fmax(
            max_dz, std::fabs(frame.span(kAtom_z)[wall_idx[k]] - wall_z0[k]));
    const int occ = occupancy_in_pore(frame);
    std::printf(
        "after 200 fs: E=%.6e K=%.6e QL=%+.6e QR=%+.6e occ=%d max|dz|=%.3e\n",
        frame.meta.get(kEnergy), frame.meta.get(kKinetic), QL, QR, occ, max_dz);
    if (!(max_dz <= 1e-10)) {
        std::printf("FAIL freeze\n");
        return 1;
    }
    if (occ < kNIons / 2) {
        std::printf("FAIL occupancy collapsed to %d\n", occ);
        return 1;
    }
    if (!(QL < 0.0 && QR > 0.0)) {
        std::printf("FAIL end polarity\n");
        return 1;
    }
    std::printf("PASS: nanoporous-carbon-const-pot\n");
    return 0;
}
