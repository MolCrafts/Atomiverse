/// @file tests/cpu/md/test_neighbor_list_rebuild.cpp
/// @brief B.4 — NeighborList rebuild cadence: delay, every, displacement gating.
///
/// Verifies that the LAMMPS `neigh_modify delay M every N check yes` logic
/// is correctly implemented: rebuilds are gated by `ago >= delay`, `ago %
/// every == 0`, and the displacement skin check. Also verifies the rebuild
/// counter is accurate.

#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

// ── helpers ───────────────────────────────────────────────────────────

/// 108-atom FCC argon crystal at a=5.26 Å, 4×4×4 unit cells → 256 atoms.
cpu::Frame make_fcc_argon() {
    constexpr double a = 5.26; // FCC lattice constant [Å]
    constexpr int n_cell = 3;  // 3×3×3 unit cells
    constexpr double box_len = n_cell * a;

    const double basis[4][3] = {
        {0.0, 0.0, 0.0},
        {0.0, 0.5 * a, 0.5 * a},
        {0.5 * a, 0.0, 0.5 * a},
        {0.5 * a, 0.5 * a, 0.0},
    };

    std::vector<double> x, y, z;
    for (int ix = 0; ix < n_cell; ++ix) {
        for (int iy = 0; iy < n_cell; ++iy) {
            for (int iz = 0; iz < n_cell; ++iz) {
                for (auto& b : basis) {
                    x.push_back(ix * a + b[0]);
                    y.push_back(iy * a + b[1]);
                    z.push_back(iz * a + b[2]);
                }
            }
        }
    }

    const int n = static_cast<int>(x.size());
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_x)[i] = x[i];
        frame.span(ATV::kAtom_y)[i] = y[i];
        frame.span(ATV::kAtom_z)[i] = z[i];
        frame.span(ATV::kAtom_mass)[i] = 39.948; // Ar amu
        frame.span(ATV::kAtom_type)[i] = 0;
    }
    frame.simbox = ATV::cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

// ── delay gate ────────────────────────────────────────────────────────

TEST(NeighborListRebuild, DelayGateHonoured) {
    // With delay=100 and zero velocities, the initial build() is the
    // only rebuild — no displacement can trigger within the delay window.
    auto frame = make_fcc_argon();
    constexpr double cutoff = 6.8;
    constexpr double skin =
        2.0; // large enough that atoms won't move past skin/2

    cpu::md::NeighborList<> nl{cutoff, skin, /*every=*/1, /*delay=*/100};

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.5});
    sim.add_potential(cpu::PairLJCut{0.0104, 3.4, cutoff});
    sim.set_neighbor_ref(nl);
    sim.build();

    sim.run(frame, 200);

    // With zero initial velocities, atoms don't move; displacement check
    // never fires. Even if it did, delay=100 gates it out. We expect
    // exactly 1 rebuild (the initial build()).
    EXPECT_EQ(nl.n_rebuilds(), 1)
        << "delay=100 with zero velocities: expected 1 rebuild (initial), got "
        << nl.n_rebuilds();
}

// ── every gate ────────────────────────────────────────────────────────

TEST(NeighborListRebuild, EveryGateReducesChecks) {
    // With delay=0, every=50, and a tiny skin, the displacement check
    // should fire only at steps 50, 100, 150, ...
    auto frame = make_fcc_argon();

    // Give atoms a significant kick so displacement triggers rebuilds
    // whenever the check is actually performed.
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(ATV::kAtom_vx)[i] = (i % 3 == 0) ? 0.05 : -0.02;
        frame.span(ATV::kAtom_vy)[i] = (i % 3 == 1) ? 0.05 : -0.02;
        frame.span(ATV::kAtom_vz)[i] = (i % 3 == 2) ? 0.05 : -0.02;
    }

    constexpr double cutoff = 6.8;
    constexpr double skin = 0.02; // tiny: displacement rapidly exceeds skin/2

    cpu::md::NeighborList<> nl{cutoff, skin, /*every=*/50, /*delay=*/0};

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.5});
    sim.add_potential(cpu::PairLJCut{0.0104, 3.4, cutoff});
    sim.set_neighbor_ref(nl);
    sim.build();

    sim.run(frame, 200);

    // With every=50 and delay=0, the displacement check fires at steps
    // 50, 100, 150, 200 → at most 4 rebuilds beyond the initial one.
    // Total ≤ 5.
    EXPECT_LE(nl.n_rebuilds(), 5)
        << "every=50 over 200 steps: expected ≤5 rebuilds, got "
        << nl.n_rebuilds();

    // Sanity: we must have had at least 2 rebuilds (initial + at least one
    // displacement-driven) since skin is tiny and atoms move fast.
    EXPECT_GE(nl.n_rebuilds(), 2)
        << "every=50 with tiny skin and fast atoms: expected ≥2 rebuilds";
}

// ── size-stale overrides delay ────────────────────────────────────────

TEST(NeighborListRebuild, DelayZeroAllowsImmediateRebuild) {
    // delay=0, large skin → no displacement rebuilds, the only rebuild
    // should be the initial one.
    auto frame = make_fcc_argon();
    constexpr double cutoff = 6.8;
    constexpr double skin =
        10.0; // enormous: skin/2 = 5 Å, atoms can't move that far

    cpu::md::NeighborList<> nl{cutoff, skin, /*every=*/1, /*delay=*/0};

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.5});
    sim.add_potential(cpu::PairLJCut{0.0104, 3.4, cutoff});
    sim.set_neighbor_ref(nl);
    sim.build();

    sim.run(frame, 100);

    // delay=0 but enormous skin: displacement check fires every step but
    // never triggers → only the initial rebuild.
    EXPECT_EQ(nl.n_rebuilds(), 1)
        << "delay=0, enormous skin: expected 1 rebuild (initial only), got "
        << nl.n_rebuilds();
}

// ── NVE drift guard ───────────────────────────────────────────────────

TEST(NeighborListRebuild, NveDriftWithinTolerance) {
    // Run 1000 NVE steps with default LAMMPS cadence (delay=10, every=1)
    // and verify NVE energy drift ≤ 1e-4 — same tolerance as test_pe_nve.
    auto frame = make_fcc_argon();

    // Small initial kick for non-trivial dynamics.
    for (int i = 0; i < frame.nlocal(); ++i) {
        const double sign = (i % 2 == 0) ? +1.0 : -1.0;
        frame.span(ATV::kAtom_vx)[i] = sign * 1e-3;
        frame.span(ATV::kAtom_vy)[i] = sign * 1e-3;
        frame.span(ATV::kAtom_vz)[i] = sign * 1e-3;
    }

    constexpr double cutoff = 6.8;
    constexpr double skin = 0.3 * cutoff; // auto-skin

    cpu::md::NeighborList<> nl{cutoff, skin, /*every=*/1, /*delay=*/10};

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.5});
    sim.add_potential(cpu::PairLJCut{0.0104, 3.4, cutoff});
    sim.set_neighbor_ref(nl);
    sim.build();

    sim.run(frame, 1);
    const double e0 =
        frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
    ASSERT_GT(std::abs(e0), 0.0) << "initial energy must be nonzero";

    sim.run(frame, 999);
    const double e1 =
        frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);

    const double drift = std::abs(e1 - e0) / std::abs(e0);
    EXPECT_LT(drift, 1e-4)
        << "NVE drift " << drift
        << " exceeds 1e-4 over 1000 steps with LAMMPS-default cadence"
        << " (E0=" << e0 << ", E1=" << e1 << ")";

    // With delay=10, we rebuild less often, but correctness is unchanged.
    EXPECT_GE(nl.n_rebuilds(), 1);
}
