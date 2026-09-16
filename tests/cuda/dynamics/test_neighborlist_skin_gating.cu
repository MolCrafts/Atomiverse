/// @file tests/cuda/md/test_neighborlist_skin_gating.cu
/// @brief Verlet-skin rebuild gating: `NeighborList::launch()` must skip
///        the full rebuild pipeline when per-atom displacement stays
///        below `skin/2`.
///
/// Covers `cuda-md-review-cleanup-02-nlist-skin-gating` ac-001/ac-002/
/// ac-003: the existing dead code (`launch_cluster_displacement`,
/// `ref_cluster_{x,y,z}`, `rebuild_flag`) is now wired into `launch()`
/// and gates the rebuild pipeline on a 1-step-lagged D2H of the device
/// flag.
///
/// Direct introspection: `cuda::NeighborList::rebuild_count()` exposes
/// the host-side rebuild counter; we drive NeighborList manually
/// through the Backend so the component instance stays addressable
/// (the categorical `set_neighbor` in `cuda::md::MD` moves the value
/// into the Plan, hiding it from test code).

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"

using namespace ATV;

namespace {

/// Build a frozen LJ-style lattice — uniform spacing, zero velocities.
/// No forces are added (test drives NeighborList in isolation), so
/// positions are pure inputs that don't evolve between launch() calls.
cpu::Frame make_frozen_lattice(int n_side, double spacing) {
    int n = n_side * n_side * n_side;
    double box_len = double(n_side) * spacing;

    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.span(kAtom_x)[idx] = (double(ix) + 0.5) * spacing;
                frame.span(kAtom_y)[idx] = (double(iy) + 0.5) * spacing;
                frame.span(kAtom_z)[idx] = (double(iz) + 0.5) * spacing;
                frame.span(kAtom_mass)[idx] = 1.0;
                frame.span(kAtom_type)[idx] = 0;
                frame.span(kAtom_vx)[idx] = 0.0;
                frame.span(kAtom_vy)[idx] = 0.0;
                frame.span(kAtom_vz)[idx] = 0.0;
                ++idx;
            }
    frame.box = cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

// ── ac-003: skin gate suppresses rebuilds for a frozen system ──────

TEST(NeighborListSkinGating, FrozenLatticeStaysAtOneRebuild) {
    // Build a frozen 64-atom cubic lattice. We drive only the
    // NeighborList (no integrator, no forces), so positions never
    // change → per-atom displacement is exactly zero → the skin gate
    // should never fire after the initial build.
    //
    // Geometry choices:
    //   n_side=4 (64 atoms) — smallest cube whose periodic images
    //                         cleanly separate at box_len = 2·cutoff.
    //   spacing=1.5         — box_len = 6.0 = 2·cutoff exactly.
    //   skin = 0.3·cutoff   — the default in `NeighborList(cutoff)`,
    //                         giving half_skin = 0.45.
    auto frame = make_frozen_lattice(4, 1.5); // 64 atoms, box_len=6.0
    const double cutoff = 3.0;
    const double skin = 0.9; // half_skin = 0.45

    cuda::Backend backend;
    backend.init();
    backend.materialize(frame);

    // Construct the NeighborList stand-alone so we keep a stable handle
    // to query `rebuild_count()` after the run. (The categorical
    // `cuda::md::MD::set_neighbor` would move the component into the
    // driver's Plan, hiding it.)
    cuda::NeighborList nl{cutoff, skin};
    PlanBuilder<cuda::Component> builder;
    builder.add({0, 0, &nl});
    auto plan = builder.build(ATV::slot_count<md::Phase>());
    backend.load(plan);

    cuda::BuildContext bctx{backend.context(), backend.frame()};
    nl.build(bctx);

    // Sanity: build() does not count as a launch; rebuild_count starts at 0
    EXPECT_EQ(nl.rebuild_count(), 0);

    const int n_steps = 30;
    for (int step = 0; step < n_steps; ++step) {
        backend.step(step);
    }
    backend.sync();

    // The first launch always rebuilds (no prior flag) → rebuild_count
    // is exactly 1 if the skin gate works. We allow `<= 2` to absorb a
    // theoretical 1-step-lag carry-over (the second launch reads the
    // first launch's flag-event; if the event hasn't materialized yet
    // on the host side, the implementation may treat it conservatively
    // and rebuild — though in practice the synchronous lattice keeps
    // the displacement flag exactly 0).
    EXPECT_LE(nl.rebuild_count(), 2)
        << "Skin gate should suppress rebuilds for a frozen lattice (got "
        << nl.rebuild_count() << " rebuilds over " << n_steps << " steps)";
    EXPECT_GE(nl.rebuild_count(), 1)
        << "Initial launch must rebuild at least once";

    cuda::DestroyContext dctx{backend.context(), backend.frame()};
    nl.destroy(dctx);
    backend.shutdown();
}

// ── ac-001/ac-002: gate forces rebuild when displacement exceeds half_skin ──
//
// Re-init the lattice between launches, perturbing positions by more
// than half_skin on selected steps. The next launch (after the lagged
// flag arrives) must rebuild.

TEST(NeighborListSkinGating, RebuildsWhenDisplacementExceedsHalfSkin) {
    // Same geometry as `FrozenLatticeStaysAtOneRebuild`. We perturb
    // atom 0 by 0.5 Å (just above half_skin = 0.45) so the gate must
    // fire on the next-launch-after-lag.
    auto frame = make_frozen_lattice(4, 1.5);
    const double cutoff = 3.0;
    const double skin = 0.9; // half_skin = 0.45

    cuda::Backend backend;
    backend.init();
    backend.materialize(frame);

    cuda::NeighborList nl{cutoff, skin};
    PlanBuilder<cuda::Component> builder;
    builder.add({0, 0, &nl});
    auto plan = builder.build(ATV::slot_count<md::Phase>());
    backend.load(plan);

    cuda::BuildContext bctx{backend.context(), backend.frame()};
    nl.build(bctx);

    // Initial launch — rebuilds.
    backend.step(0);
    backend.sync();
    int initial = nl.rebuild_count();
    EXPECT_EQ(initial, 1);

    // Perturb one atom by 0.5 (> half_skin=0.45) on the device side.
    // The simplest way: re-materialize the frame after editing the
    // host copy.
    frame.span(kAtom_x)[0] += 0.5;
    backend.materialize(frame);

    // Two more launches: the first arrives with the lagged flag from
    // the previous (initial) step (which saw zero displacement, so it
    // skips). The atom_displacement_max on this step will see the 0.5
    // displacement and set the flag. The third launch reads that flag
    // and rebuilds.
    backend.step(1);
    backend.sync();
    backend.step(2);
    backend.sync();

    // After 3 total launches, expect ≥ 2 rebuilds (initial + the one
    // triggered by the perturbation, with 1-step lag).
    EXPECT_GE(nl.rebuild_count(), 2) << "Skin gate must rebuild after a "
                                        "displacement of 0.5 > half_skin=0.45 "
                                        "(got "
                                     << nl.rebuild_count() << " rebuilds)";

    cuda::DestroyContext dctx{backend.context(), backend.frame()};
    nl.destroy(dctx);
    backend.shutdown();
}
