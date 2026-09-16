/// @file tests/cpu/md/test_neighbor_list_arena.cpp
/// @brief Phase 2 cleanup regression: `CellList::refresh_half` reuses the
///        caller-supplied `PairList` buffers across steady-state calls.
///
/// Prior to the arena refactor, `NeighborList::launch` ran
/// `pairs_ = strategy_.build_half(...)` every step, which re-created the
/// CSR arrays and the flat `neighbors` buffer fresh each call. That
/// violated the "no allocation in `launch()`" component-lifecycle rule.
///
/// This test drives `CellList::refresh_half` directly against a stable
/// frame and asserts that, after one warmup rebuild, **no subsequent call
/// moves** the backing storage of any PairList array. All four arrays
/// (`ilist`, `numneigh`, `firstneigh`, `neighbors`) are `std::vector<int>`
/// and should stay at their amortised capacity without reallocation.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/cell_list.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

using namespace ATV;

namespace {

cpu::Frame make_uniform_cubic(int n_per_side, double box_len) {
    const int n = n_per_side * n_per_side * n_per_side;
    cpu::Frame frame(n);
    const double step = box_len / static_cast<double>(n_per_side);
    int idx = 0;
    for (int ix = 0; ix < n_per_side; ++ix) {
        for (int iy = 0; iy < n_per_side; ++iy) {
            for (int iz = 0; iz < n_per_side; ++iz) {
                frame.span(ATV::kAtom_x)[idx] =
                    (static_cast<double>(ix) + 0.5) * step;
                frame.span(ATV::kAtom_y)[idx] =
                    (static_cast<double>(iy) + 0.5) * step;
                frame.span(ATV::kAtom_z)[idx] =
                    (static_cast<double>(iz) + 0.5) * step;
                frame.span(ATV::kAtom_mass)[idx] = 1.0;
                ++idx;
            }
        }
    }
    frame.box = ATV::cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

TEST(NeighborListArena, RefreshHalfReusesBuffersInSteadyState) {
    // 5^3 = 125 atoms in a 10 Å box (step 2.0 Å). Cutoff 2.6 finds
    // nearest-and-second-nearest neighbors via the cell list.
    auto frame = make_uniform_cubic(5, 10.0);
    const int n = frame.nlocal();

    // cutoff = 2.6 Å, skin = 0.4 Å ⇒ cutoff_skin = 3.0 Å (covers multiple
    // cell layers in the 10 Å box so the cell list is non-trivial).
    cpu::md::CellList cl{2.6, 0.4};
    cl.init(frame, 3.0);
    cl.build(frame, n);

    cpu::md::PairList pairs;
    cl.refresh_half(frame, n, pairs); // warmup 1

    // Steady-state reference: pointers + capacity after the first real
    // rebuild. All subsequent rebuilds must reuse these same allocations.
    const void* ilist_ptr = pairs.ilist.data();
    const void* numneigh_ptr = pairs.numneigh.data();
    const void* firstneigh_ptr = pairs.firstneigh.data();
    const void* neighbors_ptr = pairs.neighbors.data();
    const std::size_t neighbors_cap = pairs.neighbors.capacity();
    const int ref_total_pairs = pairs.total_pairs;

    ASSERT_GT(ref_total_pairs, 0)
        << "expected a nonzero neighbor count on a 125-atom cubic grid";

    // Many more refreshes against the same unchanged frame.
    for (int step = 0; step < 100; ++step) {
        cl.refresh_half(frame, n, pairs);

        EXPECT_EQ(pairs.total_pairs, ref_total_pairs)
            << "neighbor count drifted on step " << step
            << " despite identical frame";

        EXPECT_EQ(pairs.ilist.data(), ilist_ptr)
            << "ilist backing storage moved on step " << step;
        EXPECT_EQ(pairs.numneigh.data(), numneigh_ptr)
            << "numneigh backing storage moved on step " << step;
        EXPECT_EQ(pairs.firstneigh.data(), firstneigh_ptr)
            << "firstneigh backing storage moved on step " << step;
        EXPECT_EQ(pairs.neighbors.data(), neighbors_ptr)
            << "neighbors vector moved on step " << step
            << " (std::vector reallocated)";
        EXPECT_EQ(pairs.neighbors.capacity(), neighbors_cap)
            << "neighbors vector capacity grew on step " << step << " from "
            << neighbors_cap << " to " << pairs.neighbors.capacity();
    }
}
