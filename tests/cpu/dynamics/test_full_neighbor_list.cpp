/// @file tests/cpu/neighbor/test_full_neighbor_list.cpp
/// @brief Unit: the opt-in FULL neighbor list (ctx.md.full_list_requested).
///
/// Pins the one property every many-body style depends on: for each owned atom
/// i, the full list carries EVERY neighbour of i within the cutoff, in both
/// directions — i.e. it is exactly the union of the half list's two directions.
/// Until this landed, CellList::refresh_full was written but called by nothing.

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/cell_list.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

using namespace ATV;

namespace {

/// Deterministic jittered lattice — several call sites below.
cpu::Frame lattice(int n_side, double spacing) {
    const int n = n_side * n_side * n_side;
    cpu::Frame f(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                // A reproducible jitter so the pairs are not all at equal r.
                const double j = 0.07 * std::sin(double(7 * idx + 1));
                f.span(ATV::kAtom_x)[idx] = (double(ix) + 0.5) * spacing + j;
                f.span(ATV::kAtom_y)[idx] = (double(iy) + 0.5) * spacing - j;
                f.span(ATV::kAtom_z)[idx] =
                    (double(iz) + 0.5) * spacing + 0.5 * j;
                f.span(ATV::kAtom_mass)[idx] = 1.0;
                f.span(ATV::kAtom_type)[idx] = 0;
                ++idx;
            }
    f.box = ATV::cpu::Box::cubic(n_side * spacing);
    return f;
}

} // namespace

/// The full list is the union of the half list's two directions.
TEST(FullNeighborList, IsUnionOfBothHalfDirections) {
    const double cutoff = 2.6;
    cpu::Frame f = lattice(4, 1.9); // 64 atoms, box 7.6 A
    const int n = f.nlocal();

    cpu::md::CellList cl(cutoff);
    cl.init(f, cutoff);
    cl.build(f, n);

    cpu::md::PairList half, full;
    cl.refresh_half(f, n, half, -1);
    cl.refresh_full(f, n, full);

    // Symmetrize the half list by hand: every (i,j) also counts for j.
    std::vector<std::set<int>> expected(n);
    for (int ii = 0; ii < half.inum; ++ii) {
        const int i = half.ilist[ii];
        for (int jj = 0; jj < half.numneigh[i]; ++jj) {
            const int j = half.neighbors[half.firstneigh[i] + jj];
            expected[i].insert(j);
            expected[j].insert(i);
        }
    }

    std::vector<std::set<int>> got(n);
    for (int ii = 0; ii < full.inum; ++ii) {
        const int i = full.ilist[ii];
        for (int jj = 0; jj < full.numneigh[i]; ++jj)
            got[i].insert(full.neighbors[full.firstneigh[i] + jj]);
    }

    std::size_t n_half_pairs = 0, n_full_legs = 0;
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(got[i], expected[i])
            << "neighbour set differs for atom " << i;
        EXPECT_EQ(got[i].count(i), 0u)
            << "atom " << i << " is its own neighbour";
        n_full_legs += got[i].size();
    }
    for (int i = 0; i < n; ++i)
        n_half_pairs += expected[i].size();

    EXPECT_GT(n_full_legs, 0u)
        << "the cutoff caught no pairs — test is vacuous";
    // Every pair appears exactly twice in a full list, once in each direction.
    EXPECT_EQ(n_full_legs % 2, 0u);
}

/// Every leg of the full list lies inside the Verlet range (cutoff + skin), and
/// no pair inside the cutoff is missing.
///
/// The list is a VERLET list: it deliberately carries pairs out to cutoff + skin
/// so it survives a few steps of motion without a rebuild. Screening at the true
/// cutoff is the force kernel's job, not the list's — so the list's invariant is
/// "complete within the cutoff, and bounded by cutoff + skin", not "exactly the
/// cutoff".
TEST(FullNeighborList, LegsAreCompleteWithinCutoffAndBoundedBySkin) {
    const double cutoff = 2.6, skin = 0.4;
    cpu::Frame f = lattice(4, 1.9);
    const int n = f.nlocal();

    cpu::md::CellList cl(cutoff, skin);
    cl.init(f, cutoff + skin);
    cl.build(f, n);

    cpu::md::PairList full;
    cl.refresh_full(f, n, full);

    // What the list gives us, per atom.
    std::vector<std::set<int>> listed(n);
    for (int ii = 0; ii < full.inum; ++ii) {
        const int i = full.ilist[ii];
        for (int jj = 0; jj < full.numneigh[i]; ++jj)
            listed[i].insert(full.neighbors[full.firstneigh[i] + jj]);
    }

    // Brute-force O(N^2) ground truth.
    std::size_t n_inside = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            double dx = f.span(ATV::kAtom_x)[i] - f.span(ATV::kAtom_x)[j];
            double dy = f.span(ATV::kAtom_y)[i] - f.span(ATV::kAtom_y)[j];
            double dz = f.span(ATV::kAtom_z)[i] - f.span(ATV::kAtom_z)[j];
            f.box.min_image(dx, dy, dz);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (r <= cutoff) {
                EXPECT_EQ(listed[i].count(j), 1u)
                    << "pair (" << i << "," << j << ") at r=" << r
                    << " is inside the cutoff but MISSING from the full list";
                ++n_inside;
            }
            if (r > cutoff + skin) {
                EXPECT_EQ(listed[i].count(j), 0u)
                    << "pair (" << i << "," << j << ") at r=" << r
                    << " is beyond cutoff+skin yet listed";
            }
        }
    }
    EXPECT_GT(n_inside, 0u) << "no pairs inside the cutoff — test is vacuous";
}
