/// @file tests/cpu/md/test_cell_list_grid_stale.cpp
/// @brief Unit test for `CellList::grid_stale` under anisotropic box change
///        (ac-005 of mc-gcmc-npt): a change to ONLY the y or z edge must
///        trigger a grid rebuild. Before the fix `grid_stale` compared
///        `length(0)` alone, so an anisotropic volume move silently kept a
///        grid sized for the old box and dropped pair candidates (the NPT
///        blow-up documented above the `grid_stale` call in cell_list.cpp).

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "atomiverse/cpu/communication/domain.h"

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/cell_list.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "periodic_neighbor_fixture.h"

using namespace ATV;

namespace {

cpu::Frame make_frame(double lx, double ly, double lz) {
    cpu::Frame frame(8);
    for (int i = 0; i < 8; ++i) {
        frame.span(ATV::kAtom_x)[i] = 0.5 + 2.0 * (i % 2);
        frame.span(ATV::kAtom_y)[i] = 0.5 + 2.0 * ((i / 2) % 2);
        frame.span(ATV::kAtom_z)[i] = 0.5 + 2.0 * (i / 4);
        frame.span(ATV::kAtom_mass)[i] = 1.0;
    }
    frame.box.set_cell(Mat3::diagonal(lx, ly, lz));
    return frame;
}

} // namespace

TEST(CellListGridStale, TriclinicBinsFindTiltedBoundaryPair) {
    cpu::Frame frame(2);
    Mat3 cell = Mat3::diagonal(10.0, 10.0, 10.0);
    cell(0, 1) = 4.0;
    frame.box = cpu::Box(cell);
    frame.span(kAtom_x)[0] = 2.08;
    frame.span(kAtom_y)[0] = 0.2;
    frame.span(kAtom_x)[1] = 5.92;
    frame.span(kAtom_y)[1] = 9.8;
    frame.span(kAtom_z)[0] = frame.span(kAtom_z)[1] = 5.0;
    // This real near pair lies seven x bins apart in an orthorhombic grid;
    // the ±2-bin stencil would never reach it, even with a correct Box metric.
    double dx = -3.84, dy = -9.6, dz = 0.0;
    frame.box.min_image(dx, dy, dz);
    ASSERT_LT(dx * dx + dy * dy + dz * dz, 0.8 * 0.8);
    test::PeriodicNeighborFixture fixture(frame, 1.0);
    cpu::md::CellList cells(0.8, 0.2);
    cells.init(frame, 1.0);
    cells.build(frame);
    const auto full = cells.build_full(frame);
    ASSERT_EQ(full.total_pairs, 2);
    const auto half = cells.build_half(frame);
    ASSERT_EQ(half.total_pairs, 1);
    const int i = half.numneigh[0] ? 0 : 1;
    const int j = half.neighbors[half.firstneigh[i]];
    dx = frame.span(kAtom_x)[i] - frame.span(kAtom_x)[j];
    dy = frame.span(kAtom_y)[i] - frame.span(kAtom_y)[j];
    dz = frame.span(kAtom_z)[i] - frame.span(kAtom_z)[j];
    EXPECT_NEAR(dx * dx + dy * dy + dz * dz, 0.16 * 0.16 + 0.4 * 0.4, 1e-13);
}

TEST(CellListGridStale, SameDiagonalShearInvalidatesAndRebuilds) {
    auto frame = make_frame(10.0, 10.0, 10.0);
    cpu::md::CellList cells(3.0, 0.5);
    cells.init(frame, 3.5);
    cells.build(frame);
    cpu::Context context;
    cpu::md::NeighborList<> neighbors(3.0, 0.5, 1000, 1000, true);
    neighbors.set_atom_sort(1000, 0.0, false);
    neighbors.build({context, frame});
    neighbors.launch({context, frame, 0, 0});
    Mat3 cell = frame.box.cell();
    cell(0, 1) = 0.3;
    frame.box.set_cell(cell);
    EXPECT_TRUE(cells.grid_stale(frame));
    EXPECT_TRUE(neighbors.decide_reneighbor(context, frame, true));
    cells.build(frame);
    EXPECT_FALSE(cells.grid_stale(frame));
    // Geometry invalidation bypasses every/delay with no atom displacement.
    const auto previous_version = context.md.pair_list->version;
    neighbors.launch({context, frame, 0, 1});
    EXPECT_GT(context.md.pair_list->version, previous_version);
    neighbors.destroy({context, frame});
}

TEST(CellListGridStale, EachAxisAloneTriggersRebuild) {
    for (int axis = 0; axis < 3; ++axis) {
        cpu::Frame frame = make_frame(10.0, 10.0, 10.0);
        cpu::md::CellList cl(/*cutoff=*/3.0, /*skin=*/0.5);
        cl.init(frame, 3.5);
        cl.build(frame);
        ASSERT_FALSE(cl.grid_stale(frame)) << "axis " << axis;

        Mat3 cell = frame.box.cell();
        cell(axis, axis) *= 1.2; // stretch ONLY this edge
        frame.box.set_cell(cell);
        EXPECT_TRUE(cl.grid_stale(frame))
            << "stretching only axis " << axis << " must stale the grid";
    }
}

// Exercise CellList alone with owner/ghost rows for each spatial partition.
// No Runtime, force evaluation, or MPI communication is involved.
TEST(CellListGridStale, BoundaryPairIsOwnedExactlyOnce) {
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (int threads : {1, 2}) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        for (int axis = 0; axis < 3; ++axis)
            for (int ranks : {2, 3, 4})
                for (bool reverse_tags : {false, true})
                    for (bool periodic : {false, true}) {
                        const double plane = periodic ? 0.0 : 40.0 / ranks;
                        for (StorageReal b :
                             {std::nextafter(StorageReal(plane),
                                             -std::numeric_limits<
                                                 StorageReal>::infinity()),
                              StorageReal(plane),
                              std::nextafter(StorageReal(plane),
                                             std::numeric_limits<
                                                 StorageReal>::infinity())}) {
                            // Keep periodic probes in the canonical box; the
                            // lower-side neighbor is represented at L-1.
                            const std::array<double, 2> pos = {
                                periodic ? 39.0 : plane - 1.0,
                                b < 0.0
                                    ? double(std::nextafter(StorageReal(40.0),
                                                            StorageReal(0.0)))
                                    : double(b)};
                            SCOPED_TRACE(::testing::Message()
                                         << "threads=" << threads << " axis="
                                         << axis << " ranks=" << ranks
                                         << " periodic=" << periodic
                                         << " reverse_tags=" << reverse_tags
                                         << " b=" << b);
                            int pairs = 0;
                            for (int rank = 0; rank < ranks; ++rank) {
                                cpu::md::mpi::DomainDecomposition dd;
                                dd.dims[axis] = ranks;
                                dd.coords[axis] = rank;
                                for (int d = 0; d < 3; ++d) {
                                    dd.sublo[d] =
                                        d == axis ? double(rank) / ranks : 0.0;
                                    dd.subhi[d] = d == axis
                                                      ? double(rank + 1) / ranks
                                                      : 1.0;
                                }
                                std::vector<int> rows;
                                for (int j = 0; j < 2; ++j)
                                    if (pos[j] >= 40.0 * dd.sublo[axis]
                                        && pos[j] < 40.0 * dd.subhi[axis])
                                        rows.push_back(j);
                                const int owned = static_cast<int>(rows.size());
                                if (owned == 0)
                                    continue;
                                for (int j = 0; j < 2; ++j)
                                    if (!(pos[j] >= 40.0 * dd.sublo[axis]
                                          && pos[j] < 40.0 * dd.subhi[axis]))
                                        rows.push_back(j);
                                cpu::Frame frame(2);
                                frame.box = cpu::Box::cubic(40.0);
                                for (int row = 0; row < 2; ++row) {
                                    const int atom = rows[row];
                                    double q = pos[atom];
                                    if (periodic && row >= owned)
                                        q += rank == 0 ? -40.0 : 40.0;
                                    frame.span(kAtom_x)[row] =
                                        axis == 0 ? q : 1.0;
                                    frame.span(kAtom_y)[row] =
                                        axis == 1 ? q : 1.0;
                                    frame.span(kAtom_z)[row] =
                                        axis == 2 ? q : 1.0;
                                    frame.span(kAtom_tag)[row] =
                                        reverse_tags ? 2 - atom : atom + 1;
                                }
                                cpu::md::CellList cl(12.0, 0.0);
                                cl.init_extended(frame, 12.0, &dd, 2);
                                cl.build(frame, 2);
                                cpu::md::PairList list;
                                cl.refresh_half(frame, 2, list, owned);
                                for (int i = 0; i < owned; ++i)
                                    pairs += list.numneigh[i];
                            }
                            EXPECT_EQ(pairs, 1);
                        }
                    }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

TEST(CellListGridStale, PeriodicUpperEdgeRoundingKeepsCrossAxisPair) {
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (int threads : {1, 2}) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        for (bool reverse_tags : {false, true}) {
            const double length = 15.78;
            const double upper =
                std::nextafter(StorageReal(length), StorageReal(0));
            int pairs = 0;
            for (int rank = 0; rank < 2; ++rank) {
                cpu::md::mpi::DomainDecomposition dd;
                dd.dims[0] = 2;
                dd.coords[0] = rank;
                for (int d = 0; d < 3; ++d) {
                    dd.sublo[d] = d == 0 ? double(rank) / 2 : 0;
                    dd.subhi[d] = d == 0 ? double(rank + 1) / 2 : 1.0;
                }
                cpu::Frame frame(2);
                frame.box = cpu::Box::cubic(length);
                for (int row = 0; row < 2; ++row) {
                    const int atom = row == 0 ? rank : 1 - rank;
                    double x = atom == 0 ? 0 : upper;
                    if (row == 1)
                        x += rank == 0 ? -length : length;
                    frame.span(kAtom_x)[row] = x;
                    frame.span(kAtom_y)[row] = atom + 1;
                    frame.span(kAtom_z)[row] = 1;
                    frame.span(kAtom_tag)[row] =
                        reverse_tags ? 2 - atom : atom + 1;
                }
                cpu::md::CellList cl(2.2, 0);
                cl.init_extended(frame, 2.2, &dd, 2);
                cl.build(frame, 2);
                cpu::md::PairList list;
                cl.refresh_half(frame, 2, list, 1);
                pairs += list.numneigh[0];
            }
            EXPECT_EQ(pairs, 1)
                << "threads=" << threads << " reverse_tags=" << reverse_tags;
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}
