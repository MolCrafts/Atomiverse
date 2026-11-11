/// Direct neighbor-index tests. Periodic rows come from the real Comm owner;
/// the oracle enumerates lattice images independently of its swap lists.
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "atomiverse/cpu/neighbor/cell_list.h"
#include "periodic_neighbor_fixture.h"

using namespace ATV;
namespace {
using Key = std::tuple<int, int, int, int, int>;

cpu::Frame lattice(int side) {
    cpu::Frame f(side * side * side);
    f.box = cpu::Box::cubic(side * 1.9);
    int i = 0;
    for (int x = 0; x < side; ++x)
        for (int y = 0; y < side; ++y)
            for (int z = 0; z < side; ++z, ++i) {
                const double jitter = 0.07 * std::sin(7.0 * i + 1);
                f.span(kAtom_x)[i] = (x + .5) * 1.9 + jitter;
                f.span(kAtom_y)[i] = (y + .5) * 1.9 - jitter;
                f.span(kAtom_z)[i] = (z + .5) * 1.9 + .5 * jitter;
                f.span(kAtom_tag)[i] = i + 1;
            }
    return f;
}
Vec3 position(const cpu::Frame& f, int i) {
    return {f.span(kAtom_x)[i], f.span(kAtom_y)[i], f.span(kAtom_z)[i]};
}
Key reverse(Key k) {
    auto [i, j, x, y, z] = k;
    return {j, i, -x, -y, -z};
}
Key canonical(Key k) { return std::min(k, reverse(k)); }
std::set<Key> oracle(const cpu::Frame& f, double reach) {
    std::set<Key> keys;
    int bounds[3];
    for (int d = 0; d < 3; ++d)
        bounds[d] = int(std::ceil(reach / f.box.face_height(d))) + 1;
    for (int i = 0; i < f.nlocal(); ++i)
        for (int j = 0; j < f.nlocal(); ++j)
            for (int x = -bounds[0]; x <= bounds[0]; ++x)
                for (int y = -bounds[1]; y <= bounds[1]; ++y)
                    for (int z = -bounds[2]; z <= bounds[2]; ++z) {
                        if (i == j && x == 0 && y == 0 && z == 0)
                            continue;
                        const Vec3 shift = f.box.to_cartesian(
                            Vec3{double(x), double(y), double(z)});
                        if ((position(f, i) - position(f, j) - shift)
                                .norm_squared()
                            < reach * reach)
                            keys.emplace(i, j, x, y, z);
                    }
    return keys;
}
std::multiset<Key> listed(const cpu::Frame& f, const cpu::md::PairList& pairs,
                          bool half) {
    std::map<TagInt, int> owners;
    for (int i = 0; i < f.nlocal(); ++i)
        owners.emplace(f.span(kAtom_tag)[i], i);
    std::multiset<Key> result;
    for (int ii = 0; ii < pairs.inum; ++ii) {
        const int i = pairs.ilist[ii];
        for (int k = 0; k < pairs.numneigh[i]; ++k) {
            const int row = pairs.neighbors[pairs.firstneigh[i] + k];
            const int j = owners.at(f.span(kAtom_tag)[row]);
            const Vec3 shift =
                f.box.to_fractional(position(f, row) - position(f, j));
            Key key{i, j, int(std::llround(shift[0])),
                    int(std::llround(shift[1])), int(std::llround(shift[2]))};
            result.insert(half ? canonical(key) : key);
        }
    }
    return result;
}
void check_lists(cpu::Frame& f, cpu::md::CellList& cells,
                 const std::set<Key>& expected_full) {
    std::set<Key> expected_half;
    for (auto key : expected_full)
        expected_half.insert(canonical(key));
    cpu::md::PairList full, half, derived;
    cells.refresh_full(f, f.n_total(), full);
    cells.refresh_half(f, f.n_total(), half);
    cells.derive_half(f, f.n_total(), full, derived);
    EXPECT_EQ(listed(f, full, false),
              (std::multiset<Key>(expected_full.begin(), expected_full.end())));
    EXPECT_EQ(listed(f, half, true),
              (std::multiset<Key>(expected_half.begin(), expected_half.end())));
    EXPECT_EQ(listed(f, derived, true),
              (std::multiset<Key>(expected_half.begin(), expected_half.end())));
    EXPECT_EQ(full.inum, f.nlocal());
    EXPECT_EQ(half.inum, f.nlocal());
    EXPECT_EQ(full.total_pairs, 2 * half.total_pairs);
}
void check_lists(cpu::Frame& f, cpu::md::CellList& cells, double reach) {
    check_lists(f, cells, oracle(f, reach));
}
} // namespace

TEST(FullNeighborList, IsUnionOfBothHalfDirections) {
    auto f = lattice(4);
    test::PeriodicNeighborFixture fixture(f, 3.0);
    cpu::md::CellList cells(2.6, .4);
    cells.init_extended(f, 3.0, fixture.context.md.dd, f.capacity());
    cells.build(f);
    check_lists(f, cells, 3.0);
}
TEST(FullNeighborList, LegsAreCompleteWithinCutoffAndBoundedBySkin) {
    auto f = lattice(2); // More than half a box: distinct images must survive.
    test::PeriodicNeighborFixture fixture(f, 3.0);
    cpu::md::CellList cells(2.6, .4);
    cells.init_extended(f, 3.0, fixture.context.md.dd, f.capacity());
    cells.build(f);
    check_lists(f, cells, 3.0);
}
TEST(FullNeighborList, RawRowsRetainFarSkinImages) {
    cpu::Frame f(3);
    f.box = cpu::Box::orthorhombic(7.6, 20, 20);
    f.span(kAtom_x)[0] = .2;
    f.span(kAtom_x)[1] = 3.8;
    f.span(kAtom_x)[2] = -3.8;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_y)[i] = f.span(kAtom_z)[i] = 10;
        f.span(kAtom_tag)[i] = i == 0 ? 1 : 2;
    }
    f.set_nlocal(1);
    f.set_nghost(2);
    cpu::md::CellList cells(3.77118, .3);
    cells.init(f, 4.07118);
    cells.build(f);
    const auto full = cells.build_full(f);
    ASSERT_EQ(full.numneigh[0], 2);
    int inside = 0;
    for (int k = 0; k < full.numneigh[0]; ++k) {
        const int j = full.neighbors[full.firstneigh[0] + k];
        inside += (position(f, 0) - position(f, j)).norm() < 3.77118;
    }
    EXPECT_EQ(inside, 1);
}
#ifdef _OPENMP
TEST(FullNeighborList, ReusedParallelRowsMatchBruteForceAcrossTeamSizes) {
    const int saved = omp_get_max_threads(), dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    for (int side : {2, 6}) {
        auto f = lattice(side);
        test::PeriodicNeighborFixture fixture(f, 3.0);
        cpu::md::CellList cells(2.6, .4);
        cells.init_extended(f, 3.0, fixture.context.md.dd, f.capacity());
        cells.build(f);
        // The brute-force oracle does not depend on the team size; build it
        // once so the four team sizes stay well inside the unit timeout.
        const auto expected_full = oracle(f, 3.0);
        for (int threads : {4, 1, 2, 4}) {
            omp_set_num_threads(threads);
            check_lists(f, cells, expected_full);
        }
    }
    omp_set_num_threads(saved);
    omp_set_dynamic(dynamic);
}
TEST(FullNeighborList, NestedSerializedBuildAndZeroOwnersDiscardOldRows) {
    const int saved = omp_get_max_threads(),
              levels = omp_get_max_active_levels();
    omp_set_num_threads(2);
    omp_set_max_active_levels(1);
    auto f = lattice(2);
    test::PeriodicNeighborFixture fixture(f, 3.0);
    cpu::md::CellList cells(2.6, .4);
    cells.init_extended(f, 3.0, fixture.context.md.dd, f.capacity());
    cells.build(f);
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        check_lists(f, cells, 3.0);
    }
    cpu::md::PairList full, half;
    cells.refresh_full(f, f.n_total(), full, 0);
    cells.refresh_half(f, f.n_total(), half, 0);
    EXPECT_EQ(full.total_pairs, 0);
    EXPECT_EQ(half.total_pairs, 0);
    omp_set_max_active_levels(levels);
    omp_set_num_threads(saved);
}
#endif
TEST(FullNeighborList, EmptyPopulationHasNoRowsOrNeighbors) {
    cpu::Frame f(0);
    f.box = cpu::Box::cubic(10);
    cpu::md::CellList cells(2, .3);
    cells.init(f, 2.3);
    cells.build(f);
    EXPECT_EQ(cells.build_full(f).total_pairs, 0);
    EXPECT_EQ(cells.build_half(f).total_pairs, 0);
}
TEST(FullNeighborList, RestrictingCentresKeepsActiveNeighborAtoms) {
    auto f = lattice(2);
    test::PeriodicNeighborFixture fixture(f, 3.0);
    cpu::md::CellList cells(2.6, .4);
    cells.init_extended(f, 3.0, fixture.context.md.dd, f.capacity());
    cells.build(f);
    cpu::md::PairList full, one;
    cells.refresh_full(f, f.n_total(), full);
    cells.refresh_full(f, f.n_total(), one, 1);
    EXPECT_EQ(one.inum, 1);
    ASSERT_EQ(one.numneigh[0], full.numneigh[0]);
    EXPECT_TRUE(std::equal(one.neighbors.begin(), one.neighbors.end(),
                           full.neighbors.begin()));
    EXPECT_THROW(cells.refresh_full(f, f.n_total(), one, f.n_total() + 1),
                 std::invalid_argument);
}
TEST(FullNeighborList, GeneralCellKeepsEveryPeriodicImageAndDerivedHalfShift) {
    cpu::Frame f(3);
    Mat3 h = Mat3::diagonal(3.2, 3.4, 3.6);
    h(0, 1) = .7;
    h(1, 0) = .2;
    h(0, 2) = .3;
    h(2, 0) = .1;
    h(1, 2) = .4;
    h(2, 1) = .2;
    f.box = cpu::Box(h);
    const Vec3 fractions[] = {{-.1, .2, 1.2}, {.4, .61, .3}, {.9, .87, .77}};
    for (int i = 0; i < 3; ++i) {
        const auto p = f.box.to_cartesian(fractions[i]);
        f.span(kAtom_x)[i] = p[0];
        f.span(kAtom_y)[i] = p[1];
        f.span(kAtom_z)[i] = p[2];
        f.span(kAtom_tag)[i] = i + 1;
    }
    test::PeriodicNeighborFixture fixture(f, 4.1);
    cpu::md::CellList cells(3.9, .2);
    cells.init_extended(f, 4.1, fixture.context.md.dd, f.capacity());
    cells.build(f);
    check_lists(f, cells, 4.1);
}
