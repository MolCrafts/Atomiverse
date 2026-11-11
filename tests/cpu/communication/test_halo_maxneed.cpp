/// @file Periodic hop derivation, ported from LAMMPS comm_brick.cpp
/// a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b. GPL-2.0-only.
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/frame/box.h"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
using namespace ATV;
namespace {
using DD = cpu::md::mpi::DomainDecomposition;
DD domain(int nx, int coordinate) {
    DD dd;
    dd.dims[0] = nx;
    dd.coords[0] = coordinate;
    dd.compute_sublo_subhi(cpu::Box::cubic(20));
    dd.face_neighbors = {(coordinate + nx - 1) % nx,
                         (coordinate + 1) % nx,
                         coordinate,
                         coordinate,
                         coordinate,
                         coordinate};
    return dd;
}
} // namespace
TEST(TestHaloMaxneed, UniformPeriodicHopsIncludeExactEdge) {
    const auto box = cpu::Box::cubic(20);
    for (int nx : {1, 2, 4, 8}) {
        auto dd = domain(nx, 0);
        for (double shell : {0., 2., 5., 10., 21., 41.}) {
            auto hops = cpu::comm::Halo::derive_hops(dd, box, shell);
            EXPECT_EQ(hops.minus[0], static_cast<int>(shell * nx / 20) + 1);
            EXPECT_EQ(hops.plus[0], hops.minus[0]);
            EXPECT_EQ(hops.minus[1], static_cast<int>(shell / 20) + 1);
        }
    }
}
TEST(TestHaloMaxneed, NonUniformChainsIncludeCompletePeriodicRings) {
    const auto box = cpu::Box::cubic(20);
    const int minus[4] = {1, 1, 2, 3};
    const int plus[4] = {3, 2, 1, 1};
    for (int c = 0; c < 4; ++c) {
        auto dd = domain(4, c);
        dd.cuts[0] = {0., .5, .6, .7, 1.};
        auto near = cpu::comm::Halo::derive_hops(dd, box, 5.);
        EXPECT_EQ(near.minus[0], minus[c]);
        EXPECT_EQ(near.plus[0], plus[c]);
        auto far = cpu::comm::Halo::derive_hops(dd, box, 25.);
        EXPECT_EQ(far.minus[0], minus[c] + 4);
        EXPECT_EQ(far.plus[0], plus[c] + 4);
    }
}
TEST(TestHaloMaxneed, SetupTracksBoxShrinkAndIsIdempotent) {
    auto dd = domain(4, 1);
    cpu::comm::Halo halo;
    halo.setup(dd, cpu::Box::cubic(20), 6.6, 1);
    EXPECT_EQ(halo.max_hops(0), 2);
    halo.setup(dd, cpu::Box::cubic(10), 6.6, 1);
    EXPECT_EQ(halo.max_hops(0), 3);
    EXPECT_EQ(halo.nswap(0), 6);
    EXPECT_DOUBLE_EQ(halo.shell_width(), 6.6);
    halo.setup(dd, cpu::Box::cubic(10), 6.6, 1);
    EXPECT_EQ(halo.max_hops(0), 3);
    EXPECT_EQ(halo.max_hops(1), 1);
}
