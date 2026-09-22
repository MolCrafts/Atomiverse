/// @file tests/cpu/observe/test_voronoi_volume.cpp
/// @brief Unit: periodic Voronoi cell volumes (TF site-volume companion).

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/voronoi_volume.h"

using namespace ATV;

namespace {

cpu::Frame frame_at(const std::vector<std::array<double, 3>>& pos, double Lx,
                    double Ly, double Lz) {
    cpu::Frame f(static_cast<int>(pos.size()));
    for (std::size_t i = 0; i < pos.size(); ++i) {
        f.span(kAtom_x)[i] = pos[i][0];
        f.span(kAtom_y)[i] = pos[i][1];
        f.span(kAtom_z)[i] = pos[i][2];
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = static_cast<int>(i) + 1;
    }
    f.box = cpu::Box::orthorhombic(Lx, Ly, Lz);
    return f;
}

} // namespace

TEST(VoronoiVolume, SingleAtomGetsFullBox) {
    auto f = frame_at({{1.0, 2.0, 3.0}}, 3.0, 4.0, 5.0);
    const auto v = cpu::VoronoiVolume{}.volumes(f);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_NEAR(v[0], 60.0, 1e-10);
}

TEST(VoronoiVolume, UnequalOneDimensionalSplit) {
    // Sites at x = 0, 2, 5 in Lx = 10: periodic 1D widths 3.5 / 2.5 / 4.0.
    auto f = frame_at({{0.0, 2.0, 2.0}, {2.0, 2.0, 2.0}, {5.0, 2.0, 2.0}}, 10.0,
                      4.0, 4.0);
    const auto v = cpu::VoronoiVolume{}.volumes(f);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_NEAR(v[0], 3.5 * 16.0, 1e-9);
    EXPECT_NEAR(v[1], 2.5 * 16.0, 1e-9);
    EXPECT_NEAR(v[2], 4.0 * 16.0, 1e-9);
}

TEST(VoronoiVolume, FccQuarterCellVolume) {
    // FCC conventional cell: 4 sites, each Voronoi cell (rhombic
    // dodecahedron) has volume a0^3/4 exactly.
    const double a0 = 4.07;
    auto f = frame_at({{0.0, 0.0, 0.0},
                       {0.0, 0.5 * a0, 0.5 * a0},
                       {0.5 * a0, 0.0, 0.5 * a0},
                       {0.5 * a0, 0.5 * a0, 0.0}},
                      a0, a0, a0);
    const auto v = cpu::VoronoiVolume{}.volumes(f);
    const double vfcc = a0 * a0 * a0 / 4.0;
    ASSERT_EQ(v.size(), 4u);
    for (double vi : v)
        EXPECT_NEAR(vi, vfcc, vfcc * 1e-9);
}

TEST(VoronoiVolume, PerturbedLatticeSumsToBoxVolume) {
    // 2x2x2 simple cubic with deterministic asymmetric perturbations: the
    // tessellation must still tile the box exactly.
    const double a = 4.0;
    std::vector<std::array<double, 3>> pos;
    int k = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int l = 0; l < 2; ++l) {
                const double dx = 0.3 * std::sin(1.7 * ++k);
                const double dy = 0.3 * std::cos(2.3 * k);
                const double dz = 0.3 * std::sin(3.1 * k + 0.5);
                pos.push_back({i * a + dx, j * a + dy, l * a + dz});
            }
    auto f = frame_at(pos, 2.0 * a, 2.0 * a, 2.0 * a);
    const auto v = cpu::VoronoiVolume{}.volumes(f);
    double sum = 0.0;
    for (double vi : v) {
        EXPECT_GT(vi, 0.0);
        sum += vi;
    }
    const double vbox = 8.0 * a * a * a;
    EXPECT_NEAR(sum, vbox, vbox * 1e-8);
}

TEST(VoronoiVolume, ThrowsOnEmptyFrame) {
    cpu::Frame f(0);
    EXPECT_THROW(cpu::VoronoiVolume{}.volumes(f), std::invalid_argument);
}

TEST(VoronoiVolume, ThrowsOnTriclinicBox) {
    auto f = frame_at({{1.0, 1.0, 1.0}}, 10.0, 10.0, 10.0);
    Mat3 cell = Mat3::diagonal(10.0, 10.0, 10.0);
    cell(0, 1) = 2.0; // shear -> triclinic
    f.box = cpu::Box(cell);
    EXPECT_THROW(cpu::VoronoiVolume{}.volumes(f), std::invalid_argument);
}

TEST(VoronoiVolume, ThrowsOnCoincidentSites) {
    auto f = frame_at({{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}}, 10.0, 10.0, 10.0);
    EXPECT_THROW(cpu::VoronoiVolume{}.volumes(f), std::runtime_error);
}
