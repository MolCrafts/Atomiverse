/// @file Unit tests for fractional periodic halo geometry.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/frame/box.h"

using namespace ATV;

namespace {

using DomainDecomposition = cpu::md::mpi::DomainDecomposition;

constexpr double kProdL = 63.0016;  // [Angstrom]
constexpr double kProdShell = 11.0; // [Angstrom]

DomainDecomposition uniform_dd(const std::array<int, 3>& dims,
                               const std::array<int, 3>& coords,
                               const cpu::Box& box) {
    DomainDecomposition dd;
    for (int d = 0; d < 3; ++d) {
        dd.dims[d] = dims[d];
        dd.coords[d] = coords[d];
        const double L = box.length(d);
        const double w = L / static_cast<double>(dims[d]);
        dd.sublo[d] = w * static_cast<double>(coords[d]);
        dd.subhi[d] = w * static_cast<double>(coords[d] + 1);
    }
    dd.compute_sublo_subhi(box);
    return dd;
}

std::array<int, 3> max_hops_over_coords(const DomainDecomposition& dd,
                                        const cpu::Box& box, double shell) {
    std::array<int, 3> rounds{0, 0, 0};
    for (int d = 0; d < 3; ++d) {
        int axis_max = 0;
        for (int c = 0; c < dd.dims[d]; ++c) {
            DomainDecomposition scan = dd;
            scan.coords[d] = c;
            const cpu::comm::Halo::HopNeed hops =
                cpu::comm::Halo::derive_hops(scan, box, shell);
            axis_max =
                std::max(axis_max, std::max(hops.minus[d], hops.plus[d]));
        }
        rounds[static_cast<std::size_t>(d)] = axis_max;
    }
    return rounds;
}

std::string rounds_message(const DomainDecomposition& dd, const cpu::Box& box,
                           double shell) {
    try {
        (void)cpu::comm::Halo::derive_uniform_rounds(dd, box, shell);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

} // namespace

// ── 1. basics / domain golden: production closed form ─────────────────

TEST(TestHaloUniformRounds, ProductionGeometryRoundsAreTwoTwoOne) {
    const cpu::Box box = cpu::Box::cubic(kProdL);
    const DomainDecomposition dd =
        uniform_dd(/*dims=*/{8, 8, 4}, /*coords=*/{0, 0, 0}, box);

    const std::array<int, 3> rounds =
        cpu::comm::Halo::derive_uniform_rounds(dd, box, kProdShell);

    EXPECT_EQ(rounds[0], 2);
    EXPECT_EQ(rounds[1], 2);
    EXPECT_EQ(rounds[2], 1);
}

// ── 2. consistency with the serial Allreduce(MAX) equivalent ──────────

TEST(TestHaloUniformRounds, MatchesMaxOfDeriveHopsOnUniformGrid) {
    const cpu::Box box = cpu::Box::cubic(40.0);
    const DomainDecomposition dd =
        uniform_dd(/*dims=*/{4, 3, 2}, /*coords=*/{1, 1, 0}, box);
    const double shell = 11.0; // [Angstrom]; inside L - w_self on every axis

    const std::array<int, 3> rounds =
        cpu::comm::Halo::derive_uniform_rounds(dd, box, shell);
    const std::array<int, 3> expect = max_hops_over_coords(dd, box, shell);

    for (int d = 0; d < 3; ++d) {
        SCOPED_TRACE("axis " + std::to_string(d));
        EXPECT_EQ(rounds[static_cast<std::size_t>(d)],
                  expect[static_cast<std::size_t>(d)]);
    }
}

TEST(TestHaloUniformRounds, MatchesMaxOfDeriveHopsOnCutsGrid) {
    constexpr double L = 20.0; // [Angstrom]
    const cpu::Box box = cpu::Box::cubic(L);
    const double shell = 5.0; // [Angstrom]

    DomainDecomposition dd;
    dd.dims[0] = 4;
    dd.dims[1] = 3;
    dd.dims[2] = 1;
    dd.coords[0] = 0;
    dd.coords[1] = 0;
    dd.coords[2] = 0;
    // x widths {10, 2, 2, 6} A; y widths {4, 6, 10} A.
    dd.cuts[0] = std::vector<double>{0.0, 0.5, 0.6, 0.7, 1.0};
    dd.cuts[1] = std::vector<double>{0.0, 0.2, 0.5, 1.0};
    for (int d = 0; d < 3; ++d) {
        const std::vector<double>& cuts_d =
            dd.cuts[static_cast<std::size_t>(d)];
        if (!cuts_d.empty()) {
            dd.sublo[d] = cuts_d[static_cast<std::size_t>(dd.coords[d])] * L;
            dd.subhi[d] =
                cuts_d[static_cast<std::size_t>(dd.coords[d] + 1)] * L;
        } else {
            dd.sublo[d] = 0.0;
            dd.subhi[d] = L;
        }
    }

    const std::array<int, 3> rounds =
        cpu::comm::Halo::derive_uniform_rounds(dd, box, shell);
    const std::array<int, 3> expect = max_hops_over_coords(dd, box, shell);

    for (int d = 0; d < 3; ++d) {
        SCOPED_TRACE("axis " + std::to_string(d));
        EXPECT_EQ(rounds[static_cast<std::size_t>(d)],
                  expect[static_cast<std::size_t>(d)]);
    }
}

// ── 3. edges ──────────────────────────────────────────────────────────

TEST(TestHaloUniformRounds, UndecomposedAxisIncludesPeriodicRings) {
    const cpu::Box box = cpu::Box::orthorhombic(30.0, 8.0, 16.0);
    const DomainDecomposition dd =
        uniform_dd(/*dims=*/{2, 1, 1}, /*coords=*/{0, 0, 0}, box);

    const std::array<int, 3> rounds =
        cpu::comm::Halo::derive_uniform_rounds(dd, box, /*shell=*/12.0);

    EXPECT_EQ(rounds[0], 1); // ceil(12 * 2 / 30)
    EXPECT_EQ(rounds[1], 2);
    EXPECT_EQ(rounds[2], 1);
}

TEST(TestHaloUniformRounds, ShellEqualToSlabWidthIncludesNextSlabEdge) {
    const cpu::Box box = cpu::Box::cubic(20.0);
    // coords {1,1,1} makes `subhi - sublo == L/dims` bit-exact (w = 5 A).
    const DomainDecomposition dd =
        uniform_dd(/*dims=*/{4, 4, 4}, /*coords=*/{1, 1, 1}, box);
    const double w = 20.0 / 4.0; // 5 A, one slab

    const std::array<int, 3> rounds =
        cpu::comm::Halo::derive_uniform_rounds(dd, box, w);

    EXPECT_EQ(rounds[0], 2);
    EXPECT_EQ(rounds[1], 2);
    EXPECT_EQ(rounds[2], 2);
}

TEST(TestHaloUniformRounds, ZeroShellRetainsBoundaryRounds) {
    const cpu::Box box = cpu::Box::cubic(kProdL);
    const DomainDecomposition dd =
        uniform_dd(/*dims=*/{8, 8, 4}, /*coords=*/{0, 0, 0}, box);

    const std::array<int, 3> rounds =
        cpu::comm::Halo::derive_uniform_rounds(dd, box, /*shell=*/0.0);

    EXPECT_EQ(rounds[0], 1);
    EXPECT_EQ(rounds[1], 1);
    EXPECT_EQ(rounds[2], 1);
}

TEST(TestHaloUniformRounds, PeriodicReachIsRankConsistent) {
    const auto box = cpu::Box::orthorhombic(10., 20., 20.);
    for (int c = 0; c < 2; ++c) {
        auto dd = uniform_dd({2, 1, 1}, {c, 0, 0}, box);
        auto rounds = cpu::comm::Halo::derive_uniform_rounds(dd, box, 9.);
        EXPECT_EQ(rounds, (std::array<int, 3>{2, 1, 1}));
    }
}
