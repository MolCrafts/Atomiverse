/// @file tests/cpu/communication/test_halo_uniform_rounds.cpp
/// @brief Unit tests for `comm::Halo::derive_uniform_rounds`.
///
/// Spec `.claude/specs/lammps-comm-port.md`: the per-axis round count that
/// used to be an `MPI_Allreduce(MAX)` over `derive_hops` is a **pure local
/// sweep**. For each axis `d` it copies `dd`, walks `coords[d] = 0 … dims[d)`,
/// calls the existing `derive_hops`, and keeps `max(minus[d], plus[d])`.
/// Complexity `O(Σ dims[d])`. `Halo` stays MPI-free; hop math is not
/// rewritten.
///
/// Everything here is **serial**: no MPI is initialized. The DD is built by
/// hand (`dims` / `coords` / `periods` / `sublo` / `subhi` / `cuts`).
///
/// Surface pinned (new in `lammps-comm-port`, on the existing type):
/// @code{.cpp}
///   static std::array<int, 3>
///   Halo::derive_uniform_rounds(const DomainDecomposition& dd,
///                                   const Box& box, double shell_width);
/// @endcode
///
/// Categories: basics (production closed form), edge (undecomposed axis,
/// exact-slab shell, zero shell, non-periodic truncation), domain
/// (hard-coded production hops; minimum-image refusal at every coordinate).
///
/// References for the hard-coded expectations:
///   * S. J. Plimpton, J. Comput. Phys. **117**, 1 (1995),
///     DOI 10.1006/jcph.1995.1039 — ghost shell and per-hop forwarding.
///   * A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
///     DOI 10.1016/j.cpc.2021.108171 — `Comm::setup` `maxneed`/`nswap`
///     derivation; the Allreduce(MAX) this function replaces.

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

/// Production geometry of the spec: cubic `L = 63.0016 A`, `dims = {8,8,4}`.
constexpr double kProdL = 63.0016;  // [Angstrom]
constexpr double kProdShell = 11.0; // [Angstrom]

/// @brief Uniform brick over `box` at the given `dims` / `coords` / `periods`.
///
/// `sublo/subhi` are the closed-form slabs `w = L / dims`. Face neighbours
/// are unused by the hop derivation and left default.
DomainDecomposition uniform_dd(const std::array<int, 3>& dims,
                               const std::array<int, 3>& coords,
                               const cpu::Box& box,
                               const std::array<int, 3>& periods = {1, 1, 1}) {
    DomainDecomposition dd;
    for (int d = 0; d < 3; ++d) {
        dd.dims[d] = dims[d];
        dd.coords[d] = coords[d];
        dd.periods[d] = periods[d];
        const double L = box.length(d);
        const double w = L / static_cast<double>(dims[d]);
        dd.sublo[d] = w * static_cast<double>(coords[d]);
        dd.subhi[d] = w * static_cast<double>(coords[d] + 1);
    }
    return dd;
}

/// Serial equivalent of the deleted `MPI_Allreduce(MAX)`: copy `dd`, set
/// `coords[d] = c` (same as the production sweep), take
/// `max(minus[d], plus[d])` over every coordinate of the axis.
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

/// Message from `derive_uniform_rounds`, or "" when it did not throw.
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

/// Happy path (spec Testing strategy): `L = 63.0016 A`, `need = 11.0 A`,
/// `dims = {8, 8, 4}`.
///   x, y: ceil(11 * 8 / 63.0016) = ceil(1.3968) = 2
///   z:    ceil(11 * 4 / 63.0016) = ceil(0.6984) = 1
/// Hard-coded; any starting `coords` is fine — the sweep visits them all.
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

/// On a uniform brick, `derive_uniform_rounds` per axis equals the max over
/// every coordinate of `max(derive_hops.minus[d], derive_hops.plus[d])`.
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

/// Same pin on a cuts-based (RCB) grid, where per-rank hop counts differ
/// so the max is not equal to every coordinate's own hops. `cuts[d]` has
/// `dims[d] + 1` fractions in `[0, 1]`; slabs are uneven.
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
    dd.periods[0] = 1;
    dd.periods[1] = 1;
    dd.periods[2] = 1;
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

/// `dims[d] <= 1` => round count 0: the axis is not decomposed, produces
/// no swap rounds, and is **exempt from every width check**. A 12 A shell
/// on an 8 A undecomposed y axis is legal (EAM-style `2 * cutoff`).
TEST(TestHaloUniformRounds, UndecomposedAxisYieldsZero) {
    const cpu::Box box = cpu::Box::orthorhombic(30.0, 8.0, 16.0);
    const DomainDecomposition dd =
        uniform_dd(/*dims=*/{2, 1, 1}, /*coords=*/{0, 0, 0}, box);

    const std::array<int, 3> rounds =
        cpu::comm::Halo::derive_uniform_rounds(dd, box, /*shell=*/12.0);

    EXPECT_EQ(rounds[0], 1); // ceil(12 * 2 / 30)
    EXPECT_EQ(rounds[1], 0);
    EXPECT_EQ(rounds[2], 0);
}

/// A shell exactly one slab wide stays a single hop — the `ceil` convention.
TEST(TestHaloUniformRounds, ShellEqualToSlabWidthIsOneHop) {
    const cpu::Box box = cpu::Box::cubic(20.0);
    // coords {1,1,1} makes `subhi - sublo == L/dims` bit-exact (w = 5 A).
    const DomainDecomposition dd =
        uniform_dd(/*dims=*/{4, 4, 4}, /*coords=*/{1, 1, 1}, box);
    const double w = 20.0 / 4.0; // 5 A, one slab

    const std::array<int, 3> rounds =
        cpu::comm::Halo::derive_uniform_rounds(dd, box, w);

    EXPECT_EQ(rounds[0], 1);
    EXPECT_EQ(rounds[1], 1);
    EXPECT_EQ(rounds[2], 1);
}

/// A zero-width shell asks for no hops on any axis, including decomposed
/// ones.
TEST(TestHaloUniformRounds, ZeroShellYieldsAllZeros) {
    const cpu::Box box = cpu::Box::cubic(kProdL);
    const DomainDecomposition dd =
        uniform_dd(/*dims=*/{8, 8, 4}, /*coords=*/{0, 0, 0}, box);

    const std::array<int, 3> rounds =
        cpu::comm::Halo::derive_uniform_rounds(dd, box, /*shell=*/0.0);

    EXPECT_EQ(rounds[0], 0);
    EXPECT_EQ(rounds[1], 0);
    EXPECT_EQ(rounds[2], 0);
}

/// A non-periodic axis (`periods[d] == 0`) truncates the chain at the
/// global boundary and does **not** refuse, even when the same shell on a
/// periodic axis would violate `shell <= L - w_self`.
///
/// `dims = {2,1,1}`, `L_x = 10 A`, `shell = 9 A`: periodic would refuse
/// (`L - w_self = 5 A`). Non-periodic reach is 1, so both coordinates
/// truncate to 1 hop and the sweep must not throw.
TEST(TestHaloUniformRounds, NonPeriodicAxisTruncatesWithoutRefusal) {
    const cpu::Box box = cpu::Box::orthorhombic(10.0, 20.0, 20.0);
    const DomainDecomposition dd =
        uniform_dd(/*dims=*/{2, 1, 1}, /*coords=*/{0, 0, 0}, box,
                   /*periods=*/{0, 1, 1});

    std::array<int, 3> rounds{};
    ASSERT_NO_THROW(rounds = cpu::comm::Halo::derive_uniform_rounds(
                        dd, box, /*shell=*/9.0));
    EXPECT_EQ(rounds[0], 1);
    EXPECT_EQ(rounds[1], 0);
    EXPECT_EQ(rounds[2], 0);
}

// ── 4. domain: minimum-image refusal is rank-consistent ───────────────

/// `dims = {2,1,1}`, `L_x = 10 A`, `shell = 9 A` => `w_self = 5 A`,
/// bound `L - w_self = 5 A`. `refuse_minimum_image` throws
/// `std::runtime_error` naming "minimum-image", axis 0, and the numeric
/// bound. The throw must fire at **every** coordinate of that axis
/// (`coords[0] = 0` and `coords[0] = 1`): after the Allreduce flag is
/// gone, every rank must refuse locally.
TEST(TestHaloUniformRounds, RefusesMinimumImageAtEveryCoordinate) {
    const cpu::Box box = cpu::Box::orthorhombic(10.0, 20.0, 20.0);
    const double shell = 9.0; // [Angstrom]

    for (int c = 0; c < 2; ++c) {
        SCOPED_TRACE("coords[0] = " + std::to_string(c));
        const DomainDecomposition dd =
            uniform_dd(/*dims=*/{2, 1, 1}, /*coords=*/{c, 0, 0}, box);

        EXPECT_THROW(cpu::comm::Halo::derive_uniform_rounds(dd, box, shell),
                     std::runtime_error);

        const std::string message = rounds_message(dd, box, shell);
        ASSERT_FALSE(message.empty())
            << "derive_uniform_rounds accepted a 9 A shell on a 10 A box "
               "split 2 ways at coords[0] = "
            << c;
        EXPECT_NE(message.find("minimum-image"), std::string::npos) << message;
        EXPECT_NE(message.find("axis 0"), std::string::npos) << message;
        EXPECT_NE(message.find("5"), std::string::npos) << message;
    }
}
