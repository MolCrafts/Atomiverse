/// @file tests/cpu/communication/test_swap_list_maxneed.cpp
/// @brief Unit tests for `comm::SwapList`'s **multi-hop** derivation —
///        spec `.claude/specs/comm-topology-02-multihop.md`.
///
/// `comm-topology-01` delivered the swap table but capped every decomposed
/// axis at one hop and *threw* when the ghost shell was wider than the
/// rank's own slab. `-02` replaces that refusal with the LAMMPS
/// `Comm::setup` derivation:
///
///   * uniform partition (`dd.cuts[d]` empty), slab width `w_d = L_d/dims_d`
///       `max_hops(d) = ceil(need_d / w_d)`,
///   * non-uniform partition (RCB, `dd.cuts[d]` non-empty): walk the send
///     direction rank by rank, accumulating the *partner* slab widths until
///     they cover `need_d` — the thinnest chain in a send direction sets the
///     round count,
///   * `nswap(d) = 2 * max_hops(d)` rounds on axis `d`,
///   * the only surviving refusal is the **minimum-image** upper bound
///     `max_hops(d) <= dims[d] - 1`, i.e. `need_d <= L_d - w_self`.
///
/// Everything here is **serial**: `SwapList` performs no MPI, the uniform
/// derivation is closed form, and the per-rank non-uniform chain is a pure
/// function of `dims / coords / periods / cuts / L / shell`. The
/// `MPI_Allreduce(MAX)` that makes the round count rank-uniform is
/// `setup()`'s wrapper job under MPI, so the non-uniform cases below drive
/// the pure entry once per rank coordinate and take the maximum in the test
/// itself — no MPI is initialized by this file.
///
/// Surface pinned (new in `-02`, on the existing type):
/// @code{.cpp}
///   struct SwapList::HopNeed { int minus[3]; int plus[3]; };
///   static HopNeed SwapList::derive_hops(const DomainDecomposition& dd,
///                                        const Box& box,
///                                        double shell_width);
///   int    SwapList::max_hops(int d) const;    // reduced hops on axis d
///   int    SwapList::nswap(int d) const;       // == 2 * max_hops(d)
///   double SwapList::shell_width() const;      // [Angstrom]
/// @endcode
///
/// `HopNeed::minus[d]` counts the hops needed toward *decreasing* `coords[d]`
/// (the `-face` send chain), `plus[d]` toward increasing coords. Both are 0
/// on an undecomposed axis.
///
/// Categories: basics (uniform closed form), edge (undecomposed axis,
/// `need == w`, minimum-image refusal), lifecycle (`setup()` re-derivation
/// under a shrinking box), domain (hard-coded hop counts of the LAMMPS
/// `Comm::setup` slab chain).
///
/// References for the hard-coded expectations:
///   * S. J. Plimpton, J. Comput. Phys. **117**, 1 (1995),
///     DOI 10.1006/jcph.1995.1039 — ghost shell and per-hop forwarding.
///   * A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
///     DOI 10.1016/j.cpc.2021.108171 — `Comm::setup` `maxneed`/`nswap`
///     derivation, UNIFORM vs NONUNIFORM layout.
///   * M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*,
///     2nd ed., OUP (2017) — minimum-image convention (`r_c <= L/2`), the
///     generalization of which is `need <= L - w_self`.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/swap_list.h"
#include "atomiverse/cpu/frame/box.h"

using namespace ATV;

namespace {

using DomainDecomposition = cpu::md::mpi::DomainDecomposition;

/// Cubic box of the spec's narrow-subdomain geometry: 3^3 lattice at
/// a0 = 5.26 A => L = 15.78 A.
constexpr double kBoxL = 15.78; // [Angstrom]

/// Ghost shell of that geometry: pair cutoff 6.0 A + auto skin 0.6 A.
constexpr double kShell = 6.6; // [Angstrom]

/// @brief Uniform `dims = {dims_x, 1, 1}` brick over a cubic `L` box, seen
/// from grid coordinate `coords_x`.
///
/// The rank id equals `coords_x` under `DomainDecomposition::rank_of` when
/// `dims[1] == dims[2] == 1`. `sublo/subhi` are filled with the same
/// `w = L / dims_x` the closed form uses, and `coords_x == 1` is chosen by
/// the callers that care about exactness: `sublo = w`, `subhi = 2w` makes
/// `subhi - sublo == w` bit-exact for any binary `w`.
DomainDecomposition uniform_dd(int dims_x, int coords_x, double L = kBoxL) {
    DomainDecomposition dd;
    dd.dims[0] = dims_x;
    dd.dims[1] = 1;
    dd.dims[2] = 1;
    dd.coords[0] = coords_x;
    dd.coords[1] = 0;
    dd.coords[2] = 0;
    dd.periods[0] = 1;
    dd.periods[1] = 1;
    dd.periods[2] = 1;
    const double w = L / dims_x;
    dd.sublo[0] = w * coords_x;
    dd.subhi[0] = w * (coords_x + 1);
    dd.sublo[1] = 0.0;
    dd.subhi[1] = L;
    dd.sublo[2] = 0.0;
    dd.subhi[2] = L;
    const int lo = (coords_x - 1 + dims_x) % dims_x;
    const int hi = (coords_x + 1) % dims_x;
    // {-x, +x, -y, +y, -z, +z}; undecomposed axes point back at this rank.
    dd.face_neighbors =
        std::array<int, 6>{lo, hi, coords_x, coords_x, coords_x, coords_x};
    return dd;
}

/// @brief The spec's RCB case: `L = 20 A`, `dims = {4,1,1}`,
/// `cuts = {0, 0.5, 0.6, 0.7, 1.0}` => slab widths `{10, 2, 2, 6} A`.
constexpr double kRcbL = 20.0;                            // [Angstrom]
constexpr double kRcbCuts[5] = {0.0, 0.5, 0.6, 0.7, 1.0}; // fractions
constexpr double kRcbWidths[4] = {10.0, 2.0, 2.0, 6.0};   // [Angstrom]

DomainDecomposition rcb_dd(int coords_x) {
    DomainDecomposition dd;
    dd.dims[0] = 4;
    dd.dims[1] = 1;
    dd.dims[2] = 1;
    dd.coords[0] = coords_x;
    dd.coords[1] = 0;
    dd.coords[2] = 0;
    dd.periods[0] = 1;
    dd.periods[1] = 1;
    dd.periods[2] = 1;
    dd.cuts[0] = std::vector<double>{kRcbCuts[0], kRcbCuts[1], kRcbCuts[2],
                                     kRcbCuts[3], kRcbCuts[4]};
    dd.sublo[0] = kRcbCuts[coords_x] * kRcbL;
    dd.subhi[0] = kRcbCuts[coords_x + 1] * kRcbL;
    dd.sublo[1] = 0.0;
    dd.subhi[1] = kRcbL;
    dd.sublo[2] = 0.0;
    dd.subhi[2] = kRcbL;
    const int lo = (coords_x + 3) % 4;
    const int hi = (coords_x + 1) % 4;
    dd.face_neighbors =
        std::array<int, 6>{lo, hi, coords_x, coords_x, coords_x, coords_x};
    return dd;
}

cpu::Box cubic_box(double L) { return cpu::Box::cubic(L); }

/// Lower-case a message and fold `-` to a space, so a check for
/// "minimum image" matches "Minimum-image" / "minimum image" alike.
std::string normalized(std::string s) {
    for (char& c : s) {
        if (c == '-')
            c = ' ';
        else
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

/// Run `setup` and return the refusal message, or "" when it did not throw.
std::string setup_message(cpu::comm::SwapList& list,
                          const DomainDecomposition& dd, const cpu::Box& box,
                          double shell, int my_rank) {
    try {
        list.setup(dd, box, shell, my_rank);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

} // namespace

// ── 1. uniform: closed form ceil(need / w) ────────────────────────────

/// Happy path (spec Testing strategy, uniform block): `L = 15.78 A`,
/// `need = 6.6 A`.
///   dims = 4 -> w = 3.945 A  -> ceil(1.6730) = 2 hops, 4 swap rounds
///   dims = 8 -> w = 1.9725 A -> ceil(3.3460) = 4 hops, 8 swap rounds
///   dims = 2 -> w = 7.89 A   -> ceil(0.8365) = 1 hop,  2 swap rounds
/// All three are inside the minimum-image bound (`hops <= dims - 1`).
TEST(TestSwapListMaxneed, UniformHopsAreCeilOfShellOverSlab) {
    struct Case {
        int dims_x;
        int expect_hops;
    };
    const Case cases[3] = {{4, 2}, {8, 4}, {2, 1}};

    for (const Case& c : cases) {
        SCOPED_TRACE("dims_x = " + std::to_string(c.dims_x));
        const DomainDecomposition dd = uniform_dd(c.dims_x, /*coords_x=*/1);
        cpu::comm::SwapList list;
        ASSERT_NO_THROW(
            list.setup(dd, cubic_box(kBoxL), kShell, /*my_rank=*/1));

        EXPECT_EQ(list.max_hops(0), c.expect_hops);
        EXPECT_EQ(list.nswap(0), 2 * c.expect_hops);
    }
}

/// The shell width the table was derived against is readable back —
/// `shell_width()`, deliberately NOT a third `halo_cutoff()`
/// (`NeighborConsumer::halo_cutoff` and `HaloSend::halo_cutoff` already
/// carry that name for two different things).
TEST(TestSwapListMaxneed, ShellWidthIsRecorded) {
    cpu::comm::SwapList list;
    list.setup(uniform_dd(/*dims_x=*/4, /*coords_x=*/1), cubic_box(kBoxL),
               kShell, /*my_rank=*/1);
    EXPECT_DOUBLE_EQ(list.shell_width(), kShell);
}

/// The pure entry must reproduce the closed form on a uniform partition,
/// symmetrically in both send directions (every partner slab has the same
/// width, so the `-` chain and the `+` chain agree).
TEST(TestSwapListMaxneed, UniformDeriveHopsIsDirectionSymmetric) {
    const DomainDecomposition dd = uniform_dd(/*dims_x=*/4, /*coords_x=*/1);
    const cpu::comm::SwapList::HopNeed hops =
        cpu::comm::SwapList::derive_hops(dd, cubic_box(kBoxL), kShell);

    EXPECT_EQ(hops.minus[0], 2);
    EXPECT_EQ(hops.plus[0], 2);
    for (int d = 1; d < 3; ++d) {
        SCOPED_TRACE("axis " + std::to_string(d));
        EXPECT_EQ(hops.minus[d], 0);
        EXPECT_EQ(hops.plus[d], 0);
    }
}

// ── 2. edges: undecomposed axes and the exact-width shell ─────────────

/// `dims[d] == 1` => `max_hops(d) == 0`: the axis is not decomposed, so it
/// produces no swap rounds at all and its periodicity is the pair kernel's
/// minimum image. It is also **exempt from every width check** — an
/// EAM-style `2 * cutoff` shell on a thin undecomposed axis is legal today
/// and must stay legal (`comm-topology-01`
/// `TestSwapList.UndecomposedAxisExemptFromGuard`).
///
/// Note the deliberate difference from `-01`, which forced `maxneed = 1`
/// plus two `sendself` placeholder swaps on such an axis: the number pinned
/// here is the **hop derivation**, and `nswap(d) = 2 * max_hops(d)` is the
/// number of swap *rounds* that derivation asks for. Whether the expansion
/// keeps a `sendself` placeholder entry in the table is the expansion's
/// business (`comm-topology-01`), not this accessor's.
TEST(TestSwapListMaxneed, UndecomposedAxesNeedNoHops) {
    {
        SCOPED_TRACE("dims = {4,1,1}, shell inside the box");
        cpu::comm::SwapList list;
        ASSERT_NO_THROW(list.setup(uniform_dd(/*dims_x=*/4, /*coords_x=*/1),
                                   cubic_box(kBoxL), kShell, /*my_rank=*/1));
        for (int d = 1; d < 3; ++d) {
            SCOPED_TRACE("axis " + std::to_string(d));
            EXPECT_EQ(list.max_hops(d), 0);
            EXPECT_EQ(list.nswap(d), 0);
        }
    }
    {
        // Every axis undecomposed: no swap rounds anywhere, and a shell
        // wider than the whole box is still not a refusal.
        SCOPED_TRACE("dims = {1,1,1}, shell wider than L - w_self (= 0)");
        cpu::comm::SwapList list;
        ASSERT_NO_THROW(list.setup(uniform_dd(/*dims_x=*/1, /*coords_x=*/0),
                                   cubic_box(kBoxL), kShell, /*my_rank=*/0));
        for (int d = 0; d < 3; ++d) {
            SCOPED_TRACE("axis " + std::to_string(d));
            EXPECT_EQ(list.max_hops(d), 0);
            EXPECT_EQ(list.nswap(d), 0);
        }
    }
    {
        // The EAM shape: 30 x 8 x 16 A box, x split in two (sub = 15 A),
        // shell 12 A. It fits x (1 hop) and is wider than the whole
        // undecomposed y axis (8 A) — exempt, not refused.
        SCOPED_TRACE("thin undecomposed y axis, shell 12 A");
        DomainDecomposition dd = uniform_dd(/*dims_x=*/2, /*coords_x=*/0,
                                            /*L=*/30.0);
        dd.subhi[1] = 8.0;
        dd.subhi[2] = 16.0;
        cpu::comm::SwapList list;
        ASSERT_NO_THROW(list.setup(dd, cpu::Box::orthorhombic(30.0, 8.0, 16.0),
                                   12.0, /*my_rank=*/0));
        EXPECT_EQ(list.max_hops(0), 1); // ceil(12 / 15)
        EXPECT_EQ(list.nswap(0), 2);
        EXPECT_EQ(list.max_hops(1), 0);
        EXPECT_EQ(list.nswap(1), 0);
    }
}

/// `need` exactly equal to the slab width stays a SINGLE hop — the `ceil`
/// convention, and it must not jitter to 2 through a floating-point
/// `need / w` that lands one ulp above 1. `coords_x == 1` makes
/// `subhi - sublo == w` bit-exact, so a derivation reading either the
/// closed form (`L / dims`) or the stub's slab width sees the same number.
TEST(TestSwapListMaxneed, ShellEqualToSlabWidthIsOneHop) {
    const int dims_x = 4;
    const double w = kBoxL / dims_x; // 3.945 A
    const DomainDecomposition dd = uniform_dd(dims_x, /*coords_x=*/1);

    cpu::comm::SwapList exact;
    ASSERT_NO_THROW(exact.setup(dd, cubic_box(kBoxL), w, /*my_rank=*/1));
    EXPECT_EQ(exact.max_hops(0), 1);
    EXPECT_EQ(exact.nswap(0), 2);

    cpu::comm::SwapList just_under;
    ASSERT_NO_THROW(
        just_under.setup(dd, cubic_box(kBoxL), w - 1e-9, /*my_rank=*/1));
    EXPECT_EQ(just_under.max_hops(0), 1);
}

// ── 3. non-uniform (RCB): the send-direction slab chain ───────────────

/// Spec Testing strategy, non-uniform block. `L = 20 A`, `dims = 4`,
/// `cuts = {0, 0.5, 0.6, 0.7, 1.0}` => widths `{10, 2, 2, 6} A`,
/// `need = 5 A`. Walking a send direction accumulates the *partner* slab
/// widths (periodic wrap) until they cover `need`:
///
///   coords = 3: `-` chain w(2)=2, w(1)=2, w(0)=10 -> 2, 4, 14  => 3 hops
///               `+` chain w(0)=10                 -> 10        => 1 hop
///   coords = 2: `-` chain w(1)=2, w(0)=10         -> 2, 12     => 2 hops
///               `+` chain w(3)=6                  -> 6         => 1 hop
///   coords = 1: `-` chain w(0)=10                 -> 10        => 1 hop
///               `+` chain w(2)=2, w(3)=6          -> 2, 8      => 2 hops
///   coords = 0: `-` chain w(3)=6                  -> 6         => 1 hop
///               `+` chain w(1)=2, w(2)=2, w(3)=6  -> 2, 4, 10  => 3 hops
///
/// The global `MPI_Allreduce(MAX)` over all ranks and both directions is
/// **3**; it is `setup()`'s job under MPI, so this serial test takes the
/// maximum over the four per-rank results itself.
TEST(TestSwapListMaxneed, NonUniformChainCountsSlabsPerSendDirection) {
    const double need = 5.0; // [Angstrom]
    const int expect_minus[4] = {1, 1, 2, 3};
    const int expect_plus[4] = {3, 2, 1, 1};

    int global_max = 0;
    for (int c = 0; c < 4; ++c) {
        SCOPED_TRACE("coords_x = " + std::to_string(c) + " (slab width "
                     + std::to_string(kRcbWidths[c]) + " A)");
        const cpu::comm::SwapList::HopNeed hops =
            cpu::comm::SwapList::derive_hops(rcb_dd(c), cubic_box(kRcbL), need);
        EXPECT_EQ(hops.minus[0], expect_minus[c]);
        EXPECT_EQ(hops.plus[0], expect_plus[c]);
        // Undecomposed axes stay at zero regardless of the cut table.
        EXPECT_EQ(hops.minus[1], 0);
        EXPECT_EQ(hops.plus[1], 0);
        global_max = std::max(global_max, hops.minus[0]);
        global_max = std::max(global_max, hops.plus[0]);
    }
    EXPECT_EQ(global_max, 3) << "the reduced round count of the RCB case";
}

// ── 4. the one surviving refusal: the minimum-image bound ─────────────

/// `-01` refused a decomposed axis whose shell exceeded THIS rank's slab
/// ("use a larger box, fewer ranks along that axis, or a smaller cutoff").
/// `-02` derives extra hops instead; the only refusal left is the
/// minimum-image bound `max_hops(d) <= dims[d] - 1`, equivalently
/// `need <= L - w_self`: beyond it an atom would interact with its own
/// periodic image.
///
/// `L = 15.78 A`, `dims = 4` => `w = 3.945 A`, `L - w = 11.835 A`.
///   need = 11.0 A -> ceil(2.788) = 3 == dims - 1  -> legal, 3 hops
///   need = 12.0 A -> ceil(3.042) = 4  > dims - 1  -> refused
/// The message must name the minimum-image bound and must NOT repeat the
/// superseded "fewer ranks" how-to, which is no longer the fix.
TEST(TestSwapListMaxneed, RefusesShellBeyondMinimumImageBound) {
    const DomainDecomposition dd = uniform_dd(/*dims_x=*/4, /*coords_x=*/1);

    cpu::comm::SwapList deep;
    ASSERT_NO_THROW(deep.setup(dd, cubic_box(kBoxL), 11.0, /*my_rank=*/1));
    EXPECT_EQ(deep.max_hops(0), 3);
    EXPECT_EQ(deep.nswap(0), 6);

    cpu::comm::SwapList over;
    EXPECT_THROW(over.setup(dd, cubic_box(kBoxL), 12.0, /*my_rank=*/1),
                 std::runtime_error);

    const std::string message =
        normalized(setup_message(over, dd, cubic_box(kBoxL), 12.0,
                                 /*my_rank=*/1));
    ASSERT_FALSE(message.empty()) << "setup accepted a shell of 12 A over a "
                                     "15.78 A box split 4 ways";
    EXPECT_NE(message.find("minimum image"), std::string::npos) << message;
    EXPECT_EQ(message.find("fewer ranks"), std::string::npos)
        << "the -01 how-to no longer applies: " << message;
}

/// The bound is a property of the derivation, so the pure entry refuses it
/// too rather than silently returning `dims` hops (which would wrap a rank
/// onto itself). RCB widths `{10, 2, 2, 6} A`, `coords = 2`
/// (`w_self = 2 A`): `L - w_self = 18 A`, so a 19 A shell needs the whole
/// ring — `-` chain 2, 12, 18 still short of 19 with only the rank's own
/// slab left.
TEST(TestSwapListMaxneed, DeriveHopsRefusesBeyondMinimumImageBound) {
    EXPECT_THROW(cpu::comm::SwapList::derive_hops(rcb_dd(/*coords_x=*/2),
                                                  cubic_box(kRcbL), 19.0),
                 std::runtime_error);

    // Inside the bound the same chain is legal: the `-` chain 2, 12, 18
    // covers a 17 A shell on its third slab, i.e. 3 hops == dims - 1.
    const cpu::comm::SwapList::HopNeed hops = cpu::comm::SwapList::derive_hops(
        rcb_dd(/*coords_x=*/2), cubic_box(kRcbL), 17.0);
    EXPECT_EQ(hops.minus[0], 3);
}

// ── 5. lifecycle: setup() re-derives, and is idempotent ───────────────

/// `setup()` runs at the head of every full rebuild, so an NPT / deform box
/// shrink must RAISE the round count in place rather than throw or silently
/// truncate the shell. `L: 15.78 -> 10.0 A` at `dims = 4`:
/// `w: 3.945 -> 2.5 A`, `ceil(6.6 / 2.5) = 3` (still within
/// `dims - 1 = 3`, and `6.6 <= 10 - 2.5`). Re-running with identical
/// arguments changes nothing.
TEST(TestSwapListMaxneed, SetupRederivesHopsWhenBoxShrinks) {
    cpu::comm::SwapList list;
    list.setup(uniform_dd(/*dims_x=*/4, /*coords_x=*/1), cubic_box(kBoxL),
               kShell, /*my_rank=*/1);
    ASSERT_EQ(list.max_hops(0), 2);

    const DomainDecomposition shrunk =
        uniform_dd(/*dims_x=*/4, /*coords_x=*/1, /*L=*/10.0);
    list.setup(shrunk, cubic_box(10.0), kShell, /*my_rank=*/1);
    EXPECT_EQ(list.max_hops(0), 3);
    EXPECT_EQ(list.nswap(0), 6);
    EXPECT_DOUBLE_EQ(list.shell_width(), kShell);

    // Idempotent: the same decomposition + box + shell derives the same
    // table again (a full rebuild happens every reneighbor step).
    list.setup(shrunk, cubic_box(10.0), kShell, /*my_rank=*/1);
    EXPECT_EQ(list.max_hops(0), 3);
    EXPECT_EQ(list.nswap(0), 6);
    EXPECT_EQ(list.max_hops(1), 0);
    EXPECT_EQ(list.max_hops(2), 0);
}
