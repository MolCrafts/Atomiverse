/// @file tests/mpi/test_bcast_frame.cpp
/// @brief cpu::comm::bcast_frame roundtrip tests
///        (mpi-launcher-embedded-python-01).
///
/// Launched under `mpirun -n N`. Runs at N=1, 2, 4 per CTest config.
/// bcast_frame is a collective on MPI_COMM_WORLD: the root rank packs every
/// owned atom row via comm::pack_exchange, collectively validates the atoms
/// and typed-meta schemas, and broadcasts explicit cell/meta/native row bytes.
/// Every non-root rank reconstructs a full replica (all rows owned,
/// nghost = 0, tag→local map rebuilt, cell + metadata overwritten). The wire is
/// pure byte transit — every assertion below is EXACT (==, zero tolerance).
///
/// Verifies:
///   - Multi-rank roundtrip: every schema field, per-atom bonded topology,
///     box and every typed-meta entry replicate exactly; replica invariants
///     (nlocal == n, nghost == 0, has_bonded(), map usable); root untouched.
///   - World size 1: strict no-op — the frame is field-for-field unchanged.
///   - 0-atom frame: header-only broadcast completes; box/meta still land.
///   - No-topology frame: has_bonded() stays false on the replica.
///   - Runtime exchange columns ride exactly; divergent registration throws
///     collectively before payload.
///   - Runtime typed metadata preserves exact dtype and values.
///
/// Not testable here: the MPI-uninitialized degrade path (this binary's main
/// initializes MPI before any test runs); it is covered by the non-MPI-build
/// compile path.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/bcast_frame.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

int world_size() {
    int s = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}

int world_rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

/// Noncontiguous, nonmonotonic global tags for the 8-atom reference frame.
#if defined(ATV_ID_BIGBIG)
constexpr TagInt kRefTags[8] = {9007199254740993LL, 101, 13, 55, 2, 89, 34, 68};
#else
constexpr TagInt kRefTags[8] = {700000003, 101, 13, 55, 2, 89, 34, 68};
#endif

/// @brief Deterministic 8-atom reference frame: every core field distinct and
///        nonzero, bonded topology (bond/angle/dihedral/special), an
///        orthorhombic non-cubic box, and nonzero typed meta. Constructed
///        from literals only, so two independent constructions are bitwise
///        identical — each rank builds its own expected copy locally.
cpu::Frame make_reference_frame() {
    constexpr int n = 8;
    cpu::Frame f(n);
    for (int i = 0; i < n; ++i) {
        f.span(ATV::kAtom_x)[i] = StorageReal(0.5 + 1.25 * i);
        f.span(ATV::kAtom_y)[i] = StorageReal(-1.0 + 0.5 * i);
        f.span(ATV::kAtom_z)[i] = StorageReal(2.0 + 0.75 * i);
        f.span(ATV::kAtom_vx)[i] = StorageReal(0.01 * (i + 1));
        f.span(ATV::kAtom_vy)[i] = StorageReal(-0.02 * (i + 1));
        f.span(ATV::kAtom_vz)[i] = StorageReal(0.03 * (i + 1));
        f.span(ATV::kAtom_fx)[i] = AccReal(1.5 * i - 3.0);
        f.span(ATV::kAtom_fy)[i] = AccReal(0.25 * i + 0.125);
        f.span(ATV::kAtom_fz)[i] = AccReal(-0.5 * i + 1.0);
        f.span(ATV::kAtom_mass)[i] = StorageReal(12.011 + i);
        f.span(ATV::kAtom_charge)[i] =
            StorageReal((i % 2 == 0 ? 0.42 : -0.42) + 0.01 * i); // nonzero
        f.span(ATV::kAtom_type)[i] = 1 + (i % 3);
        f.span(ATV::kAtom_tag)[i] = kRefTags[i]; // noncontiguous
        f.span(ATV::kAtom_image)(i, 0) = static_cast<std::int8_t>(i - 4);
        f.span(ATV::kAtom_image)(i, 1) = static_cast<std::int8_t>(2 * i - 7);
        f.span(ATV::kAtom_image)(i, 2) = static_cast<std::int8_t>(-i);
    }
    // add_* prerequisites: nlocal set (Frame ctor), tags set, map usable.
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    // Chain over rows 0-1-2-3 by global tag: 7 - 101 - 13 - 55.
    f.add_bond(kRefTags[0], kRefTags[1], 1);
    f.add_bond(kRefTags[1], kRefTags[2], 2);
    f.add_bond(kRefTags[2], kRefTags[3], 3);
    f.add_angle(kRefTags[0], kRefTags[1], kRefTags[2], 1);
    f.add_angle(kRefTags[1], kRefTags[2], kRefTags[3], 2);
    f.add_dihedral(kRefTags[0], kRefTags[1], kRefTags[2], kRefTags[3], 1);
    f.build_special_list();

    f.box = ATV::cpu::Box::orthorhombic(12.0, 15.5,
                                        19.25); // orthorhombic, non-cubic

    f.meta.get(ATV::kEnergy) = -123.5;
    f.meta.get(ATV::kEnergyPair) = -100.25;
    f.meta.get(ATV::kEnergyBond) = -13.5;
    f.meta.get(ATV::kEnergyAngle) = -6.25;
    f.meta.get(ATV::kEnergyDihedral) = -3.5;
    f.meta.get(ATV::kKinetic) = 45.75;
    f.meta.get(ATV::kConserved) = -77.75;
    for (int k = 0; k < 6; ++k) {
        f.meta.get(ATV::kStress)[k] = 0.5 * (k + 1);
        f.meta.get(ATV::kStressKinetic)[k] = -0.25 * (k + 1);
    }
    return f;
}

/// @brief Assert `got.box` == `want.box` element-for-element (exact).
void expect_box_equal(const cpu::Frame& got, const cpu::Frame& want) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_EQ(got.box.cell()(r, c), want.box.cell()(r, c))
                << "box(" << r << ", " << c << ")";
    EXPECT_EQ(got.box.is_orthorhombic(), want.box.is_orthorhombic());
}

/// @brief Assert every serializable built-in typed-meta value equal (exact).
void expect_meta_equal(const cpu::Frame& got, const cpu::Frame& want) {
    EXPECT_EQ(got.meta.get(ATV::kEnergy), want.meta.get(ATV::kEnergy));
    EXPECT_EQ(got.meta.get(ATV::kEnergyPair), want.meta.get(ATV::kEnergyPair));
    EXPECT_EQ(got.meta.get(ATV::kEnergyBond), want.meta.get(ATV::kEnergyBond));
    EXPECT_EQ(got.meta.get(ATV::kEnergyAngle),
              want.meta.get(ATV::kEnergyAngle));
    EXPECT_EQ(got.meta.get(ATV::kEnergyDihedral),
              want.meta.get(ATV::kEnergyDihedral));
    EXPECT_EQ(got.meta.get(ATV::kKinetic), want.meta.get(ATV::kKinetic));
    EXPECT_EQ(got.meta.get(ATV::kConserved), want.meta.get(ATV::kConserved));
    for (int k = 0; k < 6; ++k) {
        EXPECT_EQ(got.meta.get(ATV::kStress)[k], want.meta.get(ATV::kStress)[k])
            << "k=" << k;
        EXPECT_EQ(got.meta.get(ATV::kStressKinetic)[k],
                  want.meta.get(ATV::kStressKinetic)[k])
            << "k=" << k;
    }
}

/// @brief Assert every per-atom field of rows [0, want.nlocal()) equal
///        (exact ==; the payload is byte transit — no tolerance is legal),
///        plus the replica invariants (counts, has_bonded, tag→local map).
///        Topology strides may legitimately differ (the receiver grows
///        defensively), so wide arrays are compared through the per-row
///        counts, never by full stride.
void expect_frame_equal(const cpu::Frame& got, const cpu::Frame& want) {
    const int n = want.nlocal();
    ASSERT_EQ(got.nlocal(), n);
    EXPECT_EQ(got.nghost(), 0);
    EXPECT_EQ(got.has_bonded(), want.has_bonded());

    for (int i = 0; i < n; ++i) {
        SCOPED_TRACE("atom row " + std::to_string(i));
        EXPECT_EQ(got.span(ATV::kAtom_x)[i], want.span(ATV::kAtom_x)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_y)[i], want.span(ATV::kAtom_y)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_z)[i], want.span(ATV::kAtom_z)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_vx)[i], want.span(ATV::kAtom_vx)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_vy)[i], want.span(ATV::kAtom_vy)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_vz)[i], want.span(ATV::kAtom_vz)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_fx)[i], want.span(ATV::kAtom_fx)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_fy)[i], want.span(ATV::kAtom_fy)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_fz)[i], want.span(ATV::kAtom_fz)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_mass)[i], want.span(ATV::kAtom_mass)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_charge)[i],
                  want.span(ATV::kAtom_charge)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_type)[i], want.span(ATV::kAtom_type)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_tag)[i], want.span(ATV::kAtom_tag)[i]);
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(got.span(ATV::kAtom_image)(i, d),
                      want.span(ATV::kAtom_image)(i, d))
                << "image axis " << d;

        // tag → local map is rebuilt and usable on the replica.
        const auto& tag_index = got.meta.get(cpu::kTagIndex);
        ASSERT_EQ(tag_index.count(want.span(ATV::kAtom_tag)[i]), 1u);
        EXPECT_EQ(tag_index.at(want.span(ATV::kAtom_tag)[i]), i);

        // Per-atom bonded topology, through the counts (stride-agnostic).
        ASSERT_EQ(got.span(ATV::kAtom_num_bond)[i],
                  want.span(ATV::kAtom_num_bond)[i]);
        for (int m = 0; m < want.span(ATV::kAtom_num_bond)[i]; ++m) {
            EXPECT_EQ(got.span(ATV::kAtom_bond_atom)(i, m),
                      want.span(ATV::kAtom_bond_atom)(i, m));
            EXPECT_EQ(got.span(ATV::kAtom_bond_type)(i, m),
                      want.span(ATV::kAtom_bond_type)(i, m));
        }
        ASSERT_EQ(got.span(ATV::kAtom_num_angle)[i],
                  want.span(ATV::kAtom_num_angle)[i]);
        for (int m = 0; m < want.span(ATV::kAtom_num_angle)[i]; ++m) {
            EXPECT_EQ(got.span(ATV::kAtom_angle_atom)(i, 2 * m + 0),
                      want.span(ATV::kAtom_angle_atom)(i, 2 * m + 0));
            EXPECT_EQ(got.span(ATV::kAtom_angle_atom)(i, 2 * m + 1),
                      want.span(ATV::kAtom_angle_atom)(i, 2 * m + 1));
            EXPECT_EQ(got.span(ATV::kAtom_angle_type)(i, m),
                      want.span(ATV::kAtom_angle_type)(i, m));
        }
        ASSERT_EQ(got.span(ATV::kAtom_num_dihedral)[i],
                  want.span(ATV::kAtom_num_dihedral)[i]);
        for (int m = 0; m < want.span(ATV::kAtom_num_dihedral)[i]; ++m) {
            for (int k = 0; k < 3; ++k)
                EXPECT_EQ(got.span(ATV::kAtom_dihedral_atom)(i, 3 * m + k),
                          want.span(ATV::kAtom_dihedral_atom)(i, 3 * m + k));
            EXPECT_EQ(got.span(ATV::kAtom_dihedral_type)(i, m),
                      want.span(ATV::kAtom_dihedral_type)(i, m));
        }
        for (int k = 0; k < 3; ++k)
            ASSERT_EQ(got.span(ATV::kAtom_nspecial)(i, k),
                      want.span(ATV::kAtom_nspecial)(i, k))
                << "nspecial tier " << k;
        for (int s = 0; s < want.span(ATV::kAtom_nspecial)(i, 2); ++s)
            EXPECT_EQ(got.span(ATV::kAtom_special)(i, s),
                      want.span(ATV::kAtom_special)(i, s))
                << "special slot " << s;
    }

    expect_box_equal(got, want);
    expect_meta_equal(got, want);
}

} // namespace

// ── (a) Multi-rank roundtrip: every field exact, invariants, root untouched ─
//
// Root builds the reference frame; non-root ranks start from Frame(0). After
// the broadcast EVERY rank (root included — which proves the root frame is
// untouched) must equal an independently constructed reference bit-for-bit.
// Also runs at N=1, where it degenerates to the root-untouched assertion.

TEST(MpiBcastFrame, RoundtripReplicatesEveryFieldExactly) {
    const cpu::Frame want = make_reference_frame();
    cpu::Frame frame =
        (world_rank() == 0) ? make_reference_frame() : cpu::Frame(0);

    cpu::comm::bcast_frame(frame, /*root=*/0);

    EXPECT_TRUE(frame.has_bonded());
    expect_frame_equal(frame, want);
}

// ── (b) World size 1: strict no-op degrade ──────────────────────────────────
//
// Only meaningful at N=1 (at N>1 non-root frames are legitimately rewritten);
// skipping is world-uniform, so lockstep is preserved.

TEST(MpiBcastFrame, SerialWorldDegradesToNoOp) {
    if (world_size() != 1)
        GTEST_SKIP() << "serial-degrade contract is a world-size-1 property";

    const cpu::Frame want = make_reference_frame();
    cpu::Frame frame = make_reference_frame();

    cpu::comm::bcast_frame(frame, /*root=*/0);

    expect_frame_equal(frame, want); // field-for-field identical to pre-call
}

// ── (c) 0-atom frame: header-only broadcast ─────────────────────────────────
//
// No payload rows; the header (box + scalars) must still land on non-root.

TEST(MpiBcastFrame, ZeroAtomFrameBroadcastsHeaderOnly) {
    cpu::Frame frame(0);
    if (world_rank() == 0) {
        frame.box = ATV::cpu::Box::orthorhombic(6.0, 7.5, 9.0);
        frame.meta.get(ATV::kEnergy) = -42.125;
        frame.meta.get(ATV::kKinetic) = 3.5;
    }

    EXPECT_NO_THROW(cpu::comm::bcast_frame(frame, /*root=*/0));

    EXPECT_EQ(frame.nlocal(), 0);
    EXPECT_EQ(frame.nghost(), 0);
    EXPECT_EQ(frame.box.cell()(0, 0), 6.0);
    EXPECT_EQ(frame.box.cell()(1, 1), 7.5);
    EXPECT_EQ(frame.box.cell()(2, 2), 9.0);
    EXPECT_EQ(frame.meta.get(ATV::kEnergy), -42.125);
    EXPECT_EQ(frame.meta.get(ATV::kKinetic), 3.5);
}

// ── (d) No-topology frame: replica stays atom_style atomic ─────────────────

TEST(MpiBcastFrame, NoTopologyFrameStaysAtomic) {
    constexpr int n = 4;
    cpu::Frame frame = (world_rank() == 0) ? cpu::Frame(n) : cpu::Frame(0);
    if (world_rank() == 0) {
        for (int i = 0; i < n; ++i) {
            frame.span(ATV::kAtom_x)[i] = StorageReal(1.0 + i);
            frame.span(ATV::kAtom_y)[i] = StorageReal(2.0 + i);
            frame.span(ATV::kAtom_z)[i] = StorageReal(3.0 + i);
            frame.span(ATV::kAtom_mass)[i] = StorageReal(4.0);
            frame.span(ATV::kAtom_tag)[i] = TagInt(20 + 3 * i);
        }
        frame.box = ATV::cpu::Box::cubic(10.0);
    }

    cpu::comm::bcast_frame(frame, /*root=*/0);

    EXPECT_EQ(frame.nlocal(), n);
    EXPECT_FALSE(frame.has_bonded());
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(frame.span(ATV::kAtom_x)[i], StorageReal(1.0 + i));
        EXPECT_EQ(frame.span(ATV::kAtom_tag)[i], TagInt(20 + 3 * i));
        EXPECT_EQ(frame.span(ATV::kAtom_num_bond)[i], 0);
    }
}

// ── (e-i) Exchanged extension column rides the roundtrip exactly ────────────
//
// Precondition of bcast_frame: exchanged extension-column registration must
// match on every rank. Here it does — the column's values must transit
// bit-exactly alongside the schema fields.

TEST(MpiBcastFrame, ExchangedExtensionColumnRidesTheRoundtrip) {
    constexpr int n = 3;
    cpu::Frame frame = (world_rank() == 0) ? cpu::Frame(n) : cpu::Frame(0);
    const ColKey<StorageReal> aux_key{"atoms", "aux", kDynamicFieldId,
                                      ColDType::Storage};
    frame.block("atoms").insert(
        "aux", cpu::ColumnSpec{ColDType::Storage, 2, ColGroup::Core, 0,
                               kColMigrate | kColExchange, kDynamicFieldId});
    if (world_rank() == 0) {
        auto aux = frame.span(aux_key);
        for (int i = 0; i < n; ++i) {
            frame.span(ATV::kAtom_x)[i] = StorageReal(0.5 * i);
            frame.span(ATV::kAtom_tag)[i] = TagInt(i + 1);
            aux(i, 0) = StorageReal(10.0 + i);
            aux(i, 1) = StorageReal(-20.0 - i);
        }
        frame.box = ATV::cpu::Box::cubic(8.0);
    }

    cpu::comm::bcast_frame(frame, /*root=*/0);

    ASSERT_EQ(frame.nlocal(), n);
    auto aux = frame.span(aux_key);
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(frame.span(ATV::kAtom_x)[i], StorageReal(0.5 * i));
        EXPECT_EQ(aux(i, 0), StorageReal(10.0 + i)) << "row " << i;
        EXPECT_EQ(aux(i, 1), StorageReal(-20.0 - i)) << "row " << i;
    }
}

// ── (e-ii) Mismatched exchanged registration fails collectively ──────
//
// ONLY non-root ranks register an extra exchanged column. The collective
// fingerprint gate makes every rank throw before the fixed header or payload.
// At N=1 there is no non-root rank and the call is a no-op.

TEST(MpiBcastFrame, MismatchedExchangedRegistrationThrowsOnNonRoot) {
    constexpr int n = 4;
    cpu::Frame frame = (world_rank() == 0) ? cpu::Frame(n) : cpu::Frame(0);
    if (world_rank() == 0) {
        for (int i = 0; i < n; ++i) {
            frame.span(ATV::kAtom_x)[i] = StorageReal(i);
            frame.span(ATV::kAtom_tag)[i] = TagInt(i);
        }
        frame.box = ATV::cpu::Box::cubic(5.0);
    } else {
        // The registration divergence under test: non-root only.
        frame.block("atoms").insert(
            "rogue",
            cpu::ColumnSpec{ColDType::Storage, 1, ColGroup::Core, 0,
                            kColMigrate | kColExchange, kDynamicFieldId});
    }

    if (world_size() == 1) {
        EXPECT_NO_THROW(cpu::comm::bcast_frame(frame, /*root=*/0));
    } else {
        EXPECT_THROW(cpu::comm::bcast_frame(frame, /*root=*/0),
                     std::runtime_error);
    }
}

TEST(MpiBcastFrame, TypedRuntimeMetadataPreservesExactDtypes) {
    constexpr MetaKey<bool> enabled{"runtime_enabled", true};
    constexpr MetaKey<std::uint64_t> epoch{"runtime_epoch", true};
    constexpr MetaKey<std::string> label{"runtime_label", true};
    constexpr MetaKey<MetaF32x3> origin{"runtime_origin", true};
    constexpr MetaKey<MetaI64x3> counters{"runtime_counters", true};

    cpu::Frame frame(0);
    frame.meta.insert(enabled, false);
    frame.meta.insert(epoch, std::uint64_t{0});
    frame.meta.insert(label, std::string{});
    frame.meta.insert(origin, MetaF32x3{});
    frame.meta.insert(counters, MetaI64x3{});
    if (world_rank() == 0) {
        frame.meta.set(enabled, true);
        frame.meta.set(epoch, std::uint64_t{9007199254740993ull});
        frame.meta.set(label, std::string{"native-meta"});
        frame.meta.set(origin, MetaF32x3{1.25f, -2.5f, 3.75f});
        frame.meta.set(counters, MetaI64x3{-7, 0, 9007199254740993LL});
    }

    cpu::comm::bcast_frame(frame, /*root=*/0);

    EXPECT_TRUE(frame.meta.get(enabled));
    EXPECT_EQ(frame.meta.get(epoch), std::uint64_t{9007199254740993ull});
    EXPECT_EQ(frame.meta.get(label), "native-meta");
    EXPECT_EQ(frame.meta.get(origin), (MetaF32x3{1.25f, -2.5f, 3.75f}));
    EXPECT_EQ(frame.meta.get(counters), (MetaI64x3{-7, 0, 9007199254740993LL}));
}

// ── Custom main that initializes MPI before/after GoogleTest ────────────
//
// See test_topology.cpp for the OpenMPI 5.0.x macOS teardown rationale.

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
