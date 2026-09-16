/// @file tests/cpu/communication/test_atom_pack_generic_wire.cpp
/// @brief The generic (row-major) arm of each wire codec, in-process.
///
/// `comm-maint-03-genericpath` set out to delete these arms as dead. They are
/// not: a plugin can put an extra column on any wire with a single
/// `Block::insert`, and the generic arm is what carries it. The only thing
/// exercising that was `tests/mpi/test_halo_exchange.cpp`, so removing an arm
/// surfaced as an MPI **timeout** — a rank throwing while its partners waited
/// on a receive — rather than as a failure naming the codec.
///
/// These tests close that gap. Delete a generic arm and one of them fails
/// directly, serially, with the codec in the message.
///
/// Surface pinned (no MPI, no halo, no Driver):
/// @code{.cpp}
///   pack / unpack   with a multi-component plugin column on the wire
///   pack_rows / accumulate                       with a plugin accumulator
///   pack / unpack   field-major whenever every column is scalar
/// @endcode
///
/// Categories: domain (round trip preserves values), edge (writability).

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/communication/atom_pack.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// @brief Contract for a wire that carries the base set plus a plugin column —
/// the shape that routes both ends down the generic (row-major) arm.
/// The engine's base declaration plus one plugin column, declared the way
/// the pass under test needs it. `PackKind` picks the pass; the contract
/// itself never names one.
ATV::GhostContract wire_with(std::string_view extra, ColDType dtype, int cols,
                             cpu::comm::PackKind kind) {
    ATV::GhostContract c;
    c.declare(ATV::base_fields::kEngine);
    const ATV::GhostField f =
        kind == cpu::comm::PackKind::Reverse
            ? ATV::GhostField::reduce_to_owner(extra, dtype, cols, "plugin")
        : kind == cpu::comm::PackKind::Comm
            ? ATV::GhostField::read_each_step(extra, dtype, cols, "plugin")
            : ATV::GhostField::read_stable(extra, dtype, cols, "plugin");
    c.declare(std::span<const ATV::GhostField>{&f, 1});
    return c;
}

/// @brief A two-component plugin column on the comm wire — the shape that
/// makes `plan_is_all_scalar` false and routes pack/unpack down the generic
/// row-major arm.
const ColKey<StorageReal> kAux{"atoms", "aux", kDynamicFieldId,
                               ColDType::Storage};

/// @brief A two-component plugin column on the reverse wire.
const ColKey<AccReal> kAuxAcc{"atoms", "aux_acc", kDynamicFieldId,
                              ColDType::Acc};

cpu::Frame frame_with_comm_aux(int n) {
    cpu::Frame f(n);
    f.block("atoms").insert(
        "aux", cpu::ColumnSpec{ColDType::Storage, 2, ColGroup::Core, 0,
                               kColMigrate | kColExchange, kDynamicFieldId});
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = static_cast<StorageReal>(i);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(10 * i);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(100 * i);
        f.span(kAux)(i, 0) = static_cast<StorageReal>(1000 + i);
        f.span(kAux)(i, 1) = static_cast<StorageReal>(-2000 - i);
    }
    return f;
}

} // namespace

// ── comm wire, generic arm ────────────────────────────────────────────

/// A multi-component plugin column on the refresh wire makes the plan
/// non-canonical, so both ends take the row-major arm. The round trip must
/// preserve every value, including the PBC shift on the coordinate columns
/// and NO shift on the plugin column.
TEST(TestAtomPackGenericWire, CommRoundTripCarriesAMultiComponentColumn) {
    cpu::Frame f = frame_with_comm_aux(6);
    cpu::comm::PackPlan plan = cpu::comm::make_wire_plan(
        f, cpu::comm::PackKind::Comm,
        wire_with("aux", ColDType::Storage, 2, cpu::comm::PackKind::Comm));
    // The plugin column is present, so this is NOT the canonical x/y/z plan.
    ASSERT_GT(plan.fields.size(), 3u);

    const int send[2] = {1, 2};
    const std::size_t bytes = plan.bytes(2);
    std::vector<std::byte> wire(bytes);
    const double shift[3] = {5.0, 0.0, 0.0};
    plan.pack(f, send, 2, shift, wire.data(), bytes);

    // Unpack into rows 4-5, which currently hold their own distinct values.
    plan.unpack(f, 2, /*first=*/4, wire.data(), bytes);

    for (int k = 0; k < 2; ++k) {
        SCOPED_TRACE("row " + std::to_string(k));
        const int src = send[k];
        const int dst = 4 + k;
        // Coordinates carry the image shift on x only.
        EXPECT_DOUBLE_EQ(f.span(kAtom_x)[dst],
                         static_cast<StorageReal>(src) + shift[0]);
        EXPECT_DOUBLE_EQ(f.span(kAtom_y)[dst],
                         static_cast<StorageReal>(10 * src));
        EXPECT_DOUBLE_EQ(f.span(kAtom_z)[dst],
                         static_cast<StorageReal>(100 * src));
        // The plugin column travels verbatim — both components, no shift.
        EXPECT_DOUBLE_EQ(f.span(kAux)(dst, 0),
                         static_cast<StorageReal>(1000 + src));
        EXPECT_DOUBLE_EQ(f.span(kAux)(dst, 1),
                         static_cast<StorageReal>(-2000 - src));
    }
}

// ── reverse wire, generic arm ─────────────────────────────────────────

/// A plugin column on the reverse wire routes the reverse codec down its
/// generic arm; the accumulate side must still be `+=` on every component.
TEST(TestAtomPackGenericWire, ReverseRoundTripCarriesAPluginAccumulator) {
    cpu::Frame f(6);
    f.block("atoms").insert("aux_acc",
                            cpu::ColumnSpec{ColDType::Acc, 2, ColGroup::Core, 0,
                                            kColMigrate | kColExchange,
                                            kDynamicFieldId});
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_fx)[i] = static_cast<AccReal>(i);
        f.span(kAtom_fy)[i] = static_cast<AccReal>(0);
        f.span(kAtom_fz)[i] = static_cast<AccReal>(0);
        f.span(kAuxAcc)(i, 0) = static_cast<AccReal>(7 * i);
        f.span(kAuxAcc)(i, 1) = static_cast<AccReal>(-3 * i);
    }
    cpu::comm::PackPlan plan = cpu::comm::make_wire_plan(
        f, cpu::comm::PackKind::Reverse,
        wire_with("aux_acc", ColDType::Acc, 2, cpu::comm::PackKind::Reverse));
    ASSERT_GT(plan.fields.size(), 3u);

    std::vector<std::byte> wire;
    plan.pack_rows(/*first=*/4, /*count=*/2, wire);
    ASSERT_EQ(wire.size(), plan.bytes(2));

    // Zero the owners so the fold is readable, then fold rows 4,5 onto 0,1.
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_fx)[i] = 0.0;
        f.span(kAuxAcc)(i, 0) = 0.0;
        f.span(kAuxAcc)(i, 1) = 0.0;
    }
    plan.accumulate(std::vector<int>{0, 1}, wire);

    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[0], 4.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[1], 5.0);
    EXPECT_DOUBLE_EQ(f.span(kAuxAcc)(0, 0), 7.0 * 4);
    EXPECT_DOUBLE_EQ(f.span(kAuxAcc)(0, 1), -3.0 * 4);
    EXPECT_DOUBLE_EQ(f.span(kAuxAcc)(1, 0), 7.0 * 5);
    EXPECT_DOUBLE_EQ(f.span(kAuxAcc)(1, 1), -3.0 * 5);
}

// ── border wire has exactly one layout ────────────────────────────────

TEST(TestAtomPackGenericWire, BorderAllScalarRoundTripIsFieldMajor) {
    cpu::Frame f(6);
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_x)[i] = static_cast<StorageReal>(i);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(10 * i);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(100 * i);
        f.span(kAtom_tag)[i] = static_cast<TagInt>(500 + i);
        f.span(kAtom_type)[i] = i;
    }
    cpu::comm::PackPlan plan = cpu::comm::make_wire_plan(
        f, cpu::comm::PackKind::Border, ATV::base_fields::engine_contract());
    const int send[2] = {1, 3};
    const std::size_t bytes = plan.bytes(2);
    std::vector<std::byte> wire(bytes);
    const double shift[3] = {5.0, 0.0, 0.0};
    plan.pack(f, send, 2, shift, wire.data(), bytes);
    plan.unpack(f, 2, /*first=*/4, wire.data(), bytes);
    EXPECT_DOUBLE_EQ(f.span(kAtom_x)[4], 1.0 + shift[0]);
    EXPECT_DOUBLE_EQ(f.span(kAtom_y)[4], 10.0);
    EXPECT_EQ(f.span(kAtom_tag)[4], static_cast<TagInt>(501));
    EXPECT_EQ(f.span(kAtom_type)[5], 3);
}

/// A 3-component `image` on the border wire forces the row-major arm.
/// Round-tripping pins that layout against the scalar field-major arm.
TEST(TestAtomPackGenericWire, BorderRoundTripIncludesTheImageTriple) {
    cpu::Frame f(6);
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_x)[i] = static_cast<StorageReal>(i);
        f.span(kAtom_tag)[i] = static_cast<TagInt>(500 + i);
        f.span(kAtom_type)[i] = i;
        f.span(kAtom_charge)[i] = static_cast<StorageReal>(0.25 * i);
        f.span(kAtom_image)(i, 0) = static_cast<std::int8_t>(i);
        f.span(kAtom_image)(i, 1) = static_cast<std::int8_t>(-i);
        f.span(kAtom_image)(i, 2) = static_cast<std::int8_t>(2 * i);
    }
    cpu::comm::PackPlan plan =
        cpu::comm::make_wire_plan(f, cpu::comm::PackKind::Border, [] {
            // Everything this test asserts survives the round trip, plus
            // `image` — the three-component column that makes the border
            // wire row-major.
            ATV::GhostContract b;
            b.declare(ATV::base_fields::kEngine);
            constexpr std::array plugin = {
                ATV::GhostField::read_stable(kAtom_charge, "test"),
                ATV::GhostField::read_stable("image", ColDType::Int8, 3,
                                             "test"),
            };
            b.declare(plugin);
            return b;
        }());

    const int send[2] = {1, 3};
    const std::size_t bytes = plan.bytes(2);
    std::vector<std::byte> wire(bytes);
    const double no_shift[3] = {0.0, 0.0, 0.0};
    plan.pack(f, send, 2, no_shift, wire.data(), bytes);
    plan.unpack(f, 2, /*first=*/4, wire.data(), bytes);

    for (int k = 0; k < 2; ++k) {
        SCOPED_TRACE("row " + std::to_string(k));
        const int src = send[k];
        const int dst = 4 + k;
        EXPECT_EQ(f.span(kAtom_tag)[dst], static_cast<TagInt>(500 + src));
        EXPECT_EQ(f.span(kAtom_type)[dst], src);
        EXPECT_DOUBLE_EQ(f.span(kAtom_charge)[dst],
                         static_cast<StorageReal>(0.25 * src));
        // All three image components survive — the reason border is row-major.
        EXPECT_EQ(f.span(kAtom_image)(dst, 0), static_cast<std::int8_t>(src));
        EXPECT_EQ(f.span(kAtom_image)(dst, 1), static_cast<std::int8_t>(-src));
        EXPECT_EQ(f.span(kAtom_image)(dst, 2),
                  static_cast<std::int8_t>(2 * src));
    }
}

/// A scalar Storage plugin column whose `wire_id` is *numerically* `x` (0)
/// must still travel unshifted: PBC displacement is `kColPbcShift`, not
/// `wire_id <= z`. The old comparison would have added a box length here.
TEST(TestAtomPackGenericWire, PluginScalarIsNotPbcShiftedByWireId) {
    cpu::Frame f(4);
    f.block("atoms").insert("qaux", cpu::ColumnSpec{ColDType::Storage, 1,
                                                    ColGroup::Core, 0,
                                                    kColMigrate | kColExchange,
                                                    /*wire_id=*/0});
    const ColKey<StorageReal> kQaux{"atoms", "qaux", /*wire_id=*/0,
                                    ColDType::Storage};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = static_cast<StorageReal>(i);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(0);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(0);
        f.span(kQaux)[i] = static_cast<StorageReal>(40 + i);
    }
    cpu::comm::PackPlan plan = cpu::comm::make_wire_plan(
        f, cpu::comm::PackKind::Comm,
        wire_with("qaux", ColDType::Storage, 1, cpu::comm::PackKind::Comm));

    const int send[1] = {1};
    const std::size_t bytes = plan.bytes(1);
    std::vector<std::byte> wire(bytes);
    const double shift[3] = {5.0, 0.0, 0.0};
    plan.pack(f, send, 1, shift, wire.data(), bytes);
    plan.unpack(f, 1, /*first=*/3, wire.data(), bytes);

    EXPECT_DOUBLE_EQ(f.span(kAtom_x)[3], static_cast<StorageReal>(1 + 5));
    EXPECT_DOUBLE_EQ(f.span(kQaux)[3], static_cast<StorageReal>(41));
}

/// The contract's cost model in one assertion. A `step`-freshness read is
/// what a style like DPD uses to get a current neighbour velocity, and the
/// wire it compiles onto must not degrade because it grew past x/y/z: any
/// number of SCALAR columns stays field-major, one sequential stream each. Pinned as a byte layout, not a round trip — a
/// round trip passes under either arm, which is how the two sides drifted
/// apart before.
TEST(TestAtomPackGenericWire, ScalarRefreshWireStaysFieldMajorBeyondXyz) {
    cpu::Frame f(4);
    f.block("atoms").insert(
        "vaux", cpu::ColumnSpec{ColDType::Storage, 1, ColGroup::Core, 0,
                                kColMigrate | kColExchange, kDynamicFieldId});
    const ColKey<StorageReal> kVaux{"atoms", "vaux", kDynamicFieldId,
                                    ColDType::Storage};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = static_cast<StorageReal>(i);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(10 + i);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(20 + i);
        f.span(kVaux)[i] = static_cast<StorageReal>(70 + i);
    }
    cpu::comm::PackPlan plan = cpu::comm::make_wire_plan(
        f, cpu::comm::PackKind::Comm,
        wire_with("vaux", ColDType::Storage, 1, cpu::comm::PackKind::Comm));
    ASSERT_EQ(plan.fields.size(), 4u);

    const int send[2] = {1, 2};
    const std::size_t bytes = plan.bytes(2);
    std::vector<std::byte> wire(bytes);
    const double no_shift[3] = {0.0, 0.0, 0.0};
    plan.pack(f, send, 2, no_shift, wire.data(), bytes);

    std::vector<StorageReal> got(8);
    std::memcpy(got.data(), wire.data(), bytes);
    // Field-major: all x, then all y, then all z, then all vaux — NOT
    // {x,y,z,vaux} per row.
    EXPECT_EQ(got, (std::vector<StorageReal>{1, 2, 11, 12, 21, 22, 71, 72}));

    plan.unpack(f, 2, /*first=*/2, wire.data(), bytes);
    EXPECT_DOUBLE_EQ(f.span(kVaux)[2], static_cast<StorageReal>(71));
    EXPECT_DOUBLE_EQ(f.span(kAtom_z)[3], static_cast<StorageReal>(22));
}

// ── edge: writability is a precondition of both arms ──────────────────

/// A plan compiled from a const Frame cannot be unpacked into, and that is
/// refused up front rather than by routing to a different layout. The
/// asymmetric `&& plan.writable` term that used to live inside the fast-path
/// condition is what made a conforming non-writable plan pack field-major and
/// unpack row-major.
TEST(TestAtomPackGenericWire, UnpackRefusesAConstPlan) {
    cpu::Frame f(4);
    const cpu::Frame& cf = f;
    const ATV::GhostContract fwd = ATV::base_fields::engine_contract();
    const cpu::comm::PackPlan ro =
        cpu::comm::make_wire_plan(cf, cpu::comm::PackKind::Comm, fwd);
    ASSERT_FALSE(ro.writable);

    const std::size_t bytes = ro.bytes(1);
    std::vector<std::byte> wire(bytes);
    EXPECT_THROW(ro.unpack(f, 1, 0, wire.data(), bytes), std::exception);
}
