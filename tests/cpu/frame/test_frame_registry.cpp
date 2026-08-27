/// @file tests/cpu/frame/test_frame_registry.cpp
/// @brief Unit tests for the frame-block registry (frame-block-01): schema
///        completeness, the typed mapping API (span/has/insert), registry-
///        owned extension columns riding grow/copy_row/pack, the typed meta
///        region, and the launch-boundary resolution discipline.

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/role.h"
#include "atomiverse/cpu/communication/atom_pack.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

// ── Schema completeness (ac-001) ────────────────────────────────────────────

TEST(FrameRegistry, BindingVisitsEverySchemaRowExactlyOnce) {
    cpu::Frame frame(4);
    frame.reserve_topology(2, 1, 1, 6);
    std::set<std::string> keys;
    for (const auto& [key, column] : frame.block("atoms").columns()) {
        (void)column;
        EXPECT_TRUE(keys.insert(key).second) << "duplicate schema key " << key;
    }
    EXPECT_EQ(static_cast<int>(frame.block("atoms").columns().size()),
              kAtomFieldCount);
    EXPECT_EQ(static_cast<int>(keys.size()), kAtomFieldCount);
    for (const FieldSpec& spec : kAtomFieldSpecs)
        EXPECT_EQ(keys.count(spec.key), 1u)
            << "missing schema key " << spec.key;
}

TEST(FrameRegistry, ColumnWidthsMatchTheBoundStorage) {
    cpu::Frame frame(4);
    frame.reserve_topology(2, 1, 1, 6);
    const auto refs = frame.block("atoms").resolve_columns();
    ASSERT_EQ(static_cast<int>(refs.size()), kAtomFieldCount);
    for (const auto& ref : refs) {
        int expected = 1;
        if (ref.key == "image" || ref.key == "nspecial")
            expected = 3;
        else if (ref.key == "bond_atom" || ref.key == "bond_type")
            expected = 2;
        else if (ref.key == "angle_atom")
            expected = 2;
        else if (ref.key == "angle_type")
            expected = 1;
        else if (ref.key == "dihedral_atom")
            expected = 3;
        else if (ref.key == "dihedral_type")
            expected = 1;
        else if (ref.key == "improper_atom")
            expected = 0; // reserve 0 improper
        else if (ref.key == "improper_type")
            expected = 0;
        else if (ref.key == "special")
            expected = 6;
        EXPECT_EQ(ref.cols, expected) << ref.key;
    }
}

// ── Typed spans (ac-002) ────────────────────────────────────────────────────

TEST(FrameRegistry, SpanResolvesTypedColumns) {
    cpu::Frame frame(3);
    auto x = frame.span(kAtom_x);
    x[1] = StorageReal(4.5);
    EXPECT_EQ(frame.span(ATV::kAtom_x)[1], StorageReal(4.5)); // same storage

    auto img = frame.span(kAtom_image);
    EXPECT_EQ(img.cols(), 3);
    img(2, 1) = 7;
    EXPECT_EQ(frame.span(ATV::kAtom_image)(2, 1), 7);
}

TEST(FrameRegistry, SpanRejectsUnknownKeyAndWrongType) {
    cpu::Frame frame(2);
    EXPECT_THROW(frame.span(ColKey<StorageReal>{
                     "atoms", "nope", kDynamicFieldId, ColDType::Storage}),
                 cpu::column_error);
    EXPECT_THROW(
        frame.span(ColKey<int>{"atoms", "x", kDynamicFieldId, ColDType::Int}),
        cpu::column_error);
    EXPECT_THROW(frame.span(ColKey<StorageReal>{"bonds", "x", kDynamicFieldId,
                                                ColDType::Storage}),
                 cpu::column_error);
}

TEST(FrameRegistry, HasTracksTheLazyTopologyGroup) {
    cpu::Frame frame(3);
    EXPECT_TRUE(frame.has("atoms", "x"));
    EXPECT_TRUE(frame.has("atoms", "num_bond")); // counts always present
    EXPECT_TRUE(frame.has("atoms", "bond_atom"));
    EXPECT_EQ(frame.span(kAtom_bond_atom).cols(), 0); // initially zero-width
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, 3);
    frame.add_bond(0, 1, 1);
    EXPECT_TRUE(frame.has("atoms", "bond_atom"));
    EXPECT_FALSE(frame.has("atoms", "rho"));
    EXPECT_FALSE(frame.has("bonds", "x"));
}

// ── Extension columns (ac-002) ──────────────────────────────────────────────

TEST(FrameRegistry, ExtensionColumnLivesTheFullLifecycle) {
    cpu::Frame frame(3);
    frame.block("atoms").insert(
        "rho", cpu::ColumnSpec{ColDType::Storage, 1, ColGroup::Core, 0,
                               kColMigrate | kColExchange, kDynamicFieldId});
    EXPECT_TRUE(frame.has("atoms", "rho"));

    const ColKey<StorageReal> rho_key{"atoms", "rho", kDynamicFieldId,
                                      ColDType::Storage};
    auto rho = frame.span(rho_key);
    rho[0] = 1.5;
    rho[2] = 2.5;

    // Survives a chunked reallocation (the probe-4 bug class)…
    frame.grow(frame.capacity() + 1);
    rho = frame.span(rho_key);
    EXPECT_EQ(rho[0], StorageReal(1.5));
    EXPECT_EQ(rho[2], StorageReal(2.5));
    EXPECT_EQ(rho[frame.capacity() - 1], StorageReal(0)); // zeroed tail

    // …and rides copy_row.
    frame.copy_row(1, 2);
    EXPECT_EQ(rho[1], StorageReal(2.5));
}

TEST(FrameRegistry, InsertRejectsCollisions) {
    cpu::Frame frame(2);
    const cpu::ColumnSpec scalar{
        ColDType::Storage, 1, ColGroup::Core, 0, kColNone, kDynamicFieldId};
    EXPECT_THROW(frame.block("atoms").insert("x", scalar), cpu::column_error);
    frame.block("atoms").insert("rho", scalar);
    EXPECT_THROW(frame.block("atoms").insert("rho", scalar), cpu::column_error);
    EXPECT_THROW(
        frame.block("atoms").insert(
            "bad", cpu::ColumnSpec{ColDType::Storage, 0, ColGroup::Core, 0,
                                   kColNone, kDynamicFieldId}),
        std::invalid_argument);
}

TEST(FrameRegistry, ExchangedExtensionColumnRidesThePayload) {
    cpu::Frame a(2);
    const ColKey<StorageReal> rho_key{"atoms", "rho", kDynamicFieldId,
                                      ColDType::Storage};
    const cpu::ColumnSpec rho_spec{
        ColDType::Storage, 2, ColGroup::Core, 0, kColMigrate | kColExchange,
        kDynamicFieldId};
    a.block("atoms").insert("rho", rho_spec);
    auto rho = a.span(rho_key);
    rho(1, 0) = 3.5;
    rho(1, 1) = 4.5;
    a.span(ATV::kAtom_x)[1] = 9.0;

    cpu::Frame b(2); // same extras layout on the receiving side (SPMD)
    b.block("atoms").insert("rho", rho_spec);

    auto send_plan = cpu::comm::make_exchange_plan(a);
    auto recv_plan = cpu::comm::make_exchange_plan(b);
    std::vector<std::byte> buf(send_plan.max_row_bytes());
    const auto packed = send_plan.pack_row(a, 1, buf.data(), buf.size());
    const auto unpacked = recv_plan.unpack_row(b, 0, buf.data(), packed);
    EXPECT_EQ(packed, unpacked);
    EXPECT_LE(packed, static_cast<int>(buf.size()));

    EXPECT_EQ(b.span(ATV::kAtom_x)[0], StorageReal(9.0));
    auto rho_b = b.span(rho_key);
    EXPECT_EQ(rho_b(0, 0), StorageReal(3.5));
    EXPECT_EQ(rho_b(0, 1), StorageReal(4.5));
}

// ── Typed meta (ac-002) ─────────────────────────────────────────────────────

TEST(FrameRegistry, MetaViewsAliasTheLiveBookkeeping) {
    cpu::Frame frame(5);
    EXPECT_EQ(frame.meta.get(kNLocal), 5);
    frame.set_nlocal(3);
    EXPECT_EQ(frame.meta.get(kNLocal), 3);

    frame.meta.get(ATV::kEnergy) = -1.25;
    EXPECT_EQ(frame.meta.get(kEnergy), -1.25);
    frame.meta.set(kEnergy, -2.5);
    EXPECT_EQ(frame.meta.get(ATV::kEnergy), -2.5);
}

TEST(FrameRegistry, MetaOwnedEntriesAndErrors) {
    cpu::Frame frame(1);
    constexpr MetaKey<std::string> note{"note", true};
    constexpr MetaKey<std::string> absent{"absent", true};
    EXPECT_FALSE(frame.meta.contains(note.key));
    frame.meta.insert(note, std::string("equilibrated"));
    EXPECT_TRUE(frame.meta.contains(note.key));
    EXPECT_EQ(frame.meta.get(note), "equilibrated");
    EXPECT_TRUE(frame.meta.contains(kNLocal.key));
    EXPECT_THROW(frame.meta.get(absent), cpu::meta_error);
}

// ── Resolution discipline (ac-003) ──────────────────────────────────────────

TEST(FrameRegistry, ResolutionWindowCountsAndThrows) {
    cpu::Frame frame(2);
    frame.begin_resolution_window(4);
    (void)frame.span(kAtom_x);
    (void)frame.span(kAtom_vx);
    (void)frame.has("atoms", "x");
    EXPECT_EQ(frame.resolution_count(), 3);
    EXPECT_NO_THROW(frame.end_resolution_window("ok_component"));

    frame.begin_resolution_window(2);
    for (int i = 0; i < 5; ++i)
        (void)frame.span(kAtom_x);
    EXPECT_THROW(frame.end_resolution_window("greedy_component"),
                 std::logic_error);
}

namespace {

/// @brief A component that breaks the 13.7× rule on purpose: resolves a key
///        per "atom" inside its loop. Kind is Potential so Evaluate accepts it.
class KeyLookupInALoop : public cpu::Potential {
  public:
    std::string name() const override { return "key_lookup_in_a_loop"; }
    void launch(const cpu::LaunchContext& ctx) override {
        double acc = 0.0;
        for (int i = 0; i < 100; ++i) {
            auto x = ctx.frame.span(kAtom_x); // WRONG: inside the loop
            acc += x[0];
        }
        ctx.frame.meta.get(ATV::kEnergy) = acc;
    }
};

} // namespace

TEST(FrameRegistry, DriverTripsTheViolatingComponent) {
    cpu::Frame frame(4);
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add(KeyLookupInALoop{});
    eval.build();
    EXPECT_THROW(eval.run(frame), std::logic_error);
}

// ── Generation-aware stale handles (frame-block-04 ac-010) ─────────────────

TEST(FrameRegistry, SpanStalesAfterCapacityGrow) {
    cpu::Frame frame(2);
    auto x = frame.span(kAtom_x);
    x[0] = 1.0;
    EXPECT_FALSE(x.stale());
    frame.grow(frame.capacity() + 8);
    EXPECT_TRUE(x.stale());
    EXPECT_THROW(x[0], cpu::stale_handle_error);
    // Fresh resolution is valid again.
    auto x2 = frame.span(kAtom_x);
    EXPECT_FALSE(x2.stale());
    EXPECT_EQ(x2[0], StorageReal(1.0));
}

TEST(FrameRegistry, SpanStalesAfterRuntimeColumnInsert) {
    cpu::Frame frame(3);
    auto x = frame.span(kAtom_x);
    frame.block("atoms").insert(
        "rho", cpu::ColumnSpec{ColDType::Storage, 1, ColGroup::Core, 0,
                               kColNone, kDynamicFieldId});
    EXPECT_TRUE(x.stale());
    EXPECT_THROW(static_cast<void>(x.data()), cpu::stale_handle_error);
}

TEST(FrameRegistry, SpanStalesAfterTopologyRestride) {
    cpu::Frame frame(2);
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, 2);
    auto bond_atom = frame.span(kAtom_bond_atom);
    EXPECT_EQ(bond_atom.cols(), 0);
    frame.add_bond(0, 1, 0); // grows topology column width
    EXPECT_TRUE(bond_atom.stale());
    EXPECT_THROW(bond_atom(0, 0), cpu::stale_handle_error);
    auto bond2 = frame.span(kAtom_bond_atom);
    EXPECT_FALSE(bond2.stale());
    EXPECT_GE(bond2.cols(), 1);
}

TEST(FrameRegistry, MetaRefStalesAfterMetaInsert) {
    cpu::Frame frame(2);
    auto e = frame.meta.ref(kEnergy);
    e.get() = 1.25;
    EXPECT_FALSE(e.stale());
    constexpr MetaKey<double> kExtra{"test_extra_meta", true};
    frame.meta.insert(kExtra, 0.0);
    EXPECT_TRUE(e.stale());
    EXPECT_THROW(e.get(), cpu::stale_handle_error);
    auto e2 = frame.meta.ref(kEnergy);
    EXPECT_FALSE(e2.stale());
    EXPECT_EQ(e2.get(), 1.25);
}
