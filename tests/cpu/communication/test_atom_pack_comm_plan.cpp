/// @file tests/cpu/communication/test_atom_pack_comm_plan.cpp
/// @brief A wire plan carries exactly what the Schema names, in Schema order.
///
/// This file used to assert the opposite kind of fact: that `kColComm`
/// resolves to x/y/z and that the border plan "still carries ghost velocity
/// and charge". Those were properties of a **global table**, decided once for
/// every run there would ever be — and they were how the border wire came to
/// ship 24 B of velocity that only `pair_dpd` reads and 8 B of mass that
/// nothing reads.
///
/// Wire membership is now a property of the **mounted component set**
/// (`comm-maint-09-schema`), so the thing worth pinning here is the
/// resolution contract — Schema in, ColumnRefs out, same fields, same order.
/// Who declares what is pinned in `tests/cpu/frame/test_schema.cpp`.
///
/// Surface pinned (no MPI, no halo, no Driver):
/// @code{.cpp}
///   PackPlan make_wire_plan(Frame&, PackKind, const Schema&);
///   std::size_t PackPlan::bytes(int n);
/// @endcode
///
/// Categories: basics (resolution), domain (order is wire order), edge
/// (a field the Frame lacks).

#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/schema.h"
#include "atomiverse/cpu/communication/atom_pack.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

std::vector<std::string_view> keys_of(const cpu::comm::PackPlan& plan) {
    std::vector<std::string_view> out;
    for (const cpu::ColumnRef& f : plan.fields)
        out.push_back(f.key);
    return out;
}

} // namespace

// ── 1. the plan is the schema, resolved ───────────────────────────────

/// Every declared field, and nothing else.
TEST(TestWirePlan, PlanCarriesExactlyTheSchema) {
    cpu::Frame frame(4);
    Schema s;
    s.require("x", ColDType::Storage, 1, "test");
    s.require("charge", ColDType::Storage, 1, "test");

    const cpu::comm::PackPlan plan =
        cpu::comm::make_wire_plan(frame, cpu::comm::PackKind::Border, s);
    EXPECT_EQ(keys_of(plan), (std::vector<std::string_view>{"x", "charge"}));
}

/// Declaration order IS wire order. Two ranks must agree on it, so it cannot
/// be re-sorted into table order behind the caller's back.
TEST(TestWirePlan, PlanPreservesSchemaOrder) {
    cpu::Frame frame(4);
    Schema forwards;
    forwards.require_all({"x", "y", "z"}, ColDType::Storage, "test");
    Schema backwards;
    backwards.require_all({"z", "y", "x"}, ColDType::Storage, "test");

    EXPECT_EQ(keys_of(cpu::comm::make_wire_plan(
                  frame, cpu::comm::PackKind::Comm, forwards)),
              (std::vector<std::string_view>{"x", "y", "z"}));
    EXPECT_EQ(keys_of(cpu::comm::make_wire_plan(
                  frame, cpu::comm::PackKind::Comm, backwards)),
              (std::vector<std::string_view>{"z", "y", "x"}));
}

/// The engine's base forward wire is the three coordinates — LAMMPS'
/// `comm_x_only`, and 24 B per ghost at the default double storage tier.
TEST(TestWirePlan, BaseForwardWireIsThreeCoordinates) {
    cpu::Frame frame(4);
    Schema s;
    base_fields::forward(s);

    const cpu::comm::PackPlan plan =
        cpu::comm::make_wire_plan(frame, cpu::comm::PackKind::Comm, s);
    EXPECT_EQ(plan.fields.size(), 3u);
    EXPECT_EQ(plan.bytes(1), 3 * sizeof(StorageReal));
    EXPECT_EQ(plan.bytes(8), 8 * 3 * sizeof(StorageReal));
}

/// Adding a declarer widens the wire, and by exactly that field. This is the
/// property the old global table could not express at all.
TEST(TestWirePlan, DeclaringMoreWidensTheWire) {
    cpu::Frame frame(4);
    Schema lean;
    base_fields::border(lean);
    Schema fat = lean;
    fat.require(kAtom_charge, "CoulLong");

    const auto lean_plan =
        cpu::comm::make_wire_plan(frame, cpu::comm::PackKind::Border, lean);
    const auto fat_plan =
        cpu::comm::make_wire_plan(frame, cpu::comm::PackKind::Border, fat);

    EXPECT_EQ(fat_plan.fields.size(), lean_plan.fields.size() + 1);
    EXPECT_EQ(fat_plan.bytes(1), lean_plan.bytes(1) + sizeof(StorageReal));
}

// ── 2. edge ───────────────────────────────────────────────────────────

/// A field the Frame does not carry is refused here too, naming the field and
/// its declarer. `check_schema` catches this at build(); reaching the wire
/// with a short Frame means the Frame changed shape afterwards.
TEST(TestWirePlan, AFieldTheFrameLacksIsRefused) {
    cpu::Frame frame(4);
    Schema s;
    s.require("x", ColDType::Storage, 1, "test");
    s.require("not_a_column", ColDType::Storage, 1, "GreedyComponent");

    try {
        (void)cpu::comm::make_wire_plan(frame, cpu::comm::PackKind::Border, s);
        FAIL() << "a wire naming a missing column must be refused";
    } catch (const std::exception& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("not_a_column"), std::string::npos);
        EXPECT_NE(what.find("GreedyComponent"), std::string::npos);
    }
}

/// An empty wire is a build-time mistake, not a run-time state.
TEST(TestWirePlan, AnEmptySchemaIsRefused) {
    cpu::Frame frame(4);
    const Schema empty;
    EXPECT_THROW((void)cpu::comm::make_wire_plan(
                     frame, cpu::comm::PackKind::Comm, empty),
                 std::exception);
}
