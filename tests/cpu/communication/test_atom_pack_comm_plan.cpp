/// @file tests/cpu/communication/test_atom_pack_comm_plan.cpp
/// @brief A wire plan carries exactly what the contract puts on that pass,
///        in declaration order.
///
/// This file used to assert the opposite kind of fact: that `kColComm`
/// resolves to x/y/z and that the border plan "still carries ghost velocity
/// and charge". Those were properties of a **global table**, decided once for
/// every run there would ever be — and they were how the border wire came to
/// ship 24 B of velocity that only `pair_dpd` reads and 8 B of mass that
/// nothing reads.
///
/// Wire membership is now a property of the **mounted component set**, so the
/// thing worth pinning here is the compile step — contract in, ColumnRefs
/// out, filtered by pass, same order. Who declares what is pinned in
/// `tests/cpu/frame/test_ghost_contract.cpp`.
///
/// Surface pinned (no MPI, no halo, no Driver):
/// @code{.cpp}
///   PackPlan make_wire_plan(Frame&, PackKind, const GhostContract&);
///   std::size_t PackPlan::bytes(int n);
/// @endcode
///
/// Categories: basics (resolution), domain (order is wire order), edge
/// (a field the Frame lacks).

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/core/precision.h"
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

/// Every declared field the pass carries, and nothing else.
TEST(TestWirePlan, PlanCarriesExactlyWhatThePassSelects) {
    cpu::Frame frame(4);
    constexpr std::array table = {
        GhostField::read_each_step("x", ColDType::Storage, 1, "test"),
        GhostField::read_stable("charge", ColDType::Storage, 1, "test"),
    };
    GhostContract c;
    c.declare(table);

    // Border carries every read; refresh carries only the `step` one.
    EXPECT_EQ(keys_of(cpu::comm::make_wire_plan(
                  frame, cpu::comm::PackKind::Border, c)),
              (std::vector<std::string_view>{"x", "charge"}));
    EXPECT_EQ(
        keys_of(cpu::comm::make_wire_plan(frame, cpu::comm::PackKind::Comm, c)),
        (std::vector<std::string_view>{"x"}));
}

/// Declaration order IS wire order. Two ranks must agree on it, so it cannot
/// be re-sorted into table order behind the caller's back.
TEST(TestWirePlan, PlanPreservesDeclarationOrder) {
    cpu::Frame frame(4);
    constexpr std::array fwd = {
        GhostField::read_each_step("x", ColDType::Storage, 1, "test"),
        GhostField::read_each_step("y", ColDType::Storage, 1, "test"),
        GhostField::read_each_step("z", ColDType::Storage, 1, "test"),
    };
    constexpr std::array bwd = {
        GhostField::read_each_step("z", ColDType::Storage, 1, "test"),
        GhostField::read_each_step("y", ColDType::Storage, 1, "test"),
        GhostField::read_each_step("x", ColDType::Storage, 1, "test"),
    };
    GhostContract forwards;
    forwards.declare(fwd);
    GhostContract backwards;
    backwards.declare(bwd);

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
    const GhostContract s = base_fields::engine_contract();

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
    const GhostContract lean = base_fields::engine_contract();
    GhostContract fat = lean;
    constexpr std::array extra = {
        GhostField::read_stable(kAtom_charge, "CoulLong"),
    };
    fat.declare(extra);

    const auto lean_plan =
        cpu::comm::make_wire_plan(frame, cpu::comm::PackKind::Border, lean);
    const auto fat_plan =
        cpu::comm::make_wire_plan(frame, cpu::comm::PackKind::Border, fat);

    EXPECT_EQ(fat_plan.fields.size(), lean_plan.fields.size() + 1);
    EXPECT_EQ(fat_plan.bytes(1), lean_plan.bytes(1) + sizeof(StorageReal));
}

// ── 2. edge ───────────────────────────────────────────────────────────

/// A field the Frame does not carry is refused here too, naming the field and
/// its declarer. `check_ghosts` catches this at build(); reaching the wire
/// with a short Frame means the Frame changed shape afterwards.
TEST(TestWirePlan, AFieldTheFrameLacksIsRefused) {
    cpu::Frame frame(4);
    constexpr std::array table = {
        GhostField::read_each_step("x", ColDType::Storage, 1, "test"),
        GhostField::read_stable("not_a_column", ColDType::Storage, 1,
                                "GreedyComponent"),
    };
    GhostContract c;
    c.declare(table);

    try {
        (void)cpu::comm::make_wire_plan(frame, cpu::comm::PackKind::Border, c);
        FAIL() << "a wire naming a missing column must be refused";
    } catch (const std::exception& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("not_a_column"), std::string::npos);
        EXPECT_NE(what.find("GreedyComponent"), std::string::npos);
    }
}

/// An empty wire is a build-time mistake, not a run-time state — including
/// the case where the contract is non-empty but this pass selects nothing
/// from it (a run whose only declarations are reductions has no refresh).
TEST(TestWirePlan, APassThatSelectsNothingIsRefused) {
    cpu::Frame frame(4);
    const GhostContract empty;
    EXPECT_THROW((void)cpu::comm::make_wire_plan(
                     frame, cpu::comm::PackKind::Comm, empty),
                 std::exception);

    constexpr std::array reductions_only = {
        GhostField::reduce_to_owner("fx", ColDType::Acc, 1, "test"),
    };
    GhostContract c;
    c.declare(reductions_only);
    EXPECT_THROW(
        (void)cpu::comm::make_wire_plan(frame, cpu::comm::PackKind::Comm, c),
        std::exception);
}
