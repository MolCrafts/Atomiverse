/// @file tests/cpu/frame/test_schema.cpp
/// @brief The halo wire is derived from the mounted components, not declared
///        globally.
///
/// `comm-maint-09-schema` inverted the declaration: instead of a global table
/// in which each field says which wires it rides, each component says what it
/// reads on ghost rows and `build()` folds the answers.
///
/// The case that motivates it, and that these tests pin: `pair_dpd` is the
/// only style in the tree that reads a neighbour's velocity
/// (`pair_dpd.cpp:147`, `v[j] - v[i]`). Under the old global flags every run
/// shipped `vx`/`vy`/`vz` on every border message because that one style
/// might be mounted. Now a run carries them exactly when it mounts DPD.
///
/// Surface pinned (no MPI, no Driver):
/// @code{.cpp}
///   ATV::Schema, ATV::base_fields::{forward,border,reverse}
///   Component::declare_fields / declare_reverse_fields
///   cpu::check_schema
/// @endcode
///
/// Categories: basics (set semantics), domain (who declares what), edge
/// (idempotence, ordering).

#include <gtest/gtest.h>

#include <string_view>
#include <vector>

#include "atomiverse/core/schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_dpd.h"

using namespace ATV;
using ATV::Schema;

// ── 1. set semantics ──────────────────────────────────────────────────

/// Duplicate declarations collapse — two components asking for the same
/// column must not put it on the wire twice.
TEST(TestCommSchema, AddIsIdempotent) {
    Schema s;
    s.require("charge", ColDType::Storage, 1, "test");
    s.require("charge", ColDType::Storage, 1, "test");
    s.require("charge", ColDType::Storage, 1, "test");
    s.require("type", ColDType::Int, 1, "test");
    EXPECT_EQ(s.size(), 2u);
    EXPECT_TRUE(s.contains("charge"));
    EXPECT_TRUE(s.contains("type"));
    EXPECT_FALSE(s.contains("mass"));
}

/// Order is insertion order, and equality is order-sensitive: two ranks must
/// agree on the wire ORDER, not merely on the membership, or one packs what
/// the other cannot read.
TEST(TestCommSchema, EqualityIsOrderSensitive) {
    Schema a;
    a.require_all({"x", "y", "z"}, ColDType::Storage, "test");
    Schema b;
    b.require_all({"x", "y", "z"}, ColDType::Storage, "test");
    Schema c;
    c.require_all({"y", "x", "z"}, ColDType::Storage, "test");

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    {
        std::vector<std::string_view> got;
        for (const ATV::Field& f : a)
            got.push_back(f.name);
        EXPECT_EQ(got, (std::vector<std::string_view>{"x", "y", "z"}));
    }
}

// ── 2. the base set is the halo's own product ─────────────────────────

/// Positions are on the forward wire regardless of what any component says —
/// producing ghost positions is what the halo is for. Mirrors LAMMPS'
/// `default_comm = {"x"}`.
TEST(TestCommSchema, BaseForwardIsPositions) {
    Schema s;
    ATV::base_fields::forward(s);
    {
        std::vector<std::string_view> got;
        for (const ATV::Field& f : s)
            got.push_back(f.name);
        EXPECT_EQ(got, (std::vector<std::string_view>{"x", "y", "z"}));
    }
}

/// Border additionally needs the identity a ghost row cannot be used without.
TEST(TestCommSchema, BaseBorderAddsIdentity) {
    Schema s;
    ATV::base_fields::border(s);
    EXPECT_TRUE(s.contains("x"));
    EXPECT_TRUE(s.contains("tag"));
    EXPECT_TRUE(s.contains("type"));
    // and NOT the two the old global table shipped with no consumer
    EXPECT_FALSE(s.contains("mass"));
    EXPECT_FALSE(s.contains("vx"));
}

/// Reverse is the force accumulators.
TEST(TestCommSchema, BaseReverseIsForces) {
    Schema s;
    ATV::base_fields::reverse(s);
    {
        std::vector<std::string_view> got;
        for (const ATV::Field& f : s)
            got.push_back(f.name);
        EXPECT_EQ(got, (std::vector<std::string_view>{"fx", "fy", "fz"}));
    }
}

// ── 3. who declares what ──────────────────────────────────────────────

/// The judicial case. DPD asks for ghost velocities; nothing else does.
TEST(TestCommSchema, DpdDeclaresGhostVelocities) {
    cpu::PairDpd dpd{/*a=*/25.0, /*gamma=*/4.5, /*temperature=*/1.0,
                     /*cutoff=*/1.0, /*seed=*/12345};
    Schema s;
    dpd.declare_fields(s);
    EXPECT_TRUE(s.contains("vx"));
    EXPECT_TRUE(s.contains("vy"));
    EXPECT_TRUE(s.contains("vz"));
}

/// Real-space Coulomb asks for the neighbour charge
/// (`pair_lj_cut_coul_long.cpp:493`, `q_ptr[j]`).
TEST(TestCommSchema, CoulombDeclaresGhostCharge) {
    cpu::CoulLong coul{/*alpha=*/0.25, /*cutoff=*/9.0};
    Schema s;
    coul.declare_fields(s);
    EXPECT_TRUE(s.contains("charge"));
    EXPECT_FALSE(s.contains("vx"));
}

// ── 4. the wire follows the component set ─────────────────────────────

/// The whole point, end to end at the fold level: the same base set plus a
/// different component set gives a different wire. A Coulomb run carries
/// charge and no velocity; adding DPD adds velocity; neither carries mass,
/// because nothing reads a ghost's mass.
TEST(TestCommSchema, TheWireFollowsTheMountedComponents) {
    cpu::CoulLong coul{0.25, 9.0};
    cpu::PairDpd dpd{25.0, 4.5, 1.0, 1.0, 12345};

    Schema without_dpd;
    ATV::base_fields::border(without_dpd);
    coul.declare_fields(without_dpd);

    Schema with_dpd = without_dpd;
    dpd.declare_fields(with_dpd);

    EXPECT_TRUE(without_dpd.contains("charge"));
    EXPECT_FALSE(without_dpd.contains("vx"));

    EXPECT_TRUE(with_dpd.contains("charge"));
    EXPECT_TRUE(with_dpd.contains("vx"));

    // 24 bytes per ghost, present exactly when something reads them.
    EXPECT_EQ(with_dpd.size(), without_dpd.size() + 3);

    // Nothing reads a ghost's mass, so it is on neither wire. Under the old
    // global kColBorder table it was on both.
    EXPECT_FALSE(without_dpd.contains("mass"));
    EXPECT_FALSE(with_dpd.contains("mass"));
}

// ── 5. the Schema CHECKS the Frame ────────────────────────────────────

/// The Schema-to-Frame relationship is validation, not construction. A Frame
/// is a mapping and arrives with whatever its source gave it; the Schema says
/// what the mounted components need, and a shortfall throws with a how-to
/// rather than defaulting to something plausible.
TEST(TestSchema, CheckPassesWhenTheFrameCarriesEverything) {
    cpu::Frame f(4);
    Schema s;
    ATV::base_fields::border(s);
    s.require("charge", ColDType::Storage, 1, "test");
    EXPECT_NO_THROW(cpu::check_schema(s, f, "test"));
}

/// A declared field the Frame does not carry is refused, and the message
/// names it — the systematic form of CLAUDE.md's per-component rule.
TEST(TestSchema, CheckNamesEveryMissingField) {
    cpu::Frame f(4);
    Schema s;
    s.require_all({"x", "definitely_not_a_column", "also_missing"},
                  ColDType::Storage, "test");
    try {
        cpu::check_schema(s, f, "unit test");
        FAIL() << "a short Frame must be refused";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("definitely_not_a_column"), std::string::npos);
        EXPECT_NE(what.find("also_missing"), std::string::npos);
        EXPECT_NE(what.find("unit test"), std::string::npos);
        // a how-to, not just an assertion
        EXPECT_NE(what.find("Block::insert"), std::string::npos);
    }
}

// ── 6. pydantic-shaped validation ─────────────────────────────────────

/// A field that is present but the WRONG TYPE is the case a bare name list
/// cannot see, and it is the one that corrupts a wire silently rather than
/// loudly.
TEST(TestSchema, WrongDtypeIsReported) {
    cpu::Frame f(4);
    Schema s;
    // charge really is StorageReal in the Frame; declare it as an integer.
    s.require("charge", ColDType::Int, 1, "LiarComponent");

    const auto errors = cpu::validate_schema(s, f);
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_EQ(errors[0].field, "charge");
    EXPECT_EQ(errors[0].required_by, "LiarComponent");
    EXPECT_NE(errors[0].reason.find("StorageReal"), std::string::npos);
}

/// A field present with the wrong WIDTH is reported too — `image` is three
/// components, and something declaring it scalar would read one third of it.
TEST(TestSchema, WrongWidthIsReported) {
    cpu::Frame f(4);
    Schema s;
    s.require("image", ColDType::Int8, 1, "NarrowComponent");

    const auto errors = cpu::validate_schema(s, f);
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_NE(errors[0].reason.find("component(s)"), std::string::npos);
}

/// EVERY problem, not the first — fixing a schema one build at a time is
/// what makes a schema hateful.
TEST(TestSchema, ValidateReportsEveryProblemAtOnce) {
    cpu::Frame f(4);
    Schema s;
    s.require("x", ColDType::Storage, 1, "ok");
    s.require("missing_one", ColDType::Storage, 1, "A");
    s.require("missing_two", ColDType::Storage, 1, "B");
    s.require("charge", ColDType::Int, 1, "C");

    const auto errors = cpu::validate_schema(s, f);
    EXPECT_EQ(errors.size(), 3u);
    EXPECT_THROW(cpu::check_schema(s, f, "test"), std::runtime_error);
}

/// Two components disagreeing about what a field IS is a contradiction in
/// the model, caught at declaration rather than at the first memcpy.
TEST(TestSchema, ConflictingDeclarationsAreRecorded) {
    Schema s;
    s.require("charge", ColDType::Storage, 1, "CoulLong");
    s.require("charge", ColDType::Int, 1, "SomethingElse");

    ASSERT_EQ(s.conflicts().size(), 1u);
    EXPECT_EQ(s.conflicts()[0].field, "charge");
    EXPECT_NE(s.conflicts()[0].reason.find("CoulLong"), std::string::npos);
    // and the first declaration stands; the schema is not silently merged
    EXPECT_EQ(s.size(), 1u);
    EXPECT_EQ(s.find("charge")->dtype, ColDType::Storage);
}

/// An identical redeclaration is normal — two components both needing
/// charge is not a conflict.
TEST(TestSchema, IdenticalRedeclarationIsNotAConflict) {
    Schema s;
    s.require("charge", ColDType::Storage, 1, "CoulLong");
    s.require("charge", ColDType::Storage, 1, "PairLJCutCoulLong");
    EXPECT_TRUE(s.conflicts().empty());
    EXPECT_EQ(s.size(), 1u);
    // provenance keeps the first declarer, which is who the error will name
    EXPECT_EQ(s.find("charge")->required_by, "CoulLong");
}
