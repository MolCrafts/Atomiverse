/// @file tests/cpu/frame/test_ghost_contract.cpp
/// @brief What a component needs of a ghost row is a data-lifetime contract,
///        not a wire membership flag.
///
/// The declaration was inverted twice. First from a global table (each field
/// naming the wires it rides) to a per-component declaration. Then from
/// per-wire lists to a contract with two orthogonal axes: **freshness**
/// (owner → ghost, `stable` / `step`) and **data flow** (ghost → owner,
/// `reduce_to_owner`). The passes a contract compiles to are the
/// communication layer's business and are named nowhere above it.
///
/// The case that motivates it, and that these tests pin: `pair_dpd` is the
/// only style in the tree that reads a neighbour's velocity
/// (`pair_dpd.cpp:147`, `v[j] - v[i]`), and what it needs is not "send
/// velocities once per rebuild" but "\f$v_j\f$ is current when I compute
/// \f$F^D\f$". Under the old global flags every run shipped `vx`/`vy`/`vz` on
/// every border message because that one style might be mounted; under a
/// per-wire list it declared them on the wrong list and read them stale.
///
/// Surface pinned (no MPI, no Driver):
/// @code{.cpp}
///   ATV::GhostField, ATV::GhostContract
///   ATV::base_fields::kEngine, Component::declare_ghosts
///   cpu::check_ghosts / cpu::validate_ghosts
/// @endcode
///
/// Categories: basics (set semantics), domain (who declares what), edge
/// (idempotence, ordering, contradictions).

#include <gtest/gtest.h>

#include <array>
#include <string_view>
#include <utility>
#include <vector>

#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_coul_soft.h"
#include "atomiverse/cpu/potentials/pair_dpd.h"
#include "atomiverse/cpu/potentials/pair_gaussian_coul.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_tip4p_cut.h"

using namespace ATV;

namespace {

/// A component's declaration is a compile-time constant. Building the table
/// in a `constexpr` context is the assertion: if `GhostField` ever stopped
/// being a literal type, this stops compiling.
constexpr std::array kConstexprTable = {
    GhostField::read_stable("charge", ColDType::Storage, 1, "test"),
    GhostField::read_each_step("vx", ColDType::Storage, 1, "test"),
    GhostField::reduce_to_owner("fx", ColDType::Acc, 1, "test"),
};
static_assert(kConstexprTable.size() == 3);
static_assert(kConstexprTable[0].freshness == Freshness::stable);
static_assert(kConstexprTable[1].freshness == Freshness::step);
static_assert(kConstexprTable[2].access == GhostAccess::reduce);

GhostContract folded(std::span<const GhostField> fields) {
    GhostContract c;
    c.declare(fields);
    return c;
}

} // namespace

// ── 1. set semantics ──────────────────────────────────────────────────

/// Duplicate declarations collapse — two components asking for the same
/// column must not put it on a wire twice.
TEST(TestGhostContract, DeclareIsIdempotent) {
    constexpr std::array table = {
        GhostField::read_stable("charge", ColDType::Storage, 1, "test"),
        GhostField::read_stable("charge", ColDType::Storage, 1, "test"),
        GhostField::read_stable("type", ColDType::Int, 1, "test"),
    };
    const GhostContract c = folded(table);
    EXPECT_EQ(c.size(), 2u);
    EXPECT_TRUE(c.contains("charge"));
    EXPECT_TRUE(c.contains("type"));
    EXPECT_FALSE(c.contains("mass"));
}

/// Order is declaration order, and equality is order-sensitive: two ranks
/// must agree on the wire ORDER, not merely on the membership, or one packs
/// what the other cannot read.
TEST(TestGhostContract, EqualityIsOrderSensitive) {
    constexpr std::array xyz = {
        GhostField::read_each_step("x", ColDType::Storage, 1, "t"),
        GhostField::read_each_step("y", ColDType::Storage, 1, "t"),
        GhostField::read_each_step("z", ColDType::Storage, 1, "t"),
    };
    constexpr std::array yxz = {
        GhostField::read_each_step("y", ColDType::Storage, 1, "t"),
        GhostField::read_each_step("x", ColDType::Storage, 1, "t"),
        GhostField::read_each_step("z", ColDType::Storage, 1, "t"),
    };
    EXPECT_EQ(folded(xyz), folded(xyz));
    EXPECT_NE(folded(xyz), folded(yxz));

    std::vector<std::string_view> got;
    for (const GhostField& f : folded(xyz))
        got.push_back(f.name);
    EXPECT_EQ(got, (std::vector<std::string_view>{"x", "y", "z"}));
}

// ── 2. the two axes, and why they are not one enum ────────────────────

/// Presence is implied by the read, not declared beside it. There is no way
/// to ask for `step` freshness without the ghost carrying the field from the
/// moment it exists, because there are not two sets to keep in sync — which
/// is what the compile step then turns into border + refresh.
TEST(TestGhostContract, StepFreshnessIsStillJustOneDeclaration) {
    constexpr std::array table = {
        GhostField::read_each_step("moving", ColDType::Storage, 1, "test"),
    };
    const GhostContract c = folded(table);
    ASSERT_EQ(c.size(), 1u);
    EXPECT_EQ(c.find("moving")->access, GhostAccess::read);
    EXPECT_EQ(c.find("moving")->freshness, Freshness::step);
}

/// The weaker guarantee is the cheaper one: a field that does not change
/// while a ghost lives says so, and the compile step keeps it off the
/// per-step pass.
TEST(TestGhostContract, StableIsTheWeakerGuarantee) {
    constexpr std::array table = {
        GhostField::read_stable("fixed_param", ColDType::Storage, 1, "test"),
    };
    EXPECT_EQ(folded(table).find("fixed_param")->freshness, Freshness::stable);
}

/// A reduction is the other axis entirely — ghost→owner data flow, with no
/// freshness to state. Keeping it out of the freshness vocabulary is why
/// there is no single enum spanning `stable`, `step` and "force".
TEST(TestGhostContract, ReductionIsDataFlowNotFreshness) {
    constexpr std::array table = {
        GhostField::reduce_to_owner("tally", ColDType::Acc, 1, "test"),
    };
    EXPECT_EQ(folded(table).find("tally")->access, GhostAccess::reduce);
}

/// Two components asking for the same field merge, and the STRICTER
/// guarantee wins — in either declaration order, so the folded contract is
/// the same on every rank whatever the Plan order happens to be.
TEST(TestGhostContract, StricterFreshnessWinsInEitherOrder) {
    constexpr std::array lax_first = {
        GhostField::read_stable("shared", ColDType::Storage, 1, "A"),
        GhostField::read_each_step("shared", ColDType::Storage, 1, "B"),
    };
    constexpr std::array strict_first = {
        GhostField::read_each_step("shared", ColDType::Storage, 1, "B"),
        GhostField::read_stable("shared", ColDType::Storage, 1, "A"),
    };
    EXPECT_EQ(folded(lax_first).find("shared")->freshness, Freshness::step);
    EXPECT_EQ(folded(strict_first).find("shared")->freshness, Freshness::step);
}

/// One column cannot be both directions. A field read by one component and
/// accumulated by another is a modelling contradiction, not something to
/// merge — it needs two columns.
TEST(TestGhostContract, OppositeDirectionsOnOneColumnIsAConflict) {
    constexpr std::array table = {
        GhostField::read_stable("both_ways", ColDType::Acc, 1, "Reader"),
        GhostField::reduce_to_owner("both_ways", ColDType::Acc, 1, "Tallier"),
    };
    const GhostContract c = folded(table);
    ASSERT_EQ(c.conflicts().size(), 1u);
    EXPECT_EQ(c.conflicts()[0].field, "both_ways");
    EXPECT_NE(c.conflicts()[0].reason.find("two columns"), std::string::npos);
}

// ── 3. the engine's own declaration ───────────────────────────────────

/// Positions are read at step freshness regardless of what any component
/// says — producing current ghost positions is what the halo is for. Mirrors
/// LAMMPS' `default_comm = {"x"}`. Identity does not change while a ghost
/// lives; forces travel the other way.
TEST(TestGhostContract, EngineDeclarationSplitsByAxis) {
    const GhostContract c = base_fields::engine_contract();
    ASSERT_TRUE(c.contains("x"));
    EXPECT_EQ(c.find("x")->freshness, Freshness::step);
    ASSERT_TRUE(c.contains("tag"));
    EXPECT_EQ(c.find("tag")->freshness, Freshness::stable);
    ASSERT_TRUE(c.contains("type"));
    EXPECT_EQ(c.find("type")->freshness, Freshness::stable);
    ASSERT_TRUE(c.contains("fx"));
    EXPECT_EQ(c.find("fx")->access, GhostAccess::reduce);
    // and NOT the two the old global table shipped with no consumer
    EXPECT_FALSE(c.contains("mass"));
    EXPECT_FALSE(c.contains("vx"));
}

// ── 4. who declares what ──────────────────────────────────────────────

/// The judicial case. DPD asks for ghost velocities; nothing else does — and
/// what it asks for is not "send them every step" but "\f$v_j\f$ is current
/// when I compute \f$F^D\f$". That is `step` freshness.
TEST(TestGhostContract, DpdReadsGhostVelocitiesAtStepFreshness) {
    cpu::PairDpd dpd{/*a=*/25.0, /*gamma=*/4.5, /*temperature=*/1.0,
                     /*cutoff=*/1.0, /*seed=*/12345};
    const GhostContract c = folded(dpd.declare_ghosts());
    for (std::string_view v : {"vx", "vy", "vz"}) {
        ASSERT_TRUE(c.contains(v)) << v;
        EXPECT_EQ(c.find(v)->freshness, Freshness::step) << v;
        EXPECT_EQ(c.find(v)->access, GhostAccess::read) << v;
    }
}

/// Real-space Coulomb reads the neighbour charge
/// (`pair_lj_cut_coul_long.cpp:493`, `q_ptr[j]`) — but a fixed charge does
/// not change while a ghost lives, so `stable` is the honest guarantee and
/// it costs one send per rebuild instead of one per step.
TEST(TestGhostContract, CoulombReadsGhostChargeAtStableFreshness) {
    cpu::CoulLong coul{/*alpha=*/0.25, /*cutoff=*/9.0};
    const GhostContract c = folded(coul.declare_ghosts());
    ASSERT_TRUE(c.contains("charge"));
    EXPECT_EQ(c.find("charge")->freshness, Freshness::stable);
    EXPECT_FALSE(c.contains("vx"));
}

/// EVERY style that reads `q[j]` must declare it, not just the two that
/// happened to be converted first.
///
/// When the wire moved from a global `kColBorder` flag to a per-component
/// declaration, `charge` stopped riding the border pass unless somebody asked
/// for it — and four styles that read a neighbour's charge never did. On one
/// rank nothing changed (there are no ghosts); at 8 ranks `coul/cut` read
/// whatever those ghost rows happened to hold and pumped 44% onto the total
/// energy, with `sum_i F_i` still exactly zero because the bogus `q_i q_j`
/// is symmetric. `regressions/md/multihop-widebox-parity` caught it; this is
/// the unit-level guard.
TEST(TestGhostContract, EveryGhostChargeReaderDeclaresIt) {
    const cpu::PairCoulCut coul_cut{/*cutoff=*/9.0};
    const cpu::PairCoulSoft coul_soft{/*cutoff=*/9.0, /*sigma_c=*/3.0};
    const cpu::PairGaussianCoul gaussian{/*cutoff=*/9.0,
                                         /*eta_by_type=*/{1.0}};
    const cpu::PairLJCutTip4pCut tip4p{
        /*type_O=*/0,          /*type_H=*/1,
        /*qdist=*/0.15,
        /*r_oh=*/0.9572,       /*theta_hoh=*/1.82421813,
        /*epsilon_OO=*/0.0067,
        /*sigma_OO=*/3.1589,   /*cut_lj=*/9.0};
    const cpu::CoulLong coul_long{/*alpha=*/0.25, /*cutoff=*/9.0};
    const cpu::PairLJCutCoulLong lj_coul_long{
        {{0, 0, 0.0067, 3.1589}}, /*alpha=*/0.25, /*cutoff=*/9.0};

    const std::vector<std::pair<std::string_view, GhostContract>> styles = {
        {"coul/cut", folded(coul_cut.declare_ghosts())},
        {"coul/soft", folded(coul_soft.declare_ghosts())},
        {"gaussian/coul", folded(gaussian.declare_ghosts())},
        {"lj/cut/tip4p/cut", folded(tip4p.declare_ghosts())},
        {"coul/long", folded(coul_long.declare_ghosts())},
        {"lj/cut/coul/long", folded(lj_coul_long.declare_ghosts())},
    };
    for (const auto& [name, c] : styles) {
        SCOPED_TRACE(name);
        ASSERT_TRUE(c.contains("charge"));
        EXPECT_EQ(c.find("charge")->access, GhostAccess::read);
        // `stable`: a fixed charge does not change while a ghost lives, and
        // a fluctuating-charge owner refreshes ghosts itself through
        // `ChargePacker`. Asking for `step` here would put charge on every
        // step's refresh message for every run that mounts a Coulomb style.
        EXPECT_EQ(c.find("charge")->freshness, Freshness::stable);
    }
}

/// The whole point, end to end at the fold level: the same base declaration
/// plus a different component set gives a different contract. A Coulomb run
/// carries charge and no velocity; adding DPD adds velocity at the stricter
/// guarantee; neither carries mass, because nothing reads a ghost's mass.
TEST(TestGhostContract, TheContractFollowsTheMountedComponents) {
    cpu::CoulLong coul{0.25, 9.0};
    cpu::PairDpd dpd{25.0, 4.5, 1.0, 1.0, 12345};

    GhostContract without;
    without.declare(base_fields::kEngine);
    without.declare(coul.declare_ghosts());

    GhostContract with = without;
    with.declare(dpd.declare_ghosts());

    EXPECT_TRUE(without.contains("charge"));
    EXPECT_FALSE(without.contains("vx"));
    EXPECT_TRUE(with.contains("charge"));
    EXPECT_TRUE(with.contains("vx"));

    // Three more fields, and they arrive at the stricter guarantee — which
    // is what makes the freshness level a real cost decision rather than a
    // formality.
    EXPECT_EQ(with.size(), without.size() + 3);
    EXPECT_EQ(with.find("vx")->freshness, Freshness::step);

    // Nothing reads a ghost's mass, so it is on no wire. Under the old
    // global kColBorder table it was on both.
    EXPECT_FALSE(without.contains("mass"));
    EXPECT_FALSE(with.contains("mass"));
}

// ── 5. the contract CHECKS the Frame ──────────────────────────────────

/// The contract-to-Frame relationship is validation, not construction. A
/// Frame is a mapping and arrives with whatever its source gave it; the
/// contract says what the mounted components need, and a shortfall throws
/// with a how-to rather than defaulting to something plausible.
TEST(TestGhostContract, CheckPassesWhenTheFrameCarriesEverything) {
    cpu::Frame f(4);
    GhostContract c = base_fields::engine_contract();
    constexpr std::array extra = {
        GhostField::read_stable("charge", ColDType::Storage, 1, "test"),
    };
    c.declare(extra);
    EXPECT_NO_THROW(cpu::check_ghosts(c, f, "test"));
}

/// A declared field the Frame does not carry is refused, and the message
/// names it — the systematic form of CLAUDE.md's per-component rule.
TEST(TestGhostContract, CheckNamesEveryMissingField) {
    cpu::Frame f(4);
    constexpr std::array table = {
        GhostField::read_each_step("x", ColDType::Storage, 1, "test"),
        GhostField::read_stable("definitely_not_a_column", ColDType::Storage, 1,
                                "test"),
        GhostField::read_stable("also_missing", ColDType::Storage, 1, "test"),
    };
    try {
        cpu::check_ghosts(folded(table), f, "unit test");
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
TEST(TestGhostContract, WrongDtypeIsReported) {
    cpu::Frame f(4);
    // charge really is StorageReal in the Frame; declare it as an integer.
    constexpr std::array table = {
        GhostField::read_stable("charge", ColDType::Int, 1, "LiarComponent"),
    };
    const auto errors = cpu::validate_ghosts(folded(table), f);
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_EQ(errors[0].field, "charge");
    EXPECT_EQ(errors[0].required_by, "LiarComponent");
    EXPECT_NE(errors[0].reason.find("StorageReal"), std::string::npos);
}

/// A field present with the wrong WIDTH is reported too — `image` is three
/// components, and something declaring it scalar would read one third of it.
TEST(TestGhostContract, WrongWidthIsReported) {
    cpu::Frame f(4);
    constexpr std::array table = {
        GhostField::read_stable("image", ColDType::Int8, 1, "NarrowComponent"),
    };
    const auto errors = cpu::validate_ghosts(folded(table), f);
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_NE(errors[0].reason.find("component(s)"), std::string::npos);
}

/// EVERY problem, not the first — fixing a contract one build at a time is
/// what makes a schema hateful.
TEST(TestGhostContract, ValidateReportsEveryProblemAtOnce) {
    cpu::Frame f(4);
    constexpr std::array table = {
        GhostField::read_each_step("x", ColDType::Storage, 1, "ok"),
        GhostField::read_stable("missing_one", ColDType::Storage, 1, "A"),
        GhostField::read_stable("missing_two", ColDType::Storage, 1, "B"),
        GhostField::read_stable("charge", ColDType::Int, 1, "C"),
    };
    const GhostContract c = folded(table);
    EXPECT_EQ(cpu::validate_ghosts(c, f).size(), 3u);
    EXPECT_THROW(cpu::check_ghosts(c, f, "test"), std::runtime_error);
}

/// Two components disagreeing about what a field IS is a contradiction in
/// the model, caught at declaration rather than at the first memcpy.
TEST(TestGhostContract, ConflictingDeclarationsAreRecorded) {
    constexpr std::array table = {
        GhostField::read_stable("charge", ColDType::Storage, 1, "CoulLong"),
        GhostField::read_stable("charge", ColDType::Int, 1, "SomethingElse"),
    };
    const GhostContract c = folded(table);
    ASSERT_EQ(c.conflicts().size(), 1u);
    EXPECT_EQ(c.conflicts()[0].field, "charge");
    EXPECT_NE(c.conflicts()[0].reason.find("CoulLong"), std::string::npos);
    // and the first declaration stands; the contract is not silently merged
    EXPECT_EQ(c.size(), 1u);
    EXPECT_EQ(c.find("charge")->dtype, ColDType::Storage);
}

/// An identical redeclaration is normal — two components both needing
/// charge is not a conflict.
TEST(TestGhostContract, IdenticalRedeclarationIsNotAConflict) {
    constexpr std::array table = {
        GhostField::read_stable("charge", ColDType::Storage, 1, "CoulLong"),
        GhostField::read_stable("charge", ColDType::Storage, 1,
                                "PairLJCutCoulLong"),
    };
    const GhostContract c = folded(table);
    EXPECT_TRUE(c.conflicts().empty());
    EXPECT_EQ(c.size(), 1u);
    // provenance keeps the first declarer, which is who the error will name
    EXPECT_EQ(c.find("charge")->required_by, "CoulLong");
}
