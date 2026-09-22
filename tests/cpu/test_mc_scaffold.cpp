/// @file tests/cpu/test_mc_scaffold.cpp
/// @brief Reference-free unit tests for the Monte Carlo sampling scaffold:
///        sampling::Phase shape, Mover/Potential kind tags, the pure-data
///        mc::Transaction, the SamplingServices handles, and cpu::mc::MC
///        component validation.

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/mover.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/driver/driver_node.h"
#include "atomiverse/cpu/montecarlo/mc.h"
#include "atomiverse/cpu/montecarlo/transaction.h"

using namespace ATV;

// ── sampling::Phase execution skeleton ────────────────────────────────────

TEST(MCScaffold, SamplingPhaseShape) {
    EXPECT_EQ(ATV::slot_of(sampling::Phase::propose), 0);
    EXPECT_EQ(ATV::slot_of(sampling::Phase::energy), 1);
    EXPECT_EQ(ATV::slot_of(sampling::Phase::end_of_step), 2);
    EXPECT_EQ(ATV::slot_of(sampling::Phase::COUNT), 3);

    const auto names = sampling::phase_names(sampling::Phase{});
    ASSERT_EQ(names.size(), 3u);
    EXPECT_EQ(names[0], "propose");
    EXPECT_EQ(names[1], "energy");
    EXPECT_EQ(names[2], "end_of_step");
}

// ── Kind tags: Mover and Potential are distinct types (no ComponentRole enum)

TEST(MCScaffold, MoverKindIsDistinct) {
    static_assert(std::is_base_of_v<cpu::Component, cpu::Mover>);
    static_assert(std::is_base_of_v<cpu::Component, cpu::Potential>);
    static_assert(!std::is_base_of_v<cpu::Mover, cpu::Potential>);
    static_assert(!std::is_base_of_v<cpu::Potential, cpu::Mover>);
    EXPECT_TRUE((std::is_base_of_v<cpu::Component, cpu::Mover>));
    EXPECT_TRUE((std::is_base_of_v<cpu::Component, cpu::Potential>));
}

// ── mc::Transaction: Frame-patch record + guards ────────────────────────────

TEST(MCScaffold, TransactionRecordsPatchRows) {
    cpu::Frame frame(6);
    frame.span(ATV::kAtom_x)[2] = 1.5;
    frame.span(ATV::kAtom_x)[5] = 2.5;

    cpu::mc::Transaction t;
    t.begin(cpu::mc::Transaction::Verb::Displace, frame);
    t.record_touch(frame, 2);
    t.record_touch(frame, 5);

    ASSERT_EQ(t.size(), 2u);
    EXPECT_EQ(t.verb, cpu::mc::Transaction::Verb::Displace);
    EXPECT_EQ(t.touched(0), LocalInt(2));
    EXPECT_EQ(t.touched(1), LocalInt(5));
    EXPECT_EQ(t.patch().nlocal(), 2);
    EXPECT_EQ(t.patch().span(ATV::kAtom_x)[0], StorageReal(1.5));
    EXPECT_EQ(t.patch().span(ATV::kAtom_x)[1], StorageReal(2.5));
}

TEST(MCScaffold, TransactionRollbackWithoutBeginThrows) {
    cpu::Frame frame(2);
    cpu::mc::Transaction t;
    EXPECT_THROW(t.rollback(frame), std::logic_error);
}

TEST(MCScaffold, TransactionClearEmpties) {
    cpu::Frame frame(3);
    cpu::mc::Transaction t;
    t.begin(cpu::mc::Transaction::Verb::Delete, frame);
    t.record_touch(frame, 1);
    ASSERT_EQ(t.size(), 1u);
    t.clear();
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.verb, cpu::mc::Transaction::Verb::None);
    EXPECT_THROW(t.rollback(frame), std::logic_error); // begin state gone too
}

// ── cpu::Context: the two new handles default to null ──────────────────────

TEST(MCScaffold, ContextHandlesDefaultNull) {
    cpu::Context ctx;
    EXPECT_EQ(ctx.mc.proposal, nullptr);
    EXPECT_EQ(ctx.mc.rng, nullptr);
}

// ── cpu::mc::MC component-validation gate ──────────────────────────────────

namespace {

/// @brief Minimal Mover-kind stub (no proposal logic) for validation tests.
struct StubMover : cpu::Mover {
    std::string name() const override { return "stub_mover"; }
    void launch(const cpu::LaunchContext&) override {}
};

/// @brief Minimal Potential-kind energy stub for validation tests.
struct StubEnergy : cpu::Potential {
    std::string name() const override { return "stub_energy"; }
    void launch(const cpu::LaunchContext&) override {}
};

/// @brief The exact shape a Gaussian method presents to an outer driver:
///        `Driver<scf::Phase>` (hence a `cpu::DriverNode`) + the pure `HF`
///        kind tag, and **not** a `Component`.
///
/// `cpu::scf::RHF` is this plus physics (`MeanFieldMethod` derives the same two
/// bases). The three `DriverNode` lifecycle ops — `build_on`,
/// `run_nested_erased`, `destroy_on` — are already implemented by
/// `cpu::Driver<PhaseEnum>`, so the only pure virtual left to satisfy is
/// `validate_components()`: a bare loop requires nothing. No physics, no
/// stages, no frame — this test is about *recognition*, and a stub keeps the
/// case out of `ATV_BUILD_HF` and in the milliseconds.
struct StubHfNode : cpu::Driver<ATV::scf::Phase>, cpu::HF {
    void validate_components() const override {}
};

/// @brief A nested loop node that is NOT an energy source: same
///        `Driver<scf::Phase>` node shape, **no** `HF` tag.
///
/// The node-aware walk must count the `HF` kind, not "any nested driver".
struct StubBareNode : cpu::Driver<ATV::scf::Phase> {
    void validate_components() const override {}
};

} // namespace

// `validate_components()` is protected on the preset, and it is `build()` that calls it
// — so these drive the gate through the public entry point rather than reaching past it.

TEST(MCScaffold, ValidateThrowsWithoutMover) {
    cpu::mc::MC d{/*temperature_K=*/300.0, /*seed=*/12345u};
    d.on(sampling::Phase::energy).add(StubEnergy{});
    EXPECT_THROW(d.build(), std::runtime_error);
}

TEST(MCScaffold, ValidateThrowsWithoutEnergyWriter) {
    // The MC ctor auto-adds ZeroEnergy at `energy` @ -999; it is not a
    // Potential kind, so it must NOT satisfy the energy-source requirement.
    cpu::mc::MC d{/*temperature_K=*/300.0, /*seed=*/12345u};
    d.on(sampling::Phase::propose).add(StubMover{});
    EXPECT_THROW(d.build(), std::runtime_error);
}

TEST(MCScaffold, ValidatePassesWithMoverAndEnergy) {
    cpu::mc::MC d{/*temperature_K=*/300.0, /*seed=*/12345u};
    d.on(sampling::Phase::propose).add(StubMover{});
    d.on(sampling::Phase::energy).add(StubEnergy{});
    EXPECT_NO_THROW(d.build());
}

// ── Node-aware energy-source recognition (esmm-contract-02, ac-001) ─────────
//
// `HF` is a pure polymorphic tag, not a Component: a Gaussian method is a
// `Driver<scf::Phase> + HF` bound as a NESTED DRIVER NODE. `McBase`'s
// validation walks components only (`for_each_pending`), so it cannot see any
// Gaussian method — the "RHF x MC" row of the free-combination matrix fails to
// build even though it would run. MD / Evaluate / Minimize / Hessian already
// use the node-aware `for_each_pending_kind` with a
// `dynamic_cast<const HF*>(node)` driver-node lambda.

/// ac-001 — a nested HF node IS the energy source. RED today: `build()` throws
/// "missing required Potential / Bias / HF / Semiempirical / PlaneWave / GPW"
/// because the component-only walk skips driver nodes entirely.
TEST(MCScaffold, NestedHfNodeSatisfiesEnergyValidation) {
    StubHfNode method;
    cpu::mc::MC d{/*temperature_K=*/300.0, /*seed=*/1u};
    d.on(sampling::Phase::propose).add(StubMover{});
    // NOTE: `add`/`add_ref`'s second positional argument is `order`, the third
    // is the nested budget — not `add(node, budget)`.
    d.on(sampling::Phase::energy)
        .add_ref(method, /*order=*/0, /*budget=*/0); // the ONLY energy source
    EXPECT_NO_THROW(d.build());
}

/// ac-004 — the walk change adds visibility, never permissiveness: a nested
/// node that does not carry the `HF` tag is still not an energy source. (Passes
/// today for the weaker reason that no node is looked at at all; it pins the
/// GREEN against a `dynamic_cast<const DriverNode*>`-shaped shortcut.)
TEST(MCScaffold, ValidateStillThrowsWithNonHfNestedNode) {
    StubBareNode not_a_method;
    cpu::mc::MC d{/*temperature_K=*/300.0, /*seed=*/1u};
    d.on(sampling::Phase::propose).add(StubMover{});
    d.on(sampling::Phase::energy)
        .add_ref(not_a_method, /*order=*/0,
                 /*budget=*/0);
    EXPECT_THROW(d.build(), std::runtime_error);
}

// The zero-source negative leg of ac-004 is `ValidateThrowsWithoutEnergyWriter`
// above (mover only, plus the driver-owned ZeroEnergy / ZeroForces: throws).

/// The temperature guard moved from `SamplingDriver`'s ctor to `cpu::Metropolis`'s;
/// `cpu::mc::MC` forwards to it, so constructing the preset with T <= 0 must still throw.
TEST(MCScaffold, MetropolisRejectsNonPositiveTemperature) {
    EXPECT_THROW((cpu::mc::MC{0.0, 1u}), std::invalid_argument);
}
