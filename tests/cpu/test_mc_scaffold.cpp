/// @file tests/cpu/test_mc_scaffold.cpp
/// @brief Reference-free unit tests for the Monte Carlo sampling scaffold:
///        sampling::Phase shape, Mover/Potential kind tags, the pure-data
///        mc::Transaction, the SamplingServices handles, and cpu::mc::MC
///        component validation.

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/mover.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/mc/mc.h"
#include "atomiverse/cpu/mc/transaction.h"
#include "atomiverse/facade/stage.h"

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

/// The temperature guard moved from `SamplingDriver`'s ctor to `cpu::Metropolis`'s;
/// `cpu::mc::MC` forwards to it, so constructing the preset with T <= 0 must still throw.
TEST(MCScaffold, MetropolisRejectsNonPositiveTemperature) {
    EXPECT_THROW((cpu::mc::MC{0.0, 1u}), std::invalid_argument);
}
