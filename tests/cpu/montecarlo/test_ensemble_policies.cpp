/// @file tests/cpu/mc/test_ensemble_policies.cpp
/// @brief ac-003 of mc-gcmc-npt: formula-level unit tests for the μVT
///        (`GcmcMetropolis`) and NPT (`NptMetropolis`) `OutcomePolicy`
///        implementations, driven directly through `after_stages` with a
///        hand-built `LoopState` — no driver run.

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "atomiverse/core/units.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/gcmc.h"
#include "atomiverse/cpu/montecarlo/npt_mc.h"
#include "atomiverse/cpu/montecarlo/transaction.h"

using namespace ATV;
using Verb = cpu::mc::Transaction::Verb;

namespace {

/// @brief Minimal harness owning everything a `LoopState` borrows.
struct PolicyHarness {
    cpu::Frame frame{4};
    cpu::Context ctx{};
    cpu::mc::Transaction txn{};
    ConvergenceMetrics metrics{};
    std::mt19937_64 rng{99u};

    PolicyHarness() {
        frame.box.set_cell(Mat3::diagonal(20.0, 20.0, 20.0));
        ctx.mc.proposal = &txn;
        ctx.mc.rng = &rng;
    }

    cpu::LoopState state(int iteration = 0) const {
        return cpu::LoopState{iteration, frame, ctx, metrics};
    }

    void stamp(Verb v) { txn.verb = v; }
};

} // namespace

// ── GcmcMetropolis ──────────────────────────────────────────────────────────

TEST(GcmcMetropolis, RejectsBadConstruction) {
    std::mt19937_64 rng(1u);
    EXPECT_THROW((cpu::mc::GcmcMetropolis{0.0, 0.0, 1.0, rng}),
                 std::invalid_argument);
    EXPECT_THROW((cpu::mc::GcmcMetropolis{300.0, 0.0, -1.0, rng}),
                 std::invalid_argument);
}

TEST(GcmcMetropolis, LambdaCubedMatchesUnits) {
    std::mt19937_64 rng(1u);
    cpu::mc::GcmcMetropolis p{120.0, 0.0, 39.948, rng};
    const double lam = units::lambda_thermal(39.948, 120.0);
    EXPECT_DOUBLE_EQ(p.lambda_cubed(), lam * lam * lam);
}

TEST(GcmcMetropolis, DownhillDisplacementAlwaysCommits) {
    PolicyHarness h;
    cpu::mc::GcmcMetropolis p{120.0, 0.0, 39.948, h.rng};
    h.frame.meta.get(ATV::kEnergy) = -1.0;
    p.on_start(h.state());
    h.frame.meta.get(ATV::kEnergy) = -1.5; // dE < 0
    h.stamp(Verb::Displace);
    EXPECT_EQ(p.after_stages(h.state()), Outcome::Commit);
    EXPECT_EQ(p.current_energy(), -1.5); // reference moved to the trial
}

TEST(GcmcMetropolis, DisplacementAcceptanceMatchesBoltzmannFactor) {
    PolicyHarness h;
    const double T = 300.0;
    const double dE = 0.02; // uphill
    cpu::mc::GcmcMetropolis p{T, 0.0, 39.948, h.rng};
    h.frame.meta.get(ATV::kEnergy) = 0.0;
    p.on_start(h.state());
    h.stamp(Verb::Displace);

    const int n_trials = 40000;
    int commits = 0;
    for (int k = 0; k < n_trials; ++k) {
        h.frame.meta.get(ATV::kEnergy) = dE; // fresh uphill trial from E=0
        if (p.after_stages(h.state()) == Outcome::Commit) {
            ++commits;
            h.frame.meta.get(ATV::kEnergy) = 0.0; // hop back so dE stays fixed
            h.stamp(Verb::Displace);
            // Bring the reference back to 0 with a certain downhill commit.
            ASSERT_EQ(p.after_stages(h.state()), Outcome::Commit);
        }
    }
    const double expected = std::exp(-dE / (units::kB_ev_per_k * T));
    const double observed = static_cast<double>(commits) / n_trials;
    const double sigma = std::sqrt(expected * (1.0 - expected) / n_trials);
    EXPECT_NEAR(observed, expected, 4.0 * sigma);
}

TEST(GcmcMetropolis, InsertionPrefactorGovernsTheDecision) {
    // dE = 0, μ = 0: arg = −ln(V / (Λ³·N_new)). V >> Λ³·N_new ⇒ arg < 0 ⇒
    // deterministic commit, no RNG draw.
    PolicyHarness h;
    cpu::mc::GcmcMetropolis p{120.0, 0.0, 39.948, h.rng};
    h.frame.meta.get(ATV::kEnergy) = 0.0;
    p.on_start(h.state());
    h.stamp(Verb::Insert); // trial frame already holds N+1 = 4 atoms
    EXPECT_EQ(p.after_stages(h.state()), Outcome::Commit);
}

TEST(GcmcMetropolis, HopelessInsertionIsRejected) {
    // μ = −10 eV at 120 K: βμ ≈ −967 ⇒ exp(-arg) ≈ 0 — rejected every time.
    PolicyHarness h;
    cpu::mc::GcmcMetropolis p{120.0, -10.0, 39.948, h.rng};
    h.frame.meta.get(ATV::kEnergy) = 0.0;
    p.on_start(h.state());
    h.stamp(Verb::Insert);
    for (int k = 0; k < 100; ++k)
        EXPECT_EQ(p.after_stages(h.state()), Outcome::Rollback);
}

TEST(GcmcMetropolis, DeletionPrefactorGovernsTheDecision) {
    // dE = 0, μ = 0: arg = −ln(Λ³·N_old / V). Make the box tiny so
    // Λ³·N_old > V ⇒ deterministic commit.
    PolicyHarness h;
    h.frame.box.set_cell(Mat3::diagonal(0.1, 0.1, 0.1));
    cpu::mc::GcmcMetropolis p{120.0, 0.0, 39.948, h.rng};
    h.frame.meta.get(ATV::kEnergy) = 0.0;
    p.on_start(h.state());
    h.stamp(Verb::Delete);
    EXPECT_EQ(p.after_stages(h.state()), Outcome::Commit);
}

TEST(GcmcMetropolis, VolumeVerbThrows) {
    PolicyHarness h;
    cpu::mc::GcmcMetropolis p{120.0, 0.0, 39.948, h.rng};
    p.on_start(h.state());
    h.stamp(Verb::Box);
    EXPECT_THROW(p.after_stages(h.state()), std::runtime_error);
}

// ── NptMetropolis ───────────────────────────────────────────────────────────

TEST(NptMetropolis, RejectsBadConstruction) {
    std::mt19937_64 rng(1u);
    EXPECT_THROW((cpu::mc::NptMetropolis{-1.0, 0.0, rng}),
                 std::invalid_argument);
}

TEST(NptMetropolis, SeedsEnergyAndVolume) {
    PolicyHarness h;
    cpu::mc::NptMetropolis p{300.0, 0.0, h.rng};
    h.frame.meta.get(ATV::kEnergy) = -2.5;
    p.on_start(h.state());
    EXPECT_EQ(p.current_energy(), -2.5);
    EXPECT_DOUBLE_EQ(p.current_volume(), 20.0 * 20.0 * 20.0);
}

TEST(NptMetropolis, ZeroPressureExpansionAlwaysCommits) {
    // dE = 0, P = 0: arg = −N·ln(V'/V) < 0 for an expansion — deterministic.
    PolicyHarness h;
    cpu::mc::NptMetropolis p{300.0, 0.0, h.rng};
    h.frame.meta.get(ATV::kEnergy) = 0.0;
    p.on_start(h.state());
    h.frame.box.scale_isotropic(1.05);
    h.stamp(Verb::Box);
    EXPECT_EQ(p.after_stages(h.state()), Outcome::Commit);
    EXPECT_DOUBLE_EQ(p.current_volume(), h.frame.box.volume());
}

TEST(NptMetropolis, CrushingPressureRejectsExpansion) {
    // P·ΔV dominates: a large expansion against a huge pressure has
    // exp(-arg) ≈ 0.
    PolicyHarness h;
    cpu::mc::NptMetropolis p{300.0, /*P [eV/Å³]=*/10.0, h.rng};
    h.frame.meta.get(ATV::kEnergy) = 0.0;
    p.on_start(h.state());
    h.frame.box.scale_isotropic(1.5);
    h.stamp(Verb::Box);
    for (int k = 0; k < 100; ++k)
        EXPECT_EQ(p.after_stages(h.state()), Outcome::Rollback);
}

TEST(NptMetropolis, InsertVerbThrows) {
    PolicyHarness h;
    cpu::mc::NptMetropolis p{300.0, 0.0, h.rng};
    p.on_start(h.state());
    h.stamp(Verb::Insert);
    EXPECT_THROW(p.after_stages(h.state()), std::runtime_error);
}
