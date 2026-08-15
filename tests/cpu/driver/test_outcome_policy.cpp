/// @file tests/cpu/driver/test_outcome_policy.cpp
/// @brief Unit tests for `cpu::Metropolis`, the NVT `OutcomePolicy`
///        (`include/atomiverse/cpu/driver/outcome_policy.h`).
///
/// True unit tests: `after_stages()` is called directly against a hand-built
/// `LoopState`, exactly as the sibling policies are exercised in
/// tests/cpu/montecarlo/test_ensemble_policies.cpp. No driver run — a full
/// `cpu::mc::MC::run()` is an end-to-end path and belongs to
/// `regressions/mc/`, not to this ms-scale suite.
///
/// The contract under test is the **verb** contract. `Metropolis` implements
/// one ensemble (canonical, fixed N and V), so `exp(-βΔE)` is the whole
/// acceptance rule only for a `Displace` proposal. A `Box` / `Insert` /
/// `Delete` trial carries the volume and particle-number prefactors that
/// `NptMetropolis` and `GcmcMetropolis` supply; judging one by `exp(-βΔE)`
/// alone samples no defined ensemble at all, so it must throw rather than
/// silently accept — the same guard both siblings already have
/// (npt_mc.h:90, gcmc.h:104).

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <string>

#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/outcome_policy.h"
#include "atomiverse/cpu/frame/frame.h"
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

bool message_mentions(const std::runtime_error& e, const std::string& needle) {
    return std::string(e.what()).find(needle) != std::string::npos;
}

} // namespace

// ── cpu::Metropolis — the canonical (NVT) verb contract ────────────────────

TEST(MetropolisPolicy, VolumeVerbThrowsNamingTheNptPreset) {
    // A VolumeMove under `cpu::mc::MC` is currently judged by exp(-βΔE) with
    // no P·ΔV and no N·ln(V'/V) term, and a downhill draw is silently
    // committed — an undefined ensemble, not an approximation of NPT.
    PolicyHarness h;
    cpu::Metropolis p{300.0, h.rng};
    h.frame.meta.get(ATV::kEnergy) = 0.0;
    p.on_start(h.state());
    h.frame.box.scale_isotropic(1.05);     // trial box, as VolumeMove leaves it
    h.frame.meta.get(ATV::kEnergy) = -1.0; // downhill ⇒ today: silent Commit
    h.stamp(Verb::Box);
    try {
        p.after_stages(h.state());
        FAIL() << "expected throw — NVT Metropolis has no rule for a Box verb";
    } catch (const std::runtime_error& e) {
        // Substring the message must carry: the preset that DOES own this
        // verb, so the guidance is actionable. `NptMc` is the class name in
        // include/atomiverse/cpu/montecarlo/npt_mc.h:140.
        EXPECT_TRUE(message_mentions(e, "NptMc")) << e.what();
    }
}

TEST(MetropolisPolicy, InsertVerbThrowsNamingTheGcmcPreset) {
    // Same defect on the μVT side: an insertion judged without the
    // V/(Λ³·N) prefactor is not grand-canonical sampling.
    PolicyHarness h;
    cpu::Metropolis p{300.0, h.rng};
    h.frame.meta.get(ATV::kEnergy) = 0.0;
    p.on_start(h.state());
    h.frame.meta.get(ATV::kEnergy) = -1.0;
    h.stamp(Verb::Insert);
    try {
        p.after_stages(h.state());
        FAIL() << "expected throw — NVT Metropolis has no rule for an Insert "
                  "verb";
    } catch (const std::runtime_error& e) {
        // `Gcmc` is the class name in
        // include/atomiverse/cpu/montecarlo/gcmc.h:155.
        EXPECT_TRUE(message_mentions(e, "Gcmc")) << e.what();
    }
}

TEST(MetropolisPolicy, DisplaceVerbIsStillJudgedByBoltzmann) {
    // Guard on the fix: the verb check must reject foreign verbs, not every
    // verb. A downhill displacement is a deterministic commit.
    PolicyHarness h;
    cpu::Metropolis p{300.0, h.rng};
    h.frame.meta.get(ATV::kEnergy) = -1.0;
    p.on_start(h.state());
    h.frame.meta.get(ATV::kEnergy) = -1.5; // dE < 0
    h.stamp(Verb::Displace);
    EXPECT_EQ(p.after_stages(h.state()), Outcome::Commit);
    EXPECT_EQ(p.current_energy(), -1.5);
}

TEST(MetropolisPolicy, EmptyProposalDoesNotThrow) {
    // The other half of the empty-cell path: a Mover with nothing to propose
    // stamps `Verb::None` (TranslateMove on nlocal() == 0, InsertDeleteMove
    // deleting from N == 0). The canonical policy owns that verb and must
    // dispose of the trial without a Boltzmann draw — never throw on it.
    PolicyHarness h;
    cpu::Metropolis p{300.0, h.rng};
    h.frame.meta.get(ATV::kEnergy) = -1.0;
    p.on_start(h.state());
    h.stamp(Verb::None);
    EXPECT_NO_THROW((void)p.after_stages(h.state()));
}
