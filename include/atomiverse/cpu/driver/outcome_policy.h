/// @file cpu/driver/outcome_policy.h
/// @brief OutcomePolicy — "does this iteration count?", the loop's *other* policy slot.
///
/// A loop has two policy slots, and they are orthogonal:
///
///   - `Criterion`     — *should the loop stop?*        (`component/criterion.h`)
///   - `OutcomePolicy` — *does this iteration count?*   (here)
///
/// Monte Carlo is why the second one exists. Its accept/reject test is **not** a
/// termination: rejecting a trial move discards one iteration's work and keeps going.
/// Because the two were conflated, `SamplingDriver` had to exist as a class whose entire
/// contribution was an `if` sitting between the stages and the termination test —
/// hardcoded Metropolis, unreachable from any other loop, and impossible to swap for the
/// REMD/GCMC acceptance rules that were already on the roadmap.
///
/// Split apart, `SamplingDriver` has nothing left to be, and `cpu::mc::MC` becomes what
/// it always was: a preset that attaches `Metropolis` in its constructor.
///
/// ## The division of labour on a rejected iteration
///
/// The policy **decides**; whoever recorded the change **undoes** it. `Metropolis` never
/// touches the frame — it reads an energy and returns `Outcome::Rollback`. The undo is
/// performed by the preset that owns the changeset (`mc::Transaction`, published on
/// `ctx.mc.proposal` by the Mover). A policy that could reach into the frame would have
/// to know what kind of move it was undoing, which is exactly the coupling this seam
/// exists to prevent.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/facade/loop.h"

namespace ATV {
namespace cpu {

/// @brief Per-iteration commit/rollback policy. Asked once, after the stages have run.
///
/// Like `Criterion`, deliberately **not** a template: it sees a `LoopState` and never
/// a phase, so one policy serves any workflow.
class OutcomePolicy {
  public:
    virtual ~OutcomePolicy() = default;

    /// @brief Stable identifier, for diagnostics.
    virtual const char* name() const = 0;

    /// @brief Reset to a fresh run's initial state.
    ///
    /// Called once per `run()` that starts a new chain (not on a continuation), after
    /// the bootstrap pass, so a policy can seed itself from the initial configuration.
    /// @param s The loop state after the bootstrap pass.
    virtual void on_start(const cpu::LoopState& s) { (void)s; }

    /// @brief Keep or discard the iteration that just ran.
    /// @param s State after this iteration's stages. `s.frame` holds the TRIAL state.
    /// @return `Commit` to keep it, `Rollback` to have the caller undo it.
    virtual Outcome after_stages(const cpu::LoopState& s) = 0;
};

/// @brief The Metropolis acceptance test — `min(1, exp(-ΔE / k_B T))`.
///
/// Accepts every downhill move and an uphill move with probability `exp(-ΔE/k_B T)`,
/// against the energy of the last **accepted** configuration. Owns that cached energy,
/// its own RNG stream, and the accept/attempt tallies — all of which used to sit on
/// `SamplingDriver` because there was nowhere else to put them.
///
/// This is a `Sampling`-flavoured policy but nothing in the loop knows that: it reads
/// `frame.meta.get(kEnergy)` and returns an `Outcome`.
class Metropolis : public OutcomePolicy {
  public:
    /// @brief Construct the NVT acceptance test.
    ///
    /// **Borrows** the RNG rather than seeding its own, and that is load-bearing: the
    /// Mover draws its displacements from the same stream (published on `ctx.mc.rng`),
    /// so the move proposals and the acceptance draws interleave in one reproducible
    /// sequence. Giving the policy a private generator would silently change every
    /// trajectory a given seed produces.
    ///
    /// @param temperature_K Ensemble temperature [K] (must be > 0).
    /// @param rng The run's single RNG stream. Must outlive this policy (the driver owns
    ///        it and publishes it on the Context).
    /// @throws std::invalid_argument if `temperature_K <= 0`.
    Metropolis(AccReal temperature_K, std::mt19937_64& rng)
        : temperature_(temperature_K), rng_(rng) {
        if (!(temperature_K > AccReal(0)))
            throw std::invalid_argument(
                "Metropolis: temperature must be > 0 K");
    }

    /// @brief Stable identifier, for diagnostics.
    /// @return `"metropolis"`.
    const char* name() const override { return "metropolis"; }

    /// @brief Bootstrap invariant: seed the reference energy from the INITIAL
    ///        configuration, before any trial move. Without this the first ΔE is
    ///        measured against zero and is meaningless.
    /// @param s Loop state after the bootstrap pass; `s.frame.meta.get(kEnergy)` [eV] is
    ///        the initial configuration's energy.
    void on_start(const cpu::LoopState& s) override {
        e_cur_ = s.frame.meta.get(kEnergy);
    }

    /// @brief Apply the acceptance test to the trial energy now in the frame.
    /// @param s Loop state; `s.frame.meta.get(kEnergy)` is the trial energy [eV].
    /// @return `Commit` on accept, `Rollback` on reject.
    Outcome after_stages(const cpu::LoopState& s) override {
        const AccReal e_trial = s.frame.meta.get(kEnergy);
        const AccReal dE = e_trial - e_cur_;
        ++n_attempts_;
        if (accept_(dE)) {
            e_cur_ = e_trial; // the frame already holds the move
            ++n_accepts_;
            return Outcome::Commit;
        }
        return Outcome::Rollback;
    }

    /// @brief Energy of the current (last-accepted) configuration.
    /// @return Energy [eV].
    AccReal current_energy() const { return e_cur_; }

    /// @brief Total trial moves attempted (lifetime).
    /// @return Attempt count.
    long long attempts() const { return n_attempts_; }

    /// @brief Total trial moves accepted (lifetime).
    /// @return Accept count.
    long long accepts() const { return n_accepts_; }

    /// @brief Accepted fraction of attempted moves, in [0,1] (0 if none attempted).
    double acceptance_ratio() const {
        return n_attempts_ > 0 ? static_cast<double>(n_accepts_)
                                     / static_cast<double>(n_attempts_)
                               : 0.0;
    }

  private:
    /// @brief The test itself.
    /// @param dE Trial energy change [eV].
    /// @return True to accept the trial move.
    bool accept_(AccReal dE) {
        if (dE <= AccReal(0))
            return true;
        const double beta =
            1.0 / (ATV::units::kB_ev_per_k * static_cast<double>(temperature_));
        const double p = std::exp(-static_cast<double>(dE) * beta);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return u(rng_) < p;
    }

    AccReal temperature_; ///< Ensemble temperature [K].
    std::mt19937_64&
        rng_;           ///< The run's single RNG stream, shared with the Mover.
    AccReal e_cur_ = 0; ///< Energy of the last accepted state [eV].
    long long n_attempts_ = 0; ///< Trial moves attempted (lifetime).
    long long n_accepts_ = 0;  ///< Trial moves accepted (lifetime).
};

} // namespace cpu
} // namespace ATV
