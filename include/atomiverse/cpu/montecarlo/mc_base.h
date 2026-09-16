/// @file cpu/montecarlo/mc_base.h
/// @brief cpu::mc::McBase — the shared Monte Carlo loop under the NVT / NPT /
///        μVT presets.
#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <utility>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/mover.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/transaction.h"

namespace ATV {
namespace cpu {
namespace mc {

/// @brief The Monte Carlo loop shared by every MC preset (`MC`, `NptMc`,
///        `Gcmc`). A preset is its constructor: it attaches ITS
///        `OutcomePolicy` (and default movers, if any); this base owns
///        everything ensemble-agnostic — the trial loop, the RNG stream, the
///        `Transaction` changeset, rollback, and the per-verb tallies.
///
/// Per trial (phases `propose` → `energy` → `end_of_step`):
///   1. `propose` — a Mover mutates the frame in place and records the
///      changeset (a Frame patch) into `ctx.mc.proposal`.
///   2. `energy` — ZeroEnergy (bound here, order −999), then the registered
///      evaluator recomputes the **full-frame** trial energy. Full-frame
///      (rather than a local sub-frame) is the only form valid for *all*
///      potentials, including long-range/kspace where a single-atom energy
///      is undefined.
///   3. the `OutcomePolicy` keeps or discards it. On `Rollback` the frame is
///      restored by splicing the patch back — the policy decides, the
///      Transaction that recorded the change undoes.
///
/// **Bootstrap invariant:** the policy is seeded from one full-frame energy
/// evaluation of the *initial* configuration before the first trial move
/// (step index `kBootstrapStep`; a Mover proposes nothing on a negative
/// step). Without it the first ΔE is measured against zero.
class McBase : public Driver<ATV::sampling::Phase> {
  public:
    /// @brief Flush the deferred component `destroy()` of a live chain.
    ~McBase() { this->flush_live_(); }

    /// @brief Advance the system by `n_trials` Monte Carlo trial moves.
    ///
    /// **Continuable.** Consecutive calls on the same frame extend one Markov
    /// chain — the standard `run(equilibrate); run(sample); …` pattern.
    ///
    /// The count is single trial moves, NOT sweeps: each one is one proposal
    /// plus one accept/reject decision, so an N-atom system needs roughly N
    /// times more trials here than it would "one move per particle" sweeps.
    /// @param frame Host-resident system state, updated in place.
    /// @param n_trials Number of trial moves to attempt.
    void run(cpu::Frame& frame, int n_trials) {
        if (this->begin_run_(frame)) {
            // Publish the per-move changeset + the RNG so a Mover can reach
            // them (mirrors NeighborList publishing pair_list). Addresses are
            // stable.
            this->backend_->context().mc.proposal = &proposal_;
            this->backend_->context().mc.rng = &rng_;
            this->step_once(kBootstrapStep);
            this->start_outcome_(frame); // seed the policy's reference state
        }

        for (int s = 0; s < n_trials; ++s) {
            this->mutable_outer_status().reset();
            // frame.meta.get(kEnergy) still holds the LAST ACCEPTED energy here
            // (the trial evaluator hasn't run yet) — save it for rollback.
            const AccReal e_before = frame.meta.get(kEnergy);
            const int trial = this->next_step_();
            // ONE Transaction serves the whole chain, and only a Mover that
            // calls `begin()` clears it. Open every trial on a well-formed
            // EMPTY proposal so a Mover that proposes nothing (nothing left to
            // move, a draw it refuses) cannot leave the policy judging — and
            // the rollback splicing — the PREVIOUS trial's verb and patch.
            // `begin(None, …)` is two stores and a clear: it compiles no pack
            // plan, and any Mover that does propose re-begins over it.
            proposal_.begin(Transaction::Verb::None, frame);
            this->step_once(trial); // propose (mutate + record) → energy
            const Transaction::Verb verb = proposal_.verb;
            const Outcome outcome = this->decide_outcome_(frame, trial);
            tally_(verb, outcome);
            if (outcome == Outcome::Rollback) {
                proposal_.rollback(frame);
                // The frame must reflect the restored state, not the trial.
                frame.meta.get(kEnergy) = e_before;
            }
            if (this->poll_criteria_(s) != Verdict::Continue)
                break;
        }
    }

    /// @brief Configure the trial-move generator (mirrors MD `set_integrator`).
    /// Singular: re-call with the same Mover is idempotent, a different Mover
    /// throws. Binds the Mover at its `default_bindings()` (`propose` phase).
    /// @param m Mover component (e.g. `TranslateMove`, `MoveSelector`).
    /// @return `*this` for chaining.
    template <typename M>
        requires(!std::is_lvalue_reference_v<M>)
    McBase& set_mover(M&& m) {
        this->set_singular_default_owned_(std::forward<M>(m), mover_set_,
                                          "set_mover");
        return *this;
    }

    template <typename M>
    McBase& set_mover_ref(M& m) {
        this->set_singular_default_ref_(m, mover_set_, "set_mover_ref");
        return *this;
    }

    template <typename M>
    McBase& set_mover(M&) = delete;

    /// @brief Trial moves of verb `v` attempted across all `run()` calls.
    long long attempts(Transaction::Verb v) const {
        return attempts_[static_cast<int>(v)];
    }

    /// @brief Trial moves of verb `v` accepted across all `run()` calls.
    long long accepts(Transaction::Verb v) const {
        return accepts_[static_cast<int>(v)];
    }

    /// @brief Accepted fraction of verb-`v` moves, in [0,1] (0 if none).
    double acceptance_ratio(Transaction::Verb v) const {
        const long long a = attempts(v);
        return a > 0 ? static_cast<double>(accepts(v)) / static_cast<double>(a)
                     : 0.0;
    }

  protected:
    /// @brief Construct the shared loop with one reproducible RNG stream.
    ///        Derived preset constructors attach their `OutcomePolicy` via
    ///        `set_outcome` and may register default movers.
    /// @param seed Seed for the single RNG stream shared by the Movers and
    ///        the acceptance policy (interleaved draws — one sequence per
    ///        seed, so trajectories are reproducible).
    explicit McBase(std::uint64_t seed) : rng_(seed) {
        // Zero the accumulators at the head of the trial-energy phase so the
        // evaluator accumulates onto a clean slate each move (mirrors MD's
        // pair at pre_force −1000 / −999). MC scores energies, but an
        // accumulating force source still writes kAtom_f* on every trial, and
        // an unzeroed force channel grows without bound over a long chain
        // (overflow / NaN, and a misleading force field for any observer).
        this->on(ATV::sampling::Phase::energy)
            .add(cpu::ZeroForces{}, -1000)
            .add(cpu::ZeroEnergy{}, -999);
    }

    /// @brief Enforce that a pair evaluator that needs a neighbor list has
    ///        one, and that nothing renumbers rows inside a trial.
    void install_defaults() override {
        const auto s = this->scan_pending_infra_();
        this->throw_if_missing_neighbor_list_(
            s, "MC",
            "Add `mc.set_neighbor(cpu::md::NeighborList<>{cutoff, skin, 1, 0, "
            "true})`.");
        // A trial move is a transaction on frame ROWS, and the neighbour
        // rebuild that would renumber them runs INSIDE the trial (bound at
        // `sampling::Phase::energy`, ahead of the evaluator). Left enabled,
        // `atom_modify sort` fires at its cadence part-way through a trial and
        // `Transaction::rollback` then splices a saved row over an unrelated
        // atom — two atoms land on top of each other, the sampler's reference
        // energy stops describing the configuration it holds, and the chain
        // freezes silently. See `NeighborProvider::forbid_row_reordering`.
        this->for_each_pending_mut([](cpu::Component& c) {
            if (cpu::NeighborProvider* np = c.neighbor_provider())
                np->forbid_row_reordering();
        });
    }

    /// @brief The trial `energy` phase runs energy-only; every other phase
    ///        still asks for forces.
    ///
    /// A trial move is scored, then kept or discarded — the Metropolis test
    /// reads an energy and never a gradient, so every force assembled at
    /// `energy` is wasted work. For a nested QM method it is the *dominant* cost
    /// of the trial, and the veto ANDs down into that method's own stages
    /// (`ExecServices::forces_requested`), so it skips its whole gradient.
    ///
    /// Per SLOT, not per driver: `propose` stays force-capable (a Mover may bias
    /// its displacement on the current force) and so does `end_of_step`, where
    /// MD-shaped observers and recorders live.
    /// @param p Phase whose slot is being stamped.
    /// @return False for `sampling::Phase::energy`, true otherwise.
    bool phase_wants_forces(ATV::sampling::Phase p) const override {
        return p != ATV::sampling::Phase::energy;
    }

    /// @brief A Mover to propose with, and something that can score the
    ///        proposal. Nested method-drivers count as HF energy sources via
    ///        cross-cast.
    void validate_components() const override {
        this->require_mover();
        this->require_energy();
    }

    /// @brief Reserved step index for the energy-only bootstrap pass (a Mover
    ///        proposes nothing on a negative step).
    static constexpr int kBootstrapStep = -1;

    /// The run's single reproducible RNG stream. Owned here because it is a
    /// run-level service: the Mover draws from it (via `ctx.mc.rng`) and the
    /// acceptance policy draws from it too, interleaved in one sequence.
    std::mt19937_64 rng_;
    mc::Transaction proposal_; ///< Reused per-move changeset (on the Context).

  private:
    /// @brief Count one decided trial into the per-verb tallies.
    void tally_(Transaction::Verb v, Outcome o) {
        ++attempts_[static_cast<int>(v)];
        if (o == Outcome::Commit)
            ++accepts_[static_cast<int>(v)];
    }

    static constexpr int kNVerbs = 5; ///< None/Displace/Insert/Delete/Box.
    long long attempts_[kNVerbs] = {0, 0, 0, 0, 0}; ///< Per-verb attempts.
    long long accepts_[kNVerbs] = {0, 0, 0, 0, 0};  ///< Per-verb accepts.
    std::optional<std::string> mover_set_;          ///< set_mover slot.
};

} // namespace mc
} // namespace cpu
} // namespace ATV
