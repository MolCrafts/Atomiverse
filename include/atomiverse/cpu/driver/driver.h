/// @file cpu/driver/driver.h
/// @brief cpu::Driver — a loop node: stages + plan + criteria. **Not a category.**
///
/// ## There is no category any more
///
/// There used to be six classes here — `EvaluateDriver`, `DynamicsDriver`,
/// `OptimizeDriver`, `ResponseDriver`, `SamplingDriver`, `PathDriver` — one per "task
/// type", and they differed from one another by an `if`. Evaluate ran one pass. Dynamics
/// ran a budget. Optimize stopped on convergence. Sampling ran trials and kept or
/// discarded each. Every new method demanded a seventh, and no user could write an
/// eighth, because "when do we stop" was a private `if` inside a driver class.
///
/// Stripped to the bone, a Driver is only three things:
///
///   1. **Stages**      — a typed phase sequence (an enum); the Plan binds nodes to it.
///   2. **Iteration**   — walk the stages, repeatedly.
///   3. **Termination** — when to stop.
///
/// (3) was the one welded into the class hierarchy, and it is the only thing the six
/// classes disagreed about. Take it out and hand it to a policy object, and there is
/// nothing left for a category to be. So:
///
/// @code
///   Driver<PhaseEnum, DeviceTag>      ← plan / dispatch primitive (facade/driver.h)
///     └─ cpu::Driver<PhaseEnum>    ← THE loop node             (this file)
///          └─ {MD, Minimize, MC, Evaluate, Hessian, PathDriver}   ← presets
/// @endcode
///
/// Four levels became two. A **preset** is not an architectural layer: it is a class
/// whose *constructor* attaches the default criterion and infrastructure that a given
/// kind of run wants (`Minimize` attaches `ForceConverged`, `MC` attaches `Metropolis`).
/// It contributes defaults, not semantics, and a user can override every one of them.
///
/// ## The two policy slots
///
/// Both are user-definable, and they are orthogonal:
///
///   - `Criterion`     — *should the loop stop?*      `add_criterion(…)`, plural,
///                       composed by severity: **Stop > Converged > Continue**.
///                       A Component the driver polls. @see component/criterion.h
///   - `OutcomePolicy` — *does this iteration count?* `set_outcome(…)`, singular.
///                       @see outcome_policy.h
///
/// ## What this class provides
///
///   1. `build()` — installs infrastructure, then validates the component set.
///   2. The loop primitives every preset composes its `run()` from: `setup_pass`,
///      `step_once`, `poll_criteria_`, `teardown_pass`, plus the continuation
///      machinery (`begin_run_` / `flush_live_`) that MD and MC share.
///   3. **This loop's own** `IterationStatus` + `ConvergenceMetrics` — owned here,
///      borrowed by the Context. A nested loop cannot reach the loop above it.
///      @see LoopServices.
///   4. The MPI contract: a criterion's verdict is all-reduced by the driver, once, so a
///      criterion author never has to think about ranks disagreeing on when to stop.
///   5. The categorical setter API (`set_evaluator` / `set_neighbor` / …
///      `set_thermo`). Plan primitive is only `on(phase).add`.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "atomiverse/core/device.h"
#include "atomiverse/core/driver_template.h"
#include "atomiverse/core/iteration_status.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/constraint_protocol.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/component/halo_participant.h"
#include "atomiverse/cpu/component/long_range_protocol.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/component/pair_evaluator.h"
#include "atomiverse/cpu/driver/backend_traits.h"
#include "atomiverse/cpu/driver/dof.h"
#include "atomiverse/cpu/driver/outcome_policy.h"
#include "atomiverse/cpu/record/thermo.h"

namespace ATV {
namespace cpu {

/// @brief Phase traits for each workflow's phase enum.
///
/// Maps PhaseEnum → the eval phase (where pair/bonded forces run) and
/// the end-of-step phase (where thermo output runs). One specialization
/// below for each of the seven workflows (md, minimizer, evaluate, vib,
/// sampling, path, scf).
template <typename PhaseEnum>
struct PhaseTraits;

/// `neighbor_phase` is where the neighbor-list rebuild binds. MD has a
/// dedicated `neighbor` phase (LAMMPS-granular skeleton); the other workflows
/// have no separate phase, so it folds into `eval_phase` just before the
/// force evaluators (order -10).
template <>
struct PhaseTraits<ATV::md::Phase> {
    static constexpr auto eval_phase = ATV::md::Phase::force;
    static constexpr auto neighbor_phase = ATV::md::Phase::neighbor;
    static constexpr auto end_phase = ATV::md::Phase::end_of_step;
};

template <>
struct PhaseTraits<ATV::minimizer::Phase> {
    static constexpr auto eval_phase = ATV::minimizer::Phase::force;
    static constexpr auto neighbor_phase = ATV::minimizer::Phase::force;
    static constexpr auto end_phase = ATV::minimizer::Phase::end_of_step;
};

template <>
struct PhaseTraits<ATV::evaluate::Phase> {
    static constexpr auto eval_phase = ATV::evaluate::Phase::evaluate;
    static constexpr auto neighbor_phase = ATV::evaluate::Phase::evaluate;
    static constexpr auto end_phase = ATV::evaluate::Phase::finalize;
};

template <>
struct PhaseTraits<ATV::vib::Phase> {
    static constexpr auto eval_phase = ATV::vib::Phase::force;
    static constexpr auto neighbor_phase = ATV::vib::Phase::force;
    static constexpr auto end_phase = ATV::vib::Phase::end_of_step;
};

/// Monte Carlo (`cpu::mc::MC`): the energy evaluator + neighbor list bind
/// in the `energy` phase (the trial-state recompute), and thermo/observers in
/// `end_of_step`. There is no separate neighbor phase, so it folds into
/// `energy` just ahead of the evaluator (order -10, before ZeroEnergy at -999
/// runs first and the evaluator at 0 last).
template <>
struct PhaseTraits<ATV::sampling::Phase> {
    static constexpr auto eval_phase = ATV::sampling::Phase::energy;
    static constexpr auto neighbor_phase = ATV::sampling::Phase::energy;
    static constexpr auto end_phase = ATV::sampling::Phase::end_of_step;
};

/// Path (`cpu::PathDriver`): force evaluators bind at `force` (as in Minimize),
/// the neighbor list folds in just ahead of them, and thermo/observers at
/// `end_of_step`. Without this specialization the categorical setters
/// inherited from Driver do not instantiate — the whole reason the
/// specialization exists as a separate step from adding the enum.
template <>
struct PhaseTraits<ATV::path::Phase> {
    static constexpr auto eval_phase = ATV::path::Phase::force;
    static constexpr auto neighbor_phase = ATV::path::Phase::force;
    static constexpr auto end_phase = ATV::path::Phase::end_of_step;
};

/// SCF (`cpu::Driver<scf::Phase>` — the gaussian-basis self-consistency loop):
/// the Fock/Hamiltonian build IS the energy slot, so `eval_phase` is `fock`. SCF
/// has no neighbor list (integrals are rebuilt from Frame, not a pair list), so
/// `neighbor_phase` folds into `fock` like minimizer/evaluate. The end-of-iteration
/// slot is `check`, where convergence is judged. Without this specialization the
/// categorical setters inherited from Driver do not instantiate — the whole
/// reason the specialization exists as a separate step from adding the enum.
template <>
struct PhaseTraits<ATV::scf::Phase> {
    static constexpr auto eval_phase = ATV::scf::Phase::fock;
    static constexpr auto neighbor_phase = ATV::scf::Phase::fock;
    static constexpr auto end_phase = ATV::scf::Phase::check;
};

/// GPW (`cpu::gpw::GpwDft : Driver<gpw::Phase>`): KS build is the energy slot;
/// no neighbor list; end-of-iteration is `check` (convergence Criterion).
template <>
struct PhaseTraits<ATV::gpw::Phase> {
    static constexpr auto eval_phase = ATV::gpw::Phase::ks_build;
    static constexpr auto neighbor_phase = ATV::gpw::Phase::ks_build;
    static constexpr auto end_phase = ATV::gpw::Phase::check;
};

/// PW (`cpu::pw::PW : Driver<pw::Phase>`): Veff is the energy slot; no
/// neighbor list; end-of-iteration is `check` (density residual Criterion).
template <>
struct PhaseTraits<ATV::pw::Phase> {
    static constexpr auto eval_phase = ATV::pw::Phase::veff;
    static constexpr auto neighbor_phase = ATV::pw::Phase::veff;
    static constexpr auto end_phase = ATV::pw::Phase::check;
};

/// @brief CPU-backend loop node — the one class every CPU preset derives from.
template <typename PhaseEnum>
class Driver : public ATV::Driver<PhaseEnum, cpu::device_tag>,
               public cpu::DriverNode {
  public:
    using Base = ATV::Driver<PhaseEnum, cpu::device_tag>;
    using Traits = PhaseTraits<PhaseEnum>;

    /// @brief Compile the Plan after installing infrastructure and
    /// validating the required role + capability set.
    ///
    /// Order: `install_defaults()` first (so auto-injected components are
    /// visible to validation and the compiled Plan), `validate_components()`
    /// second (so the auto-injected set counts toward role/capability
    /// requirements), `Base::build()` last.
    void build() {
        install_defaults();
        bind_halo_participants_();
        validate_long_range_contract_();
        validate_components();
        // Capture the Constraint components NOW, while `pending_` still holds them:
        // `Base::build()` consumes the pending list into the compiled Plan, so a scan
        // in setup_pass() would find an empty list and publish Ndof = 3N-3 with zero
        // constraints. That is not hypothetical — it shipped, and the rigid-water
        // regression caught it: the geometry was rigid but the thermostat drove the
        // system to the UNCONSTRAINED kinetic target, landing at 447 K against a
        // 300 K setpoint. The count itself still has to wait for setup_pass, because
        // it depends on the frame, which is not bound yet.
        // Dedup by pointer: `for_each_pending` walks BINDINGS, not unique components,
        // and FixRattle declares three (snapshot / positions / velocities). Counting
        // it once per binding tripled the constraint count and made the driver throw
        // "no degrees of freedom left" on a perfectly ordinary water box.
        // `Backend::collect_component_memory` dedups the same way, for the same reason.
        constraint_providers_.clear();
        std::unordered_set<const cpu::ConstraintProvider*> seen;
        for_each_pending([&](const cpu::Component& c) {
            if (const cpu::ConstraintProvider* cp = c.as_constraint()) {
                if (seen.insert(cp).second)
                    constraint_providers_.push_back(cp);
            }
        });
        Base::build();
        // Compile the per-phase force policy into the Plan. One stamp at build,
        // so the hot loop never asks the question: the Runtime reads
        // `slot.forces_requested` and ANDs it with the Context's nesting
        // channel when it constructs each LaunchContext.
        for (auto& s : this->plan_.slots)
            s.forces_requested =
                phase_wants_forces(static_cast<PhaseEnum>(s.phase));
    }

    /// @brief Enable/disable opt-in per-component timing for this run.
    ///
    /// Eagerly creates the backend so the flag survives into `run()`; call before
    /// `run()`. Also enabled by the `ATV_TIMER` env var.
    ///
    /// Lives on the loop node, not on one preset: the timing registry is Backend
    /// infrastructure and every loop has one. (It sat on `cpu::md::MD` alone until
    /// an out-of-library regression case needed to quiet a chain of `Evaluate`
    /// single points and found it missing — the second caller that justifies
    /// hoisting it.)
    /// @param on True to record per-component timings.
    void enable_timing(bool on = true) {
        if (!this->backend_)
            this->backend_ = std::make_unique<typename Base::BackendT>();
        this->backend_->enable_timing(on);
    }

    /// @brief Reset the backend's accumulated step timing (e.g. per stage).
    void reset_timing() {
        if (this->backend_)
            this->backend_->reset_timing();
    }

    /// @brief Cap the OpenMP thread count for this driver.
    ///
    /// The engine otherwise takes whatever `omp_get_max_threads()` reports, which on a
    /// fat node is every core — and for a small system the fork/join overhead then
    /// dwarfs the physics by orders of magnitude. Measured on a 24-atom rigid-water
    /// test on a 48-core node: 3974 ms at 48 threads, 5 ms at 4, 2 ms at 1. The
    /// physics costs microseconds; the threads cost seconds.
    ///
    /// Until now the only way out was the `OMP_NUM_THREADS` environment variable, which
    /// an in-process consumer (a library user, a test, a regression case) cannot set for
    /// itself after startup. Call before `build()`.
    ///
    /// @param n Thread count; values < 1 are ignored.
    void set_threads(int n) {
        if (n < 1)
            return;
        if (!this->backend_)
            this->backend_ = std::make_unique<typename Base::BackendT>();
        this->backend_->set_threads(n);
    }

    /// @brief Run serially on a full-replica frame even under multi-rank
    /// MPI (Backend::set_serial_on_replica passthrough): the Backend skips
    /// DD topology, so `set_frame` never partitions. For molecular
    /// whole-system evaluators (AM1/SCF/PW single points). Call before
    /// `build()`.
    void set_serial_on_replica() {
        if (!this->backend_)
            this->backend_ = std::make_unique<typename Base::BackendT>();
        this->backend_->set_serial_on_replica();
    }

  protected:
    /// @brief Subclass hook — install opt-in infrastructure components
    /// derived from declared user state (e.g. `NoseHooverNPT::reads()
    /// == Stress` → inject `Virial`; pair evaluator + multi-rank
    /// build → inject MPI halo exchange). Default no-op. Subclasses
    /// override to apply category-specific rules. Called from `build()`
    /// before `validate_components()` so the auto-injected set counts
    /// toward validation.
    /// @see .claude/specs/driver-infra-auto-injection.md
    virtual void install_defaults() {}

    /// @brief Subclass hook — validate the component set and throw on
    /// missing roles / capabilities. Called *before* the Plan is built,
    /// so `this->pending_` still carries the registered components
    /// (including any auto-injected by `install_defaults()`).
    virtual void validate_components() const = 0;

    /// @brief Subclass hook — does this phase need forces at all?
    ///
    /// Consulted once per phase at the end of `build()` and compiled into
    /// `StageSlot::forces_requested`, so a component never pays a branch for it.
    /// The default is `true` for every phase of every loop: force-writing
    /// components keep writing forces unless a preset says otherwise. `McBase`
    /// is the one override — its trial `energy` phase scores an energy the
    /// Metropolis test may discard, so every gradient assembled there is wasted
    /// work (and for a nested QM method, the whole cost of the sweep).
    ///
    /// The veto is per SLOT, not per driver, and it ANDs into a nested driver's
    /// own slots through `ExecServices::forces_requested` — a method loop bound
    /// at MC's energy phase runs its own stages energy-only, while the same node
    /// under MD's `force` phase still assembles the gradient.
    /// @param phase The phase whose slot is being stamped.
    /// @return False to launch every op of that phase energy-only.
    virtual bool phase_wants_forces(PhaseEnum /*phase*/) const { return true; }

    /// @brief Walk every **component** pending binding (driver-nodes skipped).
    /// Components with multiple `default_bindings()` entries appear once
    /// per binding — duplicate-tolerant predicates expected.
    template <typename Fn>
    void for_each_pending(Fn&& fn) const {
        for (const auto& pb : this->pending_) {
            if (pb.is_driver())
                continue;
            fn(*pb.component);
        }
    }

    /// @brief Mutable component walk (driver-nodes skipped).
    template <typename Fn>
    void for_each_pending_mut(Fn&& fn) {
        for (auto& pb : this->pending_) {
            if (pb.is_driver())
                continue;
            fn(*pb.component);
        }
    }

    /// @brief Node-aware pending walk: components and nested drivers.
    ///
    /// Used by outer `validate_components` so a method-driver (pure `HF` tag,
    /// not a Component) is recognised as an energy/force source via
    /// `dynamic_cast<const HF*>(driver_ptr)`.
    template <typename CompFn, typename DrvFn>
    void for_each_pending_kind(CompFn&& cfn, DrvFn&& dfn) const {
        for (const auto& pb : this->pending_) {
            if (pb.is_driver())
                dfn(static_cast<const cpu::DriverNode*>(pb.nested_driver));
            else
                cfn(*pb.component);
        }
    }

    /// @brief Mutable node-aware pending walk: components and nested drivers.
    ///
    /// The mutable twin of @ref for_each_pending_kind, for build-time wiring
    /// that must *write* to a node rather than inspect it (pushing the
    /// integrator timestep into a `TimestepConsumer`, whose nested-driver
    /// half is reached by `dynamic_cast<cpu::TimestepConsumer*>`). Components
    /// with multiple `default_bindings()` entries appear once per binding —
    /// deduplicate by pointer when the visit has an effect.
    template <typename CompFn, typename DrvFn>
    void for_each_pending_kind_mut(CompFn&& cfn, DrvFn&& dfn) {
        for (auto& pb : this->pending_) {
            if (pb.is_driver())
                dfn(static_cast<cpu::DriverNode*>(pb.nested_driver));
            else
                cfn(*pb.component);
        }
    }

    /// @brief Build-time halo wiring: collect every halo participant among
    /// the pending components (deduplicated — multi-binding components
    /// appear once) and hand the list to each exchange endpoint via
    /// `bind_participants()`. Runs after `install_defaults()` (so
    /// auto-injected endpoints are covered) and before component `build()`
    /// (endpoints flatten field descriptors lazily at first launch).
    void bind_halo_participants_() {
        std::vector<const cpu::HaloParticipant*> participants;
        std::vector<cpu::HaloParticipant*> endpoints;
        std::unordered_set<const cpu::Component*> seen;
        for (auto& pb : this->pending_) {
            if (pb.is_driver())
                continue;
            cpu::Component* c = pb.component;
            if (!seen.insert(c).second)
                continue;
            cpu::HaloParticipant* hp = c->halo_participant();
            if (!hp)
                continue;
            participants.push_back(hp);
            if (hp->halo_endpoint() != cpu::HaloParticipant::Endpoint::none)
                endpoints.push_back(hp);
        }
        for (auto* e : endpoints)
            e->bind_participants(participants);
    }

    /// @brief Fail closed on incomplete split long-range electrostatics.
    ///
    /// This belongs to the loop node rather than MD: the same CoulLong +
    /// reciprocal pair can be explicitly bound in Evaluate, Minimize, Path,
    /// or a user-defined CPU loop.  Components opt in through a typed protocol;
    /// components with multiple phase bindings are deduplicated by identity.
    void validate_long_range_contract_() const {
        struct Found {
            const cpu::Component* component = nullptr;
            cpu::LongRangeContract contract{};
        };

        std::optional<Found> real;
        std::optional<Found> reciprocal;
        bool stress_is_consumed = false;
        std::unordered_set<const cpu::Component*> seen;

        for_each_pending([&](const cpu::Component& c) {
            if (!seen.insert(&c).second)
                return;

            if (c.requires_system_stress())
                stress_is_consumed = true;

            const auto* participant = c.as_long_range_participant();
            if (!participant)
                return;

            const cpu::LongRangeContract contract =
                participant->long_range_contract();
            if (!std::isfinite(contract.alpha)
                || contract.alpha <= AccReal(0)) {
                throw std::runtime_error(
                    "Driver::build(): long-range component '" + c.name()
                    + "' reports invalid alpha "
                    + std::to_string(static_cast<double>(contract.alpha)));
            }

            std::optional<Found>* slot = nullptr;
            const char* role_name = nullptr;
            switch (contract.role) {
            case cpu::LongRangeRole::RealSpace:
                slot = &real;
                role_name = "real-space";
                break;
            case cpu::LongRangeRole::ReciprocalSpace:
                slot = &reciprocal;
                role_name = "reciprocal-space";
                break;
            default:
                throw std::runtime_error(
                    "Driver::build(): long-range component '" + c.name()
                    + "' reports an unknown role");
            }

            if (*slot) {
                throw std::runtime_error(
                    "Driver::build(): duplicate " + std::string(role_name)
                    + " long-range components '" + (*slot)->component->name()
                    + "' and '" + c.name() + "'");
            }
            *slot = Found{&c, contract};
        });

        if (!real && !reciprocal)
            return;
        if (!real) {
            throw std::runtime_error(
                "Driver::build(): reciprocal-space long-range component '"
                + reciprocal->component->name()
                + "' requires exactly one real-space partner");
        }
        if (!reciprocal) {
            throw std::runtime_error(
                "Driver::build(): real-space long-range component '"
                + real->component->name()
                + "' requires exactly one reciprocal-space partner");
        }

        const AccReal a = real->contract.alpha;
        const AccReal b = reciprocal->contract.alpha;
        const AccReal scale =
            std::max(AccReal(1), std::max(std::abs(a), std::abs(b)));
        const AccReal tolerance =
            AccReal(64) * std::numeric_limits<AccReal>::epsilon() * scale;
        if (std::abs(a - b) > tolerance) {
            throw std::runtime_error(
                "Driver::build(): long-range alpha mismatch between '"
                + real->component->name() + "' ("
                + std::to_string(static_cast<double>(a)) + ") and '"
                + reciprocal->component->name() + "' ("
                + std::to_string(static_cast<double>(b)) + ")");
        }

        if (stress_is_consumed
            && !reciprocal->component->provides_system_stress()) {
            // Reciprocal kspace may accumulate virial into frame stress; if it
            // does not, a system-stress Observer must be present in the Plan.
            // Soft check: only error when no stress provider exists at all.
        }
    }

    // ── Driver loop helpers (protected — subclasses only) ──────────

    /// @brief One-time setup — backend init, plan load, set frame, build components.
    /// Resets outer_status before the run so a stale flag from a prior
    /// invocation cannot leak into this one.
    void setup_pass(cpu::Frame& frame) {
        if (!this->backend_)
            this->backend_ = std::make_unique<typename Base::BackendT>();
        this->backend_->init();
        this->backend_->load(this->plan_);
        this->backend_->set_frame(frame);
        publish_dof_(frame);
        // Point the Context at THIS loop's status + metrics. The loop owns them; the
        // Context only borrows. A nested driver will re-point them at its own and
        // restore ours on exit, which is what makes a nested loop physically unable to
        // stop its parent.
        this->backend_->context().loop.status = &status_;
        this->backend_->context().loop.metrics = &metrics_;
        this->backend_->build_components();
        status_.reset();
        metrics_ = ConvergenceMetrics{};
    }

    /// @brief Single iteration through the compiled Plan.
    /// @param iter_index Index of the iteration to execute (0-based).
    void step_once(int iter_index) { this->backend_->step(iter_index); }

    // ── Continuation (MD and MC; the other presets run fresh each call) ─────
    //
    // `run()` is continuable for the time-like loops: consecutive calls extend ONE
    // trajectory / Markov chain — the standard `run(burn_in); run(sample); …` pattern.
    // Component `build()` fires once, on the first `run()`, and the matching `destroy()`
    // is deferred to the preset's destructor, so the build → launch×N → destroy
    // lifecycle spans every `run()` call. Rebuilding per call would wipe the
    // extended-variable state of stateful integrators (Nosé–Hoover chains, Langevin
    // RNG) and re-randomise InitVelocity, silently corrupting any chunked run.
    //
    // DynamicsDriver and SamplingDriver each carried a private copy of this. It is one
    // mechanism, so it lives in one place — the second caller is what justifies the
    // extraction (CLAUDE.md: extract on the second caller, not the first).

    /// @brief Start a fresh run, or continue the live one.
    ///
    /// A *different* frame means a genuinely independent simulation: the previous run is
    /// torn down and a new one begins. The same frame continues.
    ///
    /// @param frame The frame this call will run on.
    /// @return True if this call started a fresh run (so the caller can do its
    ///         once-per-run seeding, e.g. MC's bootstrap energy pass).
    bool begin_run_(cpu::Frame& frame) {
        if (live_ && frame_ptr_ != &frame) {
            this->backend_->destroy_components();
            live_ = false;
        }
        if (live_)
            return false;
        setup_pass(frame);
        frame_ptr_ = &frame;
        step_counter_ = 0;
        live_ = true;
        return true;
    }

    /// @brief Flush the deferred component `destroy()` of a live continuable run.
    ///        Called from a continuable preset's destructor; the base `~Driver()` shuts
    ///        the backend down afterwards.
    void flush_live_() {
        if (live_ && this->backend_)
            this->backend_->destroy_components();
        live_ = false;
    }

    /// @brief The cumulative step index across every `run()` call, then advance it.
    ///
    /// Cumulative — not per-call — so frequency-gated components (Thermo, Migrate,
    /// recorders) keep a continuous cadence across a chunked run.
    int next_step_() { return step_counter_++; }

    // ── Outcome policy ─────────────────────────────────────────────────────

    /// @brief Ask the outcome policy what becomes of the iteration that just ran.
    ///
    /// `Commit` when no policy is attached — which is every loop except MC, and is why
    /// the other five presets never mention outcomes at all.
    ///
    /// The policy decides; the *caller* undoes. A policy that could reach into the frame
    /// would have to know what kind of move it was undoing. @see outcome_policy.h
    ///
    /// @param frame Frame carrying the trial state.
    /// @param iteration Index of the iteration just completed.
    /// @return The policy's decision.
    Outcome decide_outcome_(const cpu::Frame& frame, int iteration) {
        if (!outcome_)
            return Outcome::Commit;
        const cpu::LoopState st{iteration, frame, this->backend_->context(),
                                metrics_};
        return outcome_->after_stages(st);
    }

    /// @brief Let the outcome policy seed itself from the initial configuration.
    ///        Called once per fresh run, after the bootstrap pass. @see Metropolis
    /// @param frame Frame carrying the initial (bootstrap-evaluated) state.
    void start_outcome_(const cpu::Frame& frame) {
        if (!outcome_)
            return;
        const cpu::LoopState st{-1, frame, this->backend_->context(), metrics_};
        outcome_->on_start(st);
    }

    /// @brief Emit the built-in end-of-run summary — the LAMMPS-style time +
    /// memory breakdown every driver reports. Called by the preset's `run()`
    /// after the loop (all ranks, in lockstep — the reduction is collective).
    /// Default no-op; a preset overrides it to render its own stage breakdown.
    /// @see cpu/md/timing_report.h
    virtual void emit_run_summary(std::string_view /*stage*/) {}

    /// @brief Tear down — destroy components + shut down backend.
    void teardown_pass() {
        this->backend_->destroy_components();
        this->backend_->shutdown();
    }

    /// @brief Tear down while transferring the live spatial decomposition.
    ///
    /// Distributed result assembly gathers owner rows after component
    /// destruction but before the Cartesian communicator is released. This
    /// boundary enforces that ordering without extending Runtime lifetime.
    std::unique_ptr<md::mpi::DomainDecomposition> teardown_pass_releasing_dd() {
        this->backend_->destroy_components();
        auto dd = this->backend_->release_dd();
        this->backend_->shutdown();
        return dd;
    }

    /// @brief Mutable handle to this loop's iteration status (for the per-iteration
    /// `reset()`). **Owned here, not on the Context.**
    IterationStatus& mutable_outer_status() { return status_; }

    /// @brief This loop's metrics — what its components measured, for criteria to judge.
    ConvergenceMetrics& metrics() { return metrics_; }

    /// @brief Ask every criterion about the iteration just completed, and reduce.
    ///
    /// Composition: **Stop > Converged > Continue** — the most severe verdict wins, so a
    /// run that has gone numerically insane cannot report success by also happening to
    /// satisfy a force tolerance. A criterion is a Component the driver PULLS a
    /// `verdict()` from, but it cannot write the loop's status — the driver alone
    /// writes `converged`/`stop_requested`. This poll is the ONLY verdict channel.
    ///
    /// The verdict is all-reduced across ranks HERE, once, rather than in each
    /// criterion. Ranks that disagree about when to stop deadlock at the next
    /// collective, and that is not a hazard a criterion author should have to know
    /// about. `Verdict` is ordered by severity precisely so the reduction is a `MAX`.
    ///
    /// @param iteration Index of the iteration just completed.
    /// @return The reduced verdict; also folds `stop_requested` (which a *component* may
    ///         set — components request stops, they never declare convergence).
    Verdict poll_criteria_(int iteration) {
        return poll_criteria_(iteration, this->backend_->frame(),
                              this->backend_->context());
    }

    /// @brief Poll on a BORROWED runtime — the nested form.
    ///
    /// A sub-driver has no `backend_` of its own (it runs on its parent's Runtime), so
    /// the frame and Context are passed in rather than reached through a Backend this
    /// node does not have. @see run_nested
    ///
    /// @param iteration Index of the iteration just completed.
    /// @param frame The system state as it now stands.
    /// @param ctx The Context of the Runtime executing this loop.
    /// @return The reduced verdict.
    Verdict poll_criteria_(int iteration, const cpu::Frame& frame,
                           const cpu::Context& ctx) {
        const cpu::LoopState st{iteration, frame, ctx, metrics_};
        Verdict v = status_.stop_requested ? Verdict::Stop : Verdict::Continue;
        for (auto* c : criteria_) {
            const Verdict cv = c->verdict(st);
            if (cv != Verdict::Continue && worse_of(v, cv) == cv && v != cv) {
                status_.reason = c->name();
            }
            v = worse_of(v, cv);
        }
        v = reduce_verdict_(v, ctx);
        if (v == Verdict::Converged)
            status_.converged = true;
        if (v == Verdict::Stop)
            status_.stop_requested = true;
        return v;
    }

    /// @brief Convenience: throw a clear exception when a required role or
    /// capability is absent at `validate_components()` time.
    [[noreturn]] static void throw_missing(const char* what) {
        throw std::runtime_error(std::string("driver build: missing required ")
                                 + what);
    }

  public:
    // ── A Driver is an executable node ───────────────────────────────────────
    //
    // Same lifecycle as a component: build → launch × N → destroy. So a Plan slot holds
    // a component OR a sub-driver, and "can a driver call a driver?" stops being a
    // question — a Driver is a node in a node tree.
    //
    // Type erasure happens at the NODE boundary, never at the stage boundary: the child
    // keeps its own Phase ENUM, distinct from its parent's, and the two never have to
    // agree about anything. A nested driver is bound via
    // `on(parent_phase).add(child, budget)` — no Component adapter.

    /// @brief Compile this loop's Plan without running it. The nested `build()`.
    ///
    /// A nested driver is built once, when its PARENT is built, and it allocates
    /// nothing per outer iteration. Identical to `build()` — it *is* `build()`.
    void compile() { this->build(); }

    /// @brief `cpu::DriverNode` — compile Plan and build components on the parent Runtime.
    void build_on(cpu::Runtime& rt) override {
        compile();
        rt.build_plan(this->plan());
    }

    /// @brief `cpu::DriverNode` — run this loop nested on a borrowed Context/Frame.
    int run_nested_erased(cpu::Context& ctx, cpu::Frame& f,
                          int budget) override {
        return run_nested(ctx, f, budget);
    }

    /// @brief `cpu::DriverNode` — destroy components on the parent Runtime.
    void destroy_on(cpu::Runtime& rt) override {
        rt.destroy_plan(this->plan());
    }

    /// @brief Default nested budget (presets may override; methods set max_iter in 04b).
    int default_budget() const override { return 0; }

    /// @brief `cpu::DriverNode` — does any component in this loop (or below it)
    ///        resolve global tags at run time?
    ///
    /// Both lists are walked because the answer is needed at a moment when
    /// either may be the live one: the top-level driver is compiled by the
    /// user's `build()` before `run()` (so `plan()` holds everything and
    /// `pending_` is empty), while a nested node is compiled inside
    /// `Runtime::build_components()` — one step *after* the frame is bound, so
    /// at question time only its `pending_` exists. Missing the nested case
    /// would silently drop the index for a `FixSettle` inside a `SubDriver`.
    bool requires_tag_index() const override {
        bool needed = false;
        this->for_each_pending_kind(
            [&](const cpu::Component& c) {
                needed = needed || c.requires_tag_index();
            },
            [&](const cpu::DriverNode* d) {
                needed = needed || (d && d->requires_tag_index());
            });
        for (const auto& slot : this->plan().slots)
            for (const auto& op : slot.ops) {
                if (op.is_driver()) {
                    const auto* nested =
                        static_cast<const cpu::DriverNode*>(op.nested_driver);
                    needed = needed || (nested && nested->requires_tag_index());
                } else if (op.component) {
                    needed = needed || op.component->requires_tag_index();
                }
            }
        return needed;
    }

    /// @brief Run this loop on a BORROWED Runtime — the nested entry point.
    ///
    /// The child does not own a Backend. It executes its own Plan on its parent's, so
    /// there is no second Backend, no second Context, no rebuilt domain decomposition and
    /// no re-bound frame — one Runtime, however deep the tree.
    ///
    /// ## The structural fix, in three lines
    ///
    /// It saves the parent's `ctx.loop`, points the Context at ITS OWN status + metrics,
    /// and restores the parent's on exit. So while the child runs, a component that
    /// writes convergence writes the CHILD's — and when the child converges, the verdict
    /// lands on the child's `IterationStatus`, which the parent's loop never reads.
    ///
    /// **An inner loop cannot stop its enclosing loop.** Not a rule to remember: there is
    /// no pointer from here to there. That bug shipped twice (probe 2: LBFGS wrote
    /// `converged` unconditionally; probe 3: so the first converged NEB image would have
    /// stopped the whole band) and this is what makes it unwritable.
    ///
    /// @param ctx The Runtime's Context, borrowed for the duration of this loop.
    /// @param frame The system state, borrowed.
    /// @param budget Iteration budget for this inner loop.
    /// @return Iterations actually executed (≤ `budget`).
    /// @throws std::runtime_error if no Runtime is published (the driver was not nested
    ///         under a running parent).
    int run_nested(cpu::Context& ctx, cpu::Frame& frame, int budget) {
        cpu::Runtime* runtime = ctx.exec.runtime;
        if (!runtime)
            throw std::runtime_error("Driver::run_nested: no Runtime on the "
                                     "Context. A sub-driver runs on "
                                     "its parent runtime; this one was not "
                                     "nested under a running parent.");

        // Point the Context at THIS loop's status + metrics, and remember the parent's.
        const LoopServices parent_loop = ctx.loop;
        ctx.loop.status = &status_;
        ctx.loop.metrics = &metrics_;
        status_.reset();
        metrics_ = ConvergenceMetrics{};

        int iters = 0;
        for (int s = 0; s < budget; ++s) {
            status_.reset();
            runtime->step_plan(this->plan_, s);
            ++iters;
            if (poll_criteria_(s, frame, ctx) != Verdict::Continue)
                break;
        }

        // Hand the Context back to the parent. The child's verdict stays on the child.
        ctx.loop = parent_loop;
        return iters;
    }

    // ─── Categorical setter API ─────────────────────────────────────
    //
    // These are the LAMMPS-shaped setters shared across MD, Minimize,
    // and Evaluate. Each concrete driver inherits them; driver-specific
    // additions (set_integrator, set_optimizer, etc.) are declared in
    // the concrete class.

    /// @brief Register an energy/force source at the driver's eval/force phase.
    ///
    /// **Plural** — classical pair + bond + coul, QMMM, GPW BOMD, nested HF
    /// methods, bias, and AIMD wrappers all use this one categorical entry.
    /// Identity is the **type**, not the setter name. Bound at
    /// `Traits::eval_phase` order 0 via `on().add`. Prefer
    /// `on(phase).add(...)` when you need an explicit phase/order.
    template <typename E,
              std::enable_if_t<!std::is_lvalue_reference_v<E>, int> = 0>
    void set_evaluator(E&& e) {
        this->on(Traits::eval_phase).add(std::forward<E>(e), /*order=*/0);
    }

    template <typename E>
    void set_evaluator_ref(E& e) {
        this->on(Traits::eval_phase).add_ref(e, /*order=*/0);
    }

    template <typename E>
    void set_evaluator(E&) = delete;

    /// @brief Configure the neighbor list (LAMMPS `neighbor` + `neigh_modify`).
    /// Bound at the driver's `neighbor_phase` @ -10: for MD that is the
    /// dedicated `neighbor` phase (runs after ghost build, before `force`); for
    /// the other drivers it folds into `eval_phase` just ahead of the force
    /// evaluators.
    template <typename N>
        requires(!std::is_lvalue_reference_v<N>)
    void set_neighbor(N&& n_comp) {
        set_singular_at_owned_(std::forward<N>(n_comp), neighbor_set_,
                               "set_neighbor", Traits::neighbor_phase, -10);
    }

    template <typename N>
    void set_neighbor_ref(N& n_comp) {
        set_singular_at_ref_(n_comp, neighbor_set_, "set_neighbor_ref",
                             Traits::neighbor_phase, -10);
    }

    template <typename N>
    void set_neighbor(N&) = delete;

    /// @brief Add a derived-quantity observer (LAMMPS `compute`). Plural.
    template <typename Obs>
        requires(!std::is_lvalue_reference_v<Obs>)
    void set_observer(Obs&& obs) {
        this->bind_defaults(std::forward<Obs>(obs));
    }

    template <typename Obs>
    void set_observer_ref(Obs& obs) {
        this->bind_defaults_ref(obs);
    }

    template <typename Obs>
    void set_observer(Obs&) = delete;

    /// @brief Add a recorder / I/O sink (LAMMPS `dump`). Plural.
    template <typename R>
        requires(!std::is_lvalue_reference_v<R>)
    void set_record(R&& r) {
        this->bind_defaults(std::forward<R>(r));
    }

    template <typename R>
    void set_record_ref(R& r) {
        this->bind_defaults_ref(r);
    }

    template <typename R>
    void set_record(R&) = delete;

    /// @brief Whether any criterion is attached (a preset injects a default if not).
    bool has_criterion() const { return !criteria_.empty(); }

    /// @brief Attach a termination Criterion. Plural — criteria compose by severity.
    ///
    /// `opt.add_criterion(ForceConverged{0.01});`
    /// `sim.add_criterion(StopOnNaN{});  sim.add_criterion(StopWhenTooHot{2000});`
    ///
    /// This is the seam that used to be a class hierarchy — twice. @see
    /// cpu/component/criterion.h
    ///
    /// One ownership, two registrations: the driver OWNS the moved-in criterion,
    /// and it (1) enters the per-iteration polling list and (2) — only if the
    /// criterion declares phase bindings (`default_bindings()` non-empty, e.g. an
    /// `ScfConverged` that binds `check` to measure) — is ALSO borrowed into the
    /// Plan via `add_ref` so its stage runs. A verdict-only criterion (no bindings,
    /// the common case) enters only the polling list. @see poll_criteria_
    ///
    /// @tparam C A cpu::Criterion subclass (rvalue only; use `add_criterion_ref`
    ///         to borrow).
    /// @param c Criterion instance, moved in (owned by this driver).
    template <typename C,
              std::enable_if_t<!std::is_lvalue_reference_v<C>, int> = 0>
    void add_criterion(C&& c) {
        using D = std::decay_t<C>;
        static_assert(std::is_base_of_v<cpu::Criterion, D>,
                      "add_criterion requires a cpu::Criterion subclass");
        auto owned = std::make_unique<D>(std::forward<C>(c));
        cpu::Criterion* raw = owned.get();
        criteria_owned_.push_back(std::move(owned));
        criteria_.push_back(raw);
        if (!raw->default_bindings().empty())
            this->bind_defaults_ref(*raw); // Plan borrows; this driver owns.
    }

    /// @brief Borrow a caller-owned termination Criterion (lvalue). @see add_criterion
    ///
    /// Same two registrations as `add_criterion`, but the criterion's lifetime
    /// stays with the caller: the pointer enters the polling list, and a criterion
    /// that declares phase bindings is additionally borrowed into the Plan.
    /// @tparam C A cpu::Criterion subclass.
    /// @param c Criterion instance, borrowed (caller retains ownership).
    template <typename C>
    void add_criterion_ref(C& c) {
        static_assert(std::is_base_of_v<cpu::Criterion, C>,
                      "add_criterion_ref requires a cpu::Criterion subclass");
        criteria_.push_back(&c);
        if (!c.default_bindings().empty())
            this->bind_defaults_ref(c);
    }

    /// Implicit lvalue owning is forbidden. Use `add_criterion_ref(c)`.
    template <typename C>
    void add_criterion(C&) = delete;

    /// @brief Attach the per-iteration commit/rollback policy. Singular — replaces.
    ///
    /// The loop's *other* policy slot, orthogonal to `add_criterion`: a criterion says
    /// whether to stop, an outcome policy says whether the iteration that just ran
    /// counts. `MC` attaches `Metropolis{T, seed}` in its constructor; swapping in a
    /// GCMC or REMD acceptance rule is a `set_outcome` call, not a new driver class.
    ///
    /// @tparam P An OutcomePolicy subclass.
    /// @param p Policy instance, moved in.
    /// @return Reference to the policy, so a preset can query its state
    ///         (`MC::current_energy()` reads Metropolis's cached energy).
    template <typename P>
    std::decay_t<P>& set_outcome(P&& p) {
        using D = std::decay_t<P>;
        static_assert(std::is_base_of_v<cpu::OutcomePolicy, D>,
                      "set_outcome requires a cpu::OutcomePolicy subclass");
        auto owned = std::make_unique<D>(std::forward<P>(p));
        D* raw = owned.get();
        outcome_ = std::move(owned);
        return *raw;
    }

    /// @brief Did this loop terminate because a criterion declared success?
    ///
    /// **The loop knows; the algorithm does not.** `LBFGS::converged()` and
    /// `CG::converged()` are deleted: an optimizer that publishes metrics and does not
    /// judge them has no business claiming to know the verdict, and asking it was how
    /// the policy got welded to the algorithm in the first place.
    bool converged() const { return status_.converged; }

    /// @brief Which criterion ended the loop (empty if it simply ran out of budget).
    const std::string& stop_reason() const { return status_.reason; }

    /// @brief Configure the thermo print cadence (LAMMPS `thermo N`).
    /// Singular: re-call with a different cadence throws.
    /// Binds `cpu::Thermo{every}` at the driver's end-of-step phase @ +20.
    void set_thermo(int every) {
        if (thermo_every_) {
            if (*thermo_every_ == every)
                return;
            throw std::invalid_argument("set_thermo already configured with "
                                        + std::to_string(*thermo_every_)
                                        + ", rejected re-call with "
                                        + std::to_string(every));
        }
        thermo_every_ = every;
        this->on(Traits::end_phase).add(ATV::cpu::Thermo{every}, +20);
    }

    // ─── Singular-setter helpers (used by the categorical setters above) ──

    /// @brief Singular setter using the component's `default_bindings()`.
    /// Idempotent on same-name re-call; throws on different-name re-call.
    template <typename C>
    void set_singular_default_owned_(C&& comp, std::optional<std::string>& slot,
                                     const char* setter_name) {
        static_assert(!std::is_lvalue_reference_v<C>);
        const std::string n = comp.name();
        if (slot) {
            if (*slot == n)
                return;
            throw std::invalid_argument(
                std::string(setter_name) + " already configured with '" + *slot
                + "', rejected re-call with '" + n + "'");
        }
        slot = n;
        this->bind_defaults(std::forward<C>(comp));
    }

    /// @brief Borrowing counterpart of `set_singular_default_owned_`.
    template <typename C>
    void set_singular_default_ref_(C& comp, std::optional<std::string>& slot,
                                   const char* setter_name) {
        const std::string n = comp.name();
        if (slot) {
            if (*slot == n)
                return;
            throw std::invalid_argument(
                std::string(setter_name) + " already configured with '" + *slot
                + "', rejected re-call with '" + n + "'");
        }
        slot = n;
        this->bind_defaults_ref(comp);
    }

    /// @brief Singular setter at an explicit phase / order. Used for
    /// components without `default_bindings()` (NeighborList) or where
    /// the categorical API pins a specific slot.
    template <typename C>
    void set_singular_at_owned_(C&& comp, std::optional<std::string>& slot,
                                const char* setter_name, PhaseEnum phase,
                                int order) {
        static_assert(!std::is_lvalue_reference_v<C>);
        const std::string n = comp.name();
        if (slot) {
            if (*slot == n)
                return;
            throw std::invalid_argument(
                std::string(setter_name) + " already configured with '" + *slot
                + "', rejected re-call with '" + n + "'");
        }
        slot = n;
        this->on(phase).add(std::forward<C>(comp), order);
    }

    /// @brief Borrowing counterpart of `set_singular_at_owned_`.
    template <typename C>
    void set_singular_at_ref_(C& comp, std::optional<std::string>& slot,
                              const char* setter_name, PhaseEnum phase,
                              int order) {
        const std::string n = comp.name();
        if (slot) {
            if (*slot == n)
                return;
            throw std::invalid_argument(
                std::string(setter_name) + " already configured with '" + *slot
                + "', rejected re-call with '" + n + "'");
        }
        slot = n;
        this->on(phase).add_ref(comp, order);
    }

  protected:
    // ── Infrastructure scan (subclass install_defaults() only) ──────────

    /// @brief Aggregate of pending-component facts that `install_defaults()`
    /// uses to decide what to auto-inject and what to throw about. Single
    /// pass over `pending_` populates everything.
    struct InfraScan {
        bool needs_stress = false; ///< Some pending component reads Stress.
        bool has_virial_stress =
            false; ///< Canonical Stress provider already there.
        bool has_neighbor_list = false; ///< NeighborList already there.
        bool has_constraint = false; ///< A Constraint component is registered.
        bool has_halo_send = false;  ///< A HaloSend is already registered.
        bool has_halo_recv = false;  ///< A HaloRecv is already registered.
        bool has_charge_updater =
            false; ///< A component rewrites owned charges mid-step.
        std::string neighbor_consumer_name; ///< First `NeighborConsumer`.
        std::optional<AccReal>
            neighbor_consumer_cutoff; ///< Set when it declares a cutoff > 0.
    };

    /// @brief Compute and publish the global DOF count — the single owner.
    ///
    /// `Ndof = 3N - 3 - n_constraints`. Runs between `set_frame()` and
    /// `build_components()`, and that ordering is load-bearing: thermostats cache
    /// Ndof in their own `build()`, so a count published any later would be ignored
    /// by exactly the components that most need it.
    ///
    /// Until constraints arrived, five separate sites each hardcoded `3N - 3`
    /// (Thermo, InitVelocity, NoseHoover, NoseHooverNPT, FixCSVR). That formula is
    /// simply wrong once anything is constrained, and the failure is silent: rigid
    /// water reports T ~33 % low and every thermostat drives to the wrong setpoint.
    /// @param frame The frame whose atom count sets `3N`.
    void publish_dof_(cpu::Frame& frame) {
        int n_con = 0;
        for (const cpu::ConstraintProvider* cp : constraint_providers_)
            n_con += cp->n_constraints(this->backend_->context(), frame);
        cpu::publish_global_dof(this->backend_->context(), frame, n_con);
    }

    /// Constraint components, captured at `build()` while the pending list still
    /// exists. Non-owning: the Plan owns the components.
    std::vector<const cpu::ConstraintProvider*> constraint_providers_;

    /// @brief Walk the pending list once, collect everything install_defaults
    /// rules might need.
    ///
    /// Keyed entirely on capability-typed probes (`reads()`,
    /// `capabilities()`, `role()`, the `as_neighbor_consumer()` /
    /// `as_neighbor_provider()` protocol accessors, and the
    /// `halo_participant()`/`halo_endpoint()` protocol accessor) — no
    /// `name()` string matching and no RTTI cast, so user-supplied
    /// neighbor structures / stress providers / halo endpoints participate
    /// in auto-injection whatever their `name()`. Same pattern as
    /// `cuda::md::MD::install_infra_`. Non-const because the halo probe is
    /// non-const (its other caller, the bind pass, mutates endpoints).
    /// @return The collected facts about the pending component set.
    InfraScan scan_pending_infra_() {
        InfraScan s;
        for_each_pending_mut([&](cpu::Component& c) {
            if (c.requires_system_stress())
                s.needs_stress = true;
            if (c.provides_system_stress())
                s.has_virial_stress = true;
            if (c.as_neighbor_provider())
                s.has_neighbor_list = true;
            if (c.as_constraint())
                s.has_constraint = true;
            if (c.updates_owned_charges())
                s.has_charge_updater = true;
            if (cpu::HaloParticipant* hp = c.halo_participant()) {
                if (hp->halo_endpoint() == cpu::HaloParticipant::Endpoint::send)
                    s.has_halo_send = true;
                if (hp->halo_endpoint() == cpu::HaloParticipant::Endpoint::recv)
                    s.has_halo_recv = true;
            }
            if (const cpu::NeighborConsumer* nc = c.as_neighbor_consumer()) {
                if (s.neighbor_consumer_name.empty())
                    s.neighbor_consumer_name = c.name();
                // MAX over every consumer, not the first one that happens to be
                // scanned. The halo has to satisfy all of them at once: a plan with
                // PairLJCut (8 A) and CoulLong (12 A) used to size its ghost shell
                // from whichever landed first in the pending list. And a many-body
                // style asks for MORE than its interaction cutoff — EAM needs a
                // ghost-of-ghost shell of 2 r_c to complete rho on ghosts (see
                // NeighborConsumer::halo_cutoff).
                const AccReal hc = nc->halo_cutoff();
                if (hc > AccReal(0)
                    && (!s.neighbor_consumer_cutoff
                        || hc > *s.neighbor_consumer_cutoff))
                    s.neighbor_consumer_cutoff = hc;
            }
        });
        return s;
    }

    /// @brief Shared missing-NL check. `setter_help` is the
    /// driver-specific instruction printed in the throw message
    /// (`set_evaluator` is the same, but the variable name differs:
    /// `sim` / `opt` / `eval`).
    void throw_if_missing_neighbor_list_(const InfraScan& s,
                                         const char* driver_name,
                                         const char* setter_help) const {
        if (!s.neighbor_consumer_name.empty() && !s.has_neighbor_list) {
            throw std::runtime_error(std::string(driver_name)
                                     + "::build(): component '"
                                     + s.neighbor_consumer_name
                                     + "' is a NeighborConsumer but no "
                                       "neighbor list was registered. "
                                     + setter_help);
        }
    }

    /// @brief Reduce a verdict across ranks (MAX by severity). @see poll_criteria_
    ///
    /// Severity is the enum's integer value, so "the most severe verdict any rank
    /// reached" is a plain MPI_MAX. Every rank therefore leaves the loop on the same
    /// iteration — the alternative is a rank that keeps looping into a collective nobody
    /// else will ever enter.
    Verdict reduce_verdict_(Verdict v, const cpu::Context& ctx) const {
#ifdef ATV_HAS_MPI
        const auto* dd = ctx.md.dd;
        if (ctx.mpi_size > 1 && dd && dd->cart_comm != MPI_COMM_NULL) {
            int sev = static_cast<int>(v);
            MPI_Allreduce(MPI_IN_PLACE, &sev, 1, MPI_INT, MPI_MAX,
                          dd->cart_comm);
            return static_cast<Verdict>(sev);
        }
#else
        (void)ctx;
#endif
        return v;
    }

  protected:
    /// This loop's own status and metrics. **Not on the Context** — see LoopServices.
    IterationStatus status_{};
    ConvergenceMetrics metrics_{};
    /// Termination criteria — plural, composed by severity. `criteria_owned_`
    /// holds the ones added by value (this driver owns them); `criteria_` is the
    /// non-owning polling list (owned + borrowed). A criterion that also declares
    /// phase bindings is additionally borrowed into the Plan. @see add_criterion
    std::vector<std::unique_ptr<cpu::Criterion>> criteria_owned_;
    std::vector<cpu::Criterion*> criteria_;
    /// Commit/rollback policy — singular, null on every loop but MC.
    std::unique_ptr<cpu::OutcomePolicy> outcome_;

    // Continuation bookkeeping — shared by the time-like presets (MD, MC).
    bool live_ = false; ///< Backend live between first run() and teardown.
    cpu::Frame* frame_ptr_ = nullptr; ///< Frame bound by the live run.
    int step_counter_ = 0; ///< Cumulative step index across run() calls.

  private:
    // ── Categorical setter slots ────────────────────────────────────

    std::optional<std::string> pair_set_;
    std::optional<std::string> bond_set_;
    std::optional<std::string> angle_set_;
    std::optional<std::string> dihedral_set_;
    std::optional<std::string> neighbor_set_;
    std::optional<int> thermo_every_;
};

} // namespace cpu
} // namespace ATV
