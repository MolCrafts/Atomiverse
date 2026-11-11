/// @file cuda/dynamics/md.cuh
/// @brief cuda::md::MD -- molecular dynamics preset with LAMMPS-style phases.
///
/// `MD` is the CUDA molecular-dynamics preset: a `cuda::Driver<md::Phase>` that owns
/// a continuable `run(frame, n_steps)` and a build-time `validate_components()` contract
/// (≥ 1 `Integrator` and ≥ 1 plugin that `provides_force()`).
///
/// There is no `cuda::DynamicsDriver` category above it any more. **The step budget IS
/// the termination**, so MD attaches no default `Criterion` — but it honours every
/// handler the user attaches (`sim.add_criterion(cuda::StopOnNaN{})`, a wall-clock guard,
/// a temperature halt), which is a thing that could not be expressed at all while "stop
/// or not" was a private `if` inside a driver class.
///
/// The constructor adds only infrastructure components — it contributes
/// no runtime behaviour beyond the base. The Zero* initializers are
/// phase-agnostic (`cuda/md/init.cuh`); `MD` binds them explicitly to
/// `md::Phase` via `on(phase).add()`:
/// - ZeroForces at pre_force (order -1000) -- zeros fx, fy, fz.
/// - ZeroScalars at pre_force (order -999) -- zeros every system scalar
///   (energies, kinetic energy and stress, virial) in one memset.
/// - KineticEnergy at end_of_step (order -100) -- accumulates KE.
///
/// ## Execution model (shared with the CPU backend)
///
/// `MD` walks the same LAMMPS-granular `md::Phase` skeleton documented in
/// `docs/developer/md-execution-model.md` — every phase is always walked,
/// even when empty. This single-GPU driver binds work to `neighbor`
/// (NeighborList), `pre_force` (Zero*), `force` (pair/bonded),
/// `post_force` (Virial), `initial_integrate`/`final_integrate`
/// (integrator), and `end_of_step` (KE, thermo, observers). The
/// communication phases `pre_exchange` (atom migration) and
/// `pre_neighbor` (ghost-atom halo send), plus `post_integrate`, carry
/// no component on a single GPU — multi-GPU halo/migration is out of
/// scope per `msu-pe-validation.md` decision #1 — but the phases still
/// exist as documented "nothing happens here" slots, ready for a
/// future multi-GPU arm without a phase-skeleton change.
///
/// ## LAMMPS-shaped categorical setter API
///
/// `MD` exposes the same 11-method categorical surface as
/// `cpu::md::MD` (chain link `msu-pe-cuda-components-02`, Phase G.4):
/// `set_evaluator / set_evaluator / set_angle / set_dihedral / set_neighbor`
/// (singular, idempotent on same-class re-call, throw on
/// different-class re-call), `set_integrator` (singular via the
/// component's own `default_bindings()`), `set_thermostat /
/// set_barostat / set_observer / set_record` (plural — each call
/// appends, insertion order preserved), and `set_thermo(int every)`
/// (singular cadence; binds `cuda::Thermo{every}` at
/// `end_of_step @ +20`).
///
/// ## Auto-injection at build()
///
/// `MD::install_defaults()` runs *before* `validate_components()` (the order
/// `Driver::build()` fixes), so auto-injected components count toward validation.
/// Single-GPU subset of the CPU rules, keyed on `requires_system_stress` /
/// `provides_system_stress` and the neighbor protocol accessors:
///
/// - **Virial** — if any pending component `requires_system_stress()` and
///   none `provides_system_stress()`, a fresh `cuda::Virial{}` is added at
///   `post_force @ -50`.
/// - **NeighborList missing** — if any pending component reports
///   itself as a `NeighborConsumer` (pair-style evaluators do, via
///   `cuda::Pair`) without a registered `cuda::NeighborList`,
///   `build()` throws naming the offending evaluator.
///
/// Multi-GPU halo exchange auto-injection is **out of scope** for the
/// CUDA arm per `msu-pe-validation.md` fixed-scope decision #1.
///
/// Plan registration primitive is only `on(phase).add` — it is the
/// documented escape hatch for components without a categorical slot,
/// and the additive setter surface must not break the existing
/// `tests/cuda/md/test_md.cu` paths that use it directly.
///
/// Usage:
/// @code{.cpp}
/// cuda::md::MD sim;
/// sim.set_evaluator(cuda::PairLJCut{1.0, 1.0, 3.0});
/// sim.set_neighbor(cuda::NeighborList{3.0});
/// sim.set_integrator(cuda::md::VelocityVerlet{0.001});
/// sim.set_thermo(5);
/// sim.build();
/// sim.run(frame, 10000);
/// @endcode
///
/// @see cuda::Driver for the loop primitives + the handler slot
/// @see md::Phase for the LAMMPS-style phase sequence
/// @see cpu::md::MD for the CPU sibling driver with the same API surface
#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/component/timestep_protocol.cuh"
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/dynamics/init.cuh"
#include "atomiverse/cuda/dynamics/kinetic_energy.cuh"
#include "atomiverse/cuda/observe/stress.cuh"
#include "atomiverse/cuda/record/thermo.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief CUDA MD preset — a `cuda::Driver<md::Phase>` with the LAMMPS-style
///        phase sequence and the 11-method categorical setter API that mirrors
///        `cpu::md::MD`.
///
/// The constructor binds the zeroing and kinetic-energy infrastructure components, then
/// user code feeds the rest of the simulation in via the categorical setters (or the raw
/// `add()` / `on(phase).add()` escape hatch).
///
/// @see cuda::minimizer::Minimize for the energy minimizer preset
class MD : public cuda::Driver<ATV::md::Phase> {
  public:
    /// @brief Construct MD driver, binding the phase-agnostic Zero* and
    ///        KineticEnergy infrastructure components to `md::Phase`.
    MD() {
        on(ATV::md::Phase::pre_force).add(ZeroForces{}, -1000);
        on(ATV::md::Phase::pre_force).add(ZeroScalars{}, -999);
        on(ATV::md::Phase::end_of_step).add(KineticEnergy{}, -100);
    }

    /// @brief Advance the system for `n_steps` MD steps.
    ///
    /// **Continuable.** Consecutive `run()` calls on the same `frame` extend one
    /// trajectory — components are built once on the first call, and the step index
    /// passed to components is cumulative across calls. This matches `cpu::md::MD` so
    /// that:
    ///   - `FixLangevin`'s per-step RNG keying (which mixes the cumulative `ctx.step`
    ///     into `base_seed`) does NOT repeat the same z-stream every `run()` call — a
    ///     critical FDT-correctness requirement; if `s` reset to 0 each call, the
    ///     Gaussian impulse sequence repeats with period `n_steps`, breaking
    ///     equipartition (T_eq diverges to ~3.5× target observed in the `Thermalises`
    ///     test).
    ///   - `NoseHoover`/`NoseHooverNPT` extended variables persist.
    ///   - Recorder cadence keys (modulo `frequency`) stay continuous, and a
    ///     recorder's step axis continues: each run first writes the current
    ///     state (LAMMPS `Output::setup`) unless that step was already
    ///     written, then labels each end-of-step write with the completed
    ///     step count.
    /// A different `frame` ⇒ genuinely independent simulation: tear the previous run
    /// down and restart the counter.
    ///
    /// **Termination.** The step budget is the termination — MD attaches no default
    /// handler. Any handler the user attached is polled after every step, and the loop
    /// breaks on anything but `Verdict::Continue` (that includes a component having set
    /// `status.stop_requested`, which `poll_criteria_` folds in).
    ///
    /// @param frame Host-resident system state; transferred H2D, advanced
    ///        on the device, retrieved D2H, updated in place.
    /// @param n_steps Step budget [dimensionless step count].
    BigInt run(cpu::Frame& frame, BigInt n_steps) {
        if (n_steps < 0)
            throw std::invalid_argument("MD step budget must be nonnegative");
        if (live_ && frame_ptr_ != &frame) {
            // The previous successful chunk already wrote back. Retrieving
            // again here could overwrite caller edits made since that return.
            live_ = false;
            frame_ptr_ = nullptr;
            this->backend_->shutdown();
            this->built_plugins_.clear();
        }
        try {
            if (!live_) {
                setup_pass(frame);
                frame_ptr_ = &frame;
                step_counter_ = 0;
                live_ = true;
            } else {
                this->flush_new_mounts_(&frame);
                this->backend_->refresh_inputs(frame);
            }
            this->backend_->prepare_run(step_counter_);
            // LAMMPS Run::command → Output::setup: every run writes the
            // current state first (recorders on cadence, never a step
            // twice), so a recorder's frame `s` is the state after `s` steps.
            this->backend_->setup_output(step_counter_);
            // The final step evaluates the energy the retrieved Frame reports.
            this->backend_->set_report_step(step_counter_ + n_steps - 1);
            BigInt iterations = 0;
            bool early = false;
            for (BigInt s = 0; s < n_steps; ++s) {
                this->mutable_outer_status().reset();
                step_once(step_counter_);
                ++step_counter_;
                ++iterations;
                // A device fault froze the state at its step; issuing more
                // steps only burns GPU time. retrieve() reports it.
                if (this->backend_->fault_observed())
                    break;
                if (this->poll_criteria_(s) != Verdict::Continue) {
                    early = s + 1 < n_steps;
                    break;
                }
            }
            this->backend_->set_report_step(-1);
            // A criterion stopped before the reported step: evaluate the
            // current state once so the retrieved scalars belong to it.
            if (early)
                this->backend_->evaluate_current(step_counter_);
            this->backend_->retrieve(frame);
            return iterations;
        } catch (...) {
            if (this->backend_) {
                this->backend_->mark_inputs_dirty();
                if (!live_) {
                    try {
                        this->backend_->shutdown();
                    } catch (...) {
                    }
                    this->built_plugins_.clear();
                }
            }
            // Never retrieve a failed device evaluation. At an input-boundary
            // failure, retry restores host inputs without rebuilding integrators.
            // This is not rollback of thermostat/RNG state after a failure in
            // the middle of an integration step or a multi-step chunk.
            throw;
        }
    }

    ~MD() {
        if (this->backend_) {
            try {
                this->backend_->shutdown();
            } catch (...) {
            }
        }
    }

    // ─── Categorical setter API (LAMMPS-shaped) ─────────────────────
    //
    // set_evaluator / set_neighbor / set_observer / set_record come from
    // cuda::Driver (PhaseTraits<md::Phase>).

    /// @brief Configure the bond force evaluator (LAMMPS `bond_style`).
    ///        Bound at `md::Phase::force @ 0`.
    template <typename B>
    MD& set_bond(B&& b) {
        set_singular_at_(std::forward<B>(b), bond_set_, "set_bond",
                         ATV::md::Phase::force, 0);
        return *this;
    }

    /// @brief Configure the angle force evaluator (LAMMPS
    ///        `angle_style`). Bound at `md::Phase::force @ 0`.
    template <typename A>
    MD& set_angle(A&& a) {
        set_singular_at_(std::forward<A>(a), angle_set_, "set_angle",
                         ATV::md::Phase::force, 0);
        return *this;
    }

    /// @brief Configure the dihedral force evaluator (LAMMPS
    ///        `dihedral_style`). Bound at `md::Phase::force @ 0`.
    template <typename D>
    MD& set_dihedral(D&& d) {
        set_singular_at_(std::forward<D>(d), dihedral_set_, "set_dihedral",
                         ATV::md::Phase::force, 0);
        return *this;
    }

    /// @brief Configure the time integrator (LAMMPS `fix … nve|nvt|npt`).
    ///        Bound via the integrator's own `default_bindings()`.
    template <typename I>
    MD& set_integrator(I&& integ) {
        set_singular_default_(std::forward<I>(integ), integrator_set_,
                              "set_integrator");
        return *this;
    }

    /// @brief Borrow the time integrator (same singular slot as
    ///        `set_integrator`). The Python CUDA driver keeps a
    ///        `NoseHooverNPT` so it can set the temperature-ramp span.
    /// @param integ Integrator; must outlive this driver.
    template <typename I>
    MD& set_integrator_ref(I& integ) {
        const std::string n = integ.name();
        if (integrator_set_) {
            if (*integrator_set_ == n)
                return *this;
            throw std::invalid_argument(
                "set_integrator already configured with '" + *integrator_set_
                + "', rejected re-call with '" + n + "'");
        }
        integrator_set_ = n;
        this->bind_defaults_ref(integ);
        return *this;
    }

    /// @brief Add a thermostat (LAMMPS `fix … langevin / temp/berendsen`).
    /// Plural: each call appends; insertion order preserved.
    template <typename T>
    MD& set_thermostat(T&& t) {
        this->bind_defaults(std::forward<T>(t));
        return *this;
    }

    /// @brief Borrow a thermostat so the caller can set its ramp span.
    /// The Python CUDA driver keeps the Langevin and calls
    /// `set_run_span` before a continued cool.
    template <typename T>
    MD& set_thermostat_ref(T& t) {
        this->bind_defaults_ref(t);
        return *this;
    }

    /// @brief Add a barostat (LAMMPS `fix … press/berendsen`). Plural.
    /// `install_infra_()` auto-injects a `cuda::Virial{}` if a
    /// Barostat-role component is present and no `virial_stress` is
    /// already registered.
    template <typename Bs>
    MD& set_barostat(Bs&& b) {
        this->bind_defaults(std::forward<Bs>(b));
        return *this;
    }

    /// @brief Configure thermo output (LAMMPS `thermo N` +
    ///        `thermo_style custom`). Singular: a re-call with a different
    ///        cadence or keyword list throws.
    /// Binds `cuda::Thermo{every, keywords}` at `end_of_step @ +20`; its
    /// keywords decide which steps evaluate the energy and the virial.
    /// @param every Print cadence [steps]. `0` prints the first and last
    ///        step of each run only (LAMMPS `thermo 0`).
    /// @param keywords `thermo_style custom` keywords; empty selects the
    ///        LAMMPS default `step temp pe ke etotal`.
    MD& set_thermo(int every, std::vector<std::string> keywords = {}) {
        if (thermo_every_) {
            if (*thermo_every_ == every && thermo_keywords_ == keywords)
                return *this;
            throw std::invalid_argument(
                "set_thermo already configured with cadence "
                + std::to_string(*thermo_every_)
                + "; rejected a re-call with a different cadence or style");
        }
        cuda::Thermo thermo{every, keywords}; // validates the keywords
        thermo_every_ = every;
        thermo_keywords_ = std::move(keywords);
        this->on(ATV::md::Phase::end_of_step).add(std::move(thermo), +20);
        return *this;
    }

  protected:
    /// @brief Integrator + force plugin (`provides_force()`, including Langevin).
    void validate_components() const override {
        this->require_integrator();
        this->require_force();
    }

    /// @brief Auto-injection pass — single-GPU subset of the CPU rules.
    ///
    /// Runs before `validate_components()` (see `Driver::build()`). Two checks,
    /// both keyed on `requires_system_stress` / `provides_system_stress` and
    /// the `as_neighbor_consumer()` / `as_neighbor_provider()` protocol
    /// accessors — no `name()` string matching, no family type list:
    ///   1. If any pending component `requires_system_stress()` and none
    ///      `provides_system_stress()`, add a fresh `cuda::Virial{}` at
    ///      `post_force @ -50`.
    ///   2. If any pending component is a `NeighborConsumer` without any
    ///      pending component being a `NeighborProvider`, throw naming
    ///      the first offender.
    ///
    /// **No default handler.** A time budget is not a policy — MD stops when the steps
    /// run out, and anything else the user wants is a `Criterion` they attach.
    void install_defaults() override {
        const auto s = this->scan_pending_infra_();

        if (s.needs_stress && !s.has_virial_stress) {
            // `cuda::Virial` has no default_bindings(); bind it
            // explicitly to `post_force @ -50` — the slot the CPU
            // sibling `cpu::md::Virial::default_bindings()` uses,
            // where pair / bonded virials are already accumulated and
            // the velocities are post-integrator-kick so Σ m·v·v
            // reflects the appropriate time-slice.
            this->on(ATV::md::Phase::post_force).add(cuda::Virial{}, -50);
        }

        // Energy and virial are evaluated per step on demand (each
        // component's `demand()`, LAMMPS ev_set); no build-time gating.

        this->throw_if_missing_neighbor_list_(
            s, "cuda::md::MD",
            "Add `sim.set_neighbor(cuda::NeighborList{cutoff})`.");

        negotiate_timestep_();
    }

    /// @brief Push the integrator's dt to every `TimestepConsumer` before any
    ///        component `build()` — the CUDA mirror of
    ///        `cpu::Driver::negotiate_timestep_` (same rules, same messages).
    ///
    /// Providers and consumers are deduplicated by identity because the walk
    /// visits bindings (VelocityVerlet binds twice). Two providers on
    /// different clocks throw naming both; a consumer that requires a dt
    /// throws when nothing provides one; an unusable dt (not finite, <= 0) is
    /// refused before the first push.
    void negotiate_timestep_() {
        struct PublishedDt {
            std::string owner;
            double dt_fs;
        };
        std::vector<PublishedDt> published;
        std::unordered_set<const cuda::TimestepProvider*> seen_providers;
        this->for_each_pending([&](const cuda::Component& c) {
            const cuda::TimestepProvider* p = c.as_timestep_provider();
            if (!p || !seen_providers.insert(p).second)
                return;
            published.push_back({c.name(), p->timestep()});
        });
        for (std::size_t i = 1; i < published.size(); ++i) {
            const double a = published[0].dt_fs;
            const double b = published[i].dt_fs;
            const double diff = std::fabs(a - b);
            const double scale = std::max(std::fabs(a), std::fabs(b));
            const bool disagree =
                std::isfinite(diff) ? diff > 1e-12 * scale : !(a == b);
            if (disagree)
                throw std::runtime_error(
                    "MD::build(): conflicting MD timesteps — integrator '"
                    + published[0].owner + "' publishes " + std::to_string(a)
                    + " fs, integrator '" + published[i].owner + "' publishes "
                    + std::to_string(b)
                    + " fs; exactly one clock may own the run");
        }
        std::vector<std::pair<cuda::TimestepConsumer*, std::string>> consumers;
        std::unordered_set<const cuda::TimestepConsumer*> seen_consumers;
        this->for_each_pending_mut([&](cuda::Component& c) {
            cuda::TimestepConsumer* tc = c.as_timestep_consumer();
            if (tc && seen_consumers.insert(tc).second)
                consumers.emplace_back(tc, c.name());
        });
        if (consumers.empty())
            return;
        if (!published.empty()
            && (!(published[0].dt_fs > 0.0)
                || !std::isfinite(published[0].dt_fs)))
            throw std::runtime_error("MD::build(): integrator '"
                                     + published[0].owner
                                     + "' publishes an unusable timestep "
                                     + std::to_string(published[0].dt_fs)
                                     + " fs (must be finite and > 0)");
        for (auto& [consumer, owner] : consumers) {
            if (published.empty()) {
                if (consumer->timestep_required())
                    throw std::runtime_error(
                        "MD::build(): '" + owner
                        + "' requires the nuclear timestep, but no registered "
                          "component implements TimestepProvider");
                continue;
            }
            consumer->set_timestep(published[0].dt_fs);
        }
    }

    void clear_singular_on_unfix_(std::string_view name_or_id) override {
        auto clear_slot = [&](std::optional<std::string>& slot) {
            if (slot && *slot == name_or_id)
                slot.reset();
        };
        clear_slot(bond_set_);
        clear_slot(angle_set_);
        clear_slot(dihedral_set_);
        clear_slot(integrator_set_);
    }

    // ── Singular-setter helpers (CUDA side, mirroring cpu::Driver) ──

    /// @brief Singular setter at an explicit phase / order. Used for
    /// components without `default_bindings()` (NeighborList) or where
    /// the categorical API pins a specific slot.
    template <typename C>
    void set_singular_at_(C&& comp, std::optional<std::string>& slot,
                          const char* setter_name, ATV::md::Phase phase,
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
        this->on(phase).add(std::forward<C>(comp), order);
    }

    /// @brief Singular setter using the component's `default_bindings()`.
    template <typename C>
    void set_singular_default_(C&& comp, std::optional<std::string>& slot,
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
        this->bind_defaults(std::forward<C>(comp));
    }

  private:
    // ── Categorical setter slots ────────────────────────────────────
    std::optional<std::string> bond_set_;
    std::optional<std::string> angle_set_;
    std::optional<std::string> dihedral_set_;
    std::optional<std::string> integrator_set_;
    std::optional<int> thermo_every_;
    std::vector<std::string> thermo_keywords_;

    // ── Continuation bookkeeping (folded down from cuda::DynamicsDriver) ──

    /// @brief True between first `run()` and driver destruction —
    /// components are live on the device, must not be rebuilt.
    bool live_ = false;
    /// @brief Frame bound by the live run; a different frame restarts.
    cpu::Frame* frame_ptr_ = nullptr;
    /// @brief Cumulative MD step index across all `run()` calls.
    BigInt step_counter_ = 0;
};

} // namespace md
} // namespace cuda
} // namespace ATV
