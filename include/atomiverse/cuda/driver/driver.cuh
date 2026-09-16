/// @file cuda/driver/driver.cuh
/// @brief cuda::Driver — a loop node: stages + plan + handlers. **Not a category.**
///
/// ## There is no category any more
///
/// There used to be three classes here — `cuda::EvaluateDriver`, `cuda::DynamicsDriver`,
/// `cuda::OptimizeDriver` — one per "task type", and they differed from one another by
/// an `if`. Evaluate ran one pass. Dynamics ran a budget. Optimize stopped when a
/// component had written `outer_status.converged`. Every new method demanded a fourth,
/// and no user could write a fifth, because "when do we stop" was a private `if` inside
/// a driver class.
///
/// Stripped to the bone, a Driver is only three things:
///
///   1. **Stages**      — a typed phase sequence (an enum); the Plan binds nodes to it.
///   2. **Iteration**   — walk the stages, repeatedly.
///   3. **Termination** — when to stop.
///
/// (3) was the one welded into the class hierarchy, and it is the only thing the three
/// classes disagreed about. Take it out and hand it to a policy object, and there is
/// nothing left for a category to be. So:
///
/// @code
///   Driver<PhaseEnum, DeviceTag>       ← plan / dispatch primitive (facade/driver.h)
///     ├─ cpu::Driver<PhaseEnum>     ← the CPU loop node
///     └─ cuda::Driver<PhaseEnum>   ← THE loop node             (this file)
///          └─ {cuda::Evaluate, cuda::md::MD, cuda::minimizer::Minimize}  ← presets
/// @endcode
///
/// Four levels became two. A **preset** is not an architectural layer: it is a class
/// whose *constructor* (or `install_defaults()`) attaches the default handler and
/// infrastructure that a given kind of run wants (`Minimize` attaches `ForceConverged`;
/// `MD` attaches none — a step budget IS its termination). It contributes defaults, not
/// semantics, and a user can override every one of them.
///
/// ## What this class provides
///
///   1. `build()` — installs infrastructure (`install_defaults()`), then validates the
///      component set, then compiles the Plan.
///   2. The loop primitives every preset composes its `run()` from: `setup_pass`,
///      `step_once`, `poll_handlers_`, `teardown_pass`. The CUDA flavour of
///      `setup_pass` materializes the host frame onto the device (H2D) and
///      `teardown_pass` retrieves it back (D2H).
///   3. **This loop's own** `IterationStatus` + `LoopMetrics` — owned here, borrowed by
///      the Context (`cuda::LoopServices`). A nested loop cannot reach the loop above
///      it.
///   4. The termination policy slot: `add_criterion(…)`, plural, composed by severity —
///      **Stop > Converged > Continue**. @see cuda/driver/criterion.cuh
///
/// Categorical setters (`set_evaluator` / `set_neighbor` / `set_observer` /
/// `set_record`) live here, same as `cpu::Driver`. Presets add only
/// loop-specific slots (`set_integrator`, `set_optimizer`, `set_thermo`).
/// There is no MPI: the CUDA driver loop is single-rank, so
/// `poll_handlers_` reduces nothing. When a multi-GPU driver loop lands,
/// the reduction goes in `poll_handlers_` and no handler changes.
///
/// User code does not subclass this directly — it constructs a preset.
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
#include "atomiverse/core/plan.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/hf.cuh"
#include "atomiverse/cuda/component/integrator.cuh"
#include "atomiverse/cuda/component/long_range_protocol.cuh"
#include "atomiverse/cuda/component/optimizer.cuh"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/driver/backend_traits.cuh"
#include "atomiverse/cuda/driver/criterion.cuh"

namespace ATV {
namespace cuda {

/// @brief Phase traits for each CUDA workflow's phase enum.
///
/// Maps PhaseEnum → eval phase (pair/bonded / electronic energy) and
/// neighbor phase. MD has a dedicated `neighbor` phase; Evaluate and
/// Minimize fold the neighbor list into `eval_phase` at order -10.
template <typename PhaseEnum>
struct PhaseTraits;

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

/// @brief CUDA-backend loop node — shared by every CUDA preset.
///
/// @tparam PhaseEnum Phase enum type for the workflow (must end with a `COUNT`
///         sentinel).
template <typename PhaseEnum>
class Driver : public ATV::Driver<PhaseEnum, cuda::device_tag> {
  public:
    using Base = ATV::Driver<PhaseEnum, cuda::device_tag>;
    using Traits = PhaseTraits<PhaseEnum>;

    /// @brief Install infrastructure, validate the component set, compile the Plan.
    ///
    /// Order matters: `install_defaults()` first (so auto-injected components — and the
    /// preset's default handler — are visible to validation and to the compiled Plan),
    /// `validate_components()` second (so a build with a missing required component
    /// throws before the Plan is compiled), `Base::build()` last.
    void build() {
        install_defaults();
        validate_long_range_contract_();
        validate_components();
        Base::build();
    }

    // ── Termination policy ─────────────────────────────────────────────────

    /// @brief Attach a termination policy. Plural — handlers compose.
    ///
    /// `min.add_criterion(cuda::ForceConverged{0.01});`
    /// `sim.add_criterion(cuda::StopOnNaN{});`
    ///
    /// This is the seam that used to be a class hierarchy.
    /// @see cuda/driver/criterion.cuh
    /// @param h A Criterion subclass, moved in.
    /// @return Reference to self, for chaining.
    template <typename H>
    Driver& add_criterion(H&& h) {
        using D = std::decay_t<H>;
        static_assert(std::is_base_of_v<cuda::Criterion, D>,
                      "add_criterion requires a cuda::Criterion subclass");
        handlers_.push_back(std::make_unique<D>(std::forward<H>(h)));
        return *this;
    }

    /// @brief Whether any handler is attached (a preset injects a default if not).
    bool has_handler() const { return !handlers_.empty(); }

    /// @brief Did this loop terminate because a handler declared success?
    ///
    /// **The loop knows; the algorithm does not.** `CG::converged()` and
    /// `LBFGS::converged()` are deleted: an optimizer that publishes metrics and does
    /// not judge them has no business claiming to know the verdict, and asking it was
    /// how the policy got welded to the algorithm in the first place.
    bool converged() const { return status_.converged; }

    /// @brief Which handler ended the loop (empty if it simply ran out of budget).
    const std::string& stop_reason() const { return status_.reason; }

    /// @brief Unmount one plugin by `name()` (LAMMPS `unfix`).
    ///
    /// Destroys the node if it has been built, drops it from the live table,
    /// recompiles the Plan, and re-runs `validate_components()`.
    /// Call between steps or `run()` chunks, not from `launch()`.
    /// @param name_or_id Exact `Component::name()`.
    /// @throws std::runtime_error if none / several match, if called during
    ///         launch, or if the remaining set fails `validate_components`.
    void unfix(std::string_view name_or_id) {
        if (launching_) {
            throw std::runtime_error(
                "unfix: cannot unfix from inside launch(). Unfix between "
                "steps or run() chunks.");
        }
        std::vector<std::size_t> hits;
        for (std::size_t i = 0; i < this->pending_.size(); ++i) {
            cuda::Component* c = this->pending_[i].component;
            if (c && c->name() == name_or_id)
                hits.push_back(i);
        }
        if (hits.empty()) {
            std::string have;
            for (const auto& pb : this->pending_) {
                if (!pb.component)
                    continue;
                if (!have.empty())
                    have += ", ";
                have += pb.component->name();
            }
            throw std::runtime_error(
                "unfix: no plugin named '" + std::string(name_or_id)
                + "'. mounted: [" + have
                + "]. Remount with on(phase).add / set_evaluator.");
        }
        if (hits.size() > 1) {
            throw std::runtime_error(
                "unfix: name '" + std::string(name_or_id)
                + "' is not unique. Give plugins distinct name() values.");
        }
        cuda::Component* c = this->pending_[hits[0]].component;
        if (this->backend_ && built_plugins_.count(c)) {
            Plan<cuda::Component> one;
            one.slots.resize(1);
            one.slots[0].phase = 0;
            Operation<cuda::Component> op;
            op.component = c;
            one.slots[0].ops.push_back(op);
            this->backend_->destroy_plan(one);
            built_plugins_.erase(c);
        }
        this->pending_.erase(this->pending_.begin()
                             + static_cast<std::ptrdiff_t>(hits[0]));
        this->owned_components_.erase(
            std::remove_if(this->owned_components_.begin(),
                           this->owned_components_.end(),
                           [c](const std::unique_ptr<cuda::Component>& p) {
                               return p.get() == c;
                           }),
            this->owned_components_.end());
        clear_singular_on_unfix_(name_or_id);
        if (neighbor_set_ && *neighbor_set_ == name_or_id)
            neighbor_set_.reset();
        rebind_from_live_table_();
    }

    /// @brief Register an energy/force source at the driver's eval/force phase.
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

    /// @brief Neighbor list at `neighbor_phase` @ -10.
    template <typename N,
              std::enable_if_t<!std::is_lvalue_reference_v<N>, int> = 0>
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

    /// @brief Derived-quantity observer (uses `default_bindings()`). Plural.
    template <typename Obs,
              std::enable_if_t<!std::is_lvalue_reference_v<Obs>, int> = 0>
    void set_observer(Obs&& obs) {
        this->bind_defaults(std::forward<Obs>(obs));
    }
    template <typename Obs>
    void set_observer_ref(Obs& obs) {
        this->bind_defaults_ref(obs);
    }
    template <typename Obs>
    void set_observer(Obs&) = delete;

    /// @brief Recorder / I/O sink (uses `default_bindings()`). Plural.
    template <typename R,
              std::enable_if_t<!std::is_lvalue_reference_v<R>, int> = 0>
    void set_record(R&& r) {
        this->bind_defaults(std::forward<R>(r));
    }
    template <typename R>
    void set_record_ref(R& r) {
        this->bind_defaults_ref(r);
    }
    template <typename R>
    void set_record(R&) = delete;

  protected:
    /// @brief Subclass hook — attach default handlers / auto-inject infrastructure
    ///        derived from declared user state (`cuda::md::MD` injects `cuda::Virial`
    ///        when a component reads Stress; `cuda::minimizer::Minimize` attaches a
    ///        `ForceConverged` handler when the user attached none). Default no-op.
    ///        Called from `build()` before `validate_components()`, so anything it adds
    ///        counts toward validation.
    virtual void install_defaults() {}

    /// @brief Subclass hook — validate the component set and throw on
    ///        missing roles / capabilities.
    ///
    /// Called *before* the Plan is built, so `this->pending_` still
    /// carries the registered components.
    virtual void validate_components() const = 0;

    /// @brief Walk every pending component binding.
    ///
    /// Components with multiple `default_bindings()` entries appear once
    /// per binding — duplicate-tolerant predicates are expected.
    /// @tparam Fn Callable invoked as `fn(const cuda::Component&)`.
    /// @param fn Visitor applied to each pending component.
    template <typename Fn>
    void for_each_pending(Fn&& fn) const {
        for (const auto& pb : this->pending_)
            fn(*pb.component);
    }

    /// @brief Mutable variant of for_each_pending — visitor may mutate the
    /// component (e.g. `install_defaults()` gating per-step virial via
    /// `set_stress_required`). @tparam Fn Callable `fn(cuda::Component&)`.
    template <typename Fn>
    void for_each_pending_mut(Fn&& fn) {
        for (auto& pb : this->pending_)
            fn(*pb.component);
    }

    // ── Driver loop helpers (protected — presets only) ──────────────

    /// @brief One-time setup — backend init, plan load, H2D materialize,
    ///        component build.
    ///
    /// The CUDA flavour transfers the host frame onto the device
    /// (`materialize`, H2D) before building components.
    /// @param frame Host-resident system state, transferred to the device.
    void setup_pass(cpu::Frame& frame) {
        if (!this->backend_)
            this->backend_ = std::make_unique<typename Base::BackendT>();
        this->backend_->init();
        this->backend_->load(this->plan_);
        this->backend_->materialize(frame);
        // Point the Context at THIS loop's status + metrics. The loop owns them; the
        // Context only borrows. A nested driver would re-point them at its own, which
        // is what makes a nested loop physically unable to stop its parent.
        this->backend_->context().loop.status = &status_;
        this->backend_->context().loop.metrics = &metrics_;
        this->backend_->build_components();
        mark_built_from_pending_();
        status_.reset();
        metrics_ = LoopMetrics{};
    }

    /// @brief Single iteration through the compiled Plan.
    /// @param iter_index Zero-based iteration index passed to components.
    void step_once(int iter_index) {
        struct Guard {
            bool& flag;
            explicit Guard(bool& f) : flag(f) { flag = true; }
            ~Guard() { flag = false; }
        } guard(launching_);
        this->backend_->step(iter_index);
    }

    /// @brief Tear down — destroy components, retrieve results (D2H),
    ///        shut down the backend.
    /// @param frame Host-resident frame populated from the device.
    void teardown_pass(cpu::Frame& frame) {
        this->backend_->destroy_components();
        this->backend_->retrieve(frame);
        this->backend_->shutdown();
        built_plugins_.clear();
    }

    /// @brief Ask every handler about the iteration just completed, and reduce.
    ///
    /// Composition: **Stop > Converged > Continue** — the most severe verdict wins, so a
    /// run that has gone numerically insane cannot report success by also happening to
    /// satisfy a force tolerance.
    ///
    /// Also folds `status_.stop_requested`, which a *component* may set — components
    /// request stops, they never declare convergence: `IterationStatus::converged` is
    /// written here, from the handlers' reduced verdict, and nowhere else.
    ///
    /// Single-rank: unlike `cpu::Driver::poll_handlers_` there is no `MPI_Allreduce`
    /// of the severity, because the CUDA driver loop has no communicator.
    ///
    /// @param iteration Index of the iteration just completed.
    /// @return The reduced verdict.
    Verdict poll_handlers_(int iteration) {
        const cuda::LoopState st{iteration, this->backend_->frame(),
                                 this->backend_->context(), metrics_};
        Verdict v = status_.stop_requested ? Verdict::Stop : Verdict::Continue;
        for (auto& h : handlers_) {
            const Verdict hv = h->after_iteration(st);
            if (hv != Verdict::Continue && worse_of(v, hv) == hv && v != hv) {
                status_.reason = h->name();
            }
            v = worse_of(v, hv);
        }
        if (v == Verdict::Converged)
            status_.converged = true;
        if (v == Verdict::Stop)
            status_.stop_requested = true;
        return v;
    }

    /// @brief Mutable handle to this loop's iteration status (for the per-iteration
    /// `reset()`). **Owned here, not on the Context.**
    IterationStatus& mutable_outer_status() { return status_; }

    /// @brief Throw a clear exception when a required role or capability
    ///        is absent at `validate_components()` time.
    /// @param what Human-readable name of the missing requirement.
    [[noreturn]] static void throw_missing(const char* what) {
        throw std::runtime_error(std::string("driver build: missing required ")
                                 + what);
    }

    /// @brief Aggregate of pending-component facts for install_defaults /
    ///        require_*. Single pass over `pending_`.
    struct InfraScan {
        bool needs_stress = false; ///< Some plugin requires system stress.
        bool has_virial_stress =
            false; ///< A system-stress provider is present.
        bool has_neighbor_list = false; ///< A NeighborProvider is registered.
        bool has_energy = false;        ///< Some plugin provides_energy().
        bool has_force = false;         ///< Some plugin provides_force().
        bool has_integrator = false;    ///< An Integrator kind is registered.
        bool has_optimizer = false;     ///< An Optimizer kind is registered.
        bool has_hessian = false;       ///< Some plugin provides_hessian().
        std::string neighbor_consumer_name; ///< First NeighborConsumer.
    };

    /// @brief Const scan: provide/require facts plus protocol probes.
    ///
    /// Energy/force come from `provides_*`. HF methods that also inherit
    /// `Component` contribute via a single `HF` cross-cast. No family type
    /// list, no `name()==langevin`.
    InfraScan scan_pending_infra_() const {
        InfraScan s;
        for_each_pending([&](const cuda::Component& c) {
            if (c.provides_energy())
                s.has_energy = true;
            if (c.provides_force())
                s.has_force = true;
            if (c.provides_hessian())
                s.has_hessian = true;
            if (const auto* hf = dynamic_cast<const cuda::HF*>(&c)) {
                if (hf->provides_energy())
                    s.has_energy = true;
                if (hf->provides_force())
                    s.has_force = true;
            }
            if (dynamic_cast<const cuda::Integrator*>(&c))
                s.has_integrator = true;
            if (dynamic_cast<const cuda::Optimizer*>(&c))
                s.has_optimizer = true;
            if (c.requires_system_stress())
                s.needs_stress = true;
            if (c.provides_system_stress())
                s.has_virial_stress = true;
            if (c.as_neighbor_provider())
                s.has_neighbor_list = true;
            if (c.as_neighbor_consumer() && s.neighbor_consumer_name.empty())
                s.neighbor_consumer_name = c.name();
        });
        return s;
    }

    /// @brief Throw unless some plugin provides energy.
    void require_energy() const {
        if (!scan_pending_infra_().has_energy)
            throw_missing("energy source");
    }
    /// @brief Throw unless some plugin provides force.
    void require_force() const {
        if (!scan_pending_infra_().has_force)
            throw_missing("force source");
    }
    /// @brief Throw unless some plugin provides energy or force.
    void require_energy_or_force() const {
        const InfraScan s = scan_pending_infra_();
        if (!s.has_energy && !s.has_force)
            throw_missing("energy or force source");
    }
    /// @brief Throw unless an Integrator is registered.
    void require_integrator() const {
        if (!scan_pending_infra_().has_integrator)
            throw_missing("Integrator");
    }
    /// @brief Throw unless an Optimizer is registered.
    void require_optimizer() const {
        if (!scan_pending_infra_().has_optimizer)
            throw_missing("Optimizer");
    }
    /// @brief Throw unless some plugin assembles Hessian rows.
    void require_hessian() const {
        if (!scan_pending_infra_().has_hessian)
            throw_missing("hessian accumulator");
    }

    /// @brief Fail closed on incomplete split long-range electrostatics.
    void validate_long_range_contract_() const {
        struct Found {
            const cuda::Component* component = nullptr;
            LongRangeContract contract{};
        };

        std::optional<Found> real;
        std::optional<Found> reciprocal;
        std::unordered_set<const cuda::Component*> seen;

        for_each_pending([&](const cuda::Component& c) {
            if (!seen.insert(&c).second)
                return;
            const auto* participant = c.as_long_range_participant();
            if (!participant)
                return;
            const LongRangeContract contract =
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
            case LongRangeRole::RealSpace:
                slot = &real;
                role_name = "real-space";
                break;
            case LongRangeRole::ReciprocalSpace:
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
    }

    void throw_if_missing_neighbor_list_(const InfraScan& s,
                                         const char* driver_name,
                                         const char* setter_help) const {
        if (!s.neighbor_consumer_name.empty() && !s.has_neighbor_list) {
            throw std::runtime_error(std::string(driver_name)
                                     + "::build(): component '"
                                     + s.neighbor_consumer_name
                                     + "' is a pair evaluator but no "
                                       "neighbor list was registered. "
                                     + setter_help);
        }
    }

    void mark_built_from_pending_() {
        built_plugins_.clear();
        for_each_pending_mut(
            [&](cuda::Component& c) { built_plugins_.insert(&c); });
    }

    void flush_new_mounts_() {
        if (!this->backend_)
            return;
        for (auto& pb : this->pending_) {
            cuda::Component* c = pb.component;
            if (!c || built_plugins_.count(c))
                continue;
            Plan<cuda::Component> one;
            one.slots.resize(1);
            one.slots[0].phase = 0;
            Operation<cuda::Component> op;
            op.component = c;
            one.slots[0].ops.push_back(op);
            this->backend_->build_plan(one);
            built_plugins_.insert(c);
        }
        rebind_from_live_table_();
    }

    void rebind_from_live_table_() {
        PlanBuilder<cuda::Component> builder;
        for (auto& pb : this->pending_) {
            typename PlanBuilder<cuda::Component>::Entry e;
            e.phase = pb.phase;
            e.order = pb.order;
            e.component = pb.component;
            builder.add(e);
        }
        this->plan_ = builder.build(ATV::slot_count<PhaseEnum>());
        if (this->backend_)
            this->backend_->load(this->plan());
        validate_long_range_contract_();
        validate_components();
    }

    /// @brief Preset hook — clear a singular setter slot whose name was unfixed.
    virtual void clear_singular_on_unfix_(std::string_view /*name_or_id*/) {}

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
    /// This loop's own status and metrics. **Not on the Context** — see LoopServices.
    IterationStatus status_{};
    LoopMetrics metrics_{};
    /// Termination policies — plural, composed by severity. @see poll_handlers_
    std::vector<std::unique_ptr<cuda::Criterion>> handlers_;
    bool launching_ = false; ///< True inside `step_once` / `launch()`.
    std::unordered_set<cuda::Component*> built_plugins_;
    std::optional<std::string> neighbor_set_;
};

} // namespace cuda
} // namespace ATV
