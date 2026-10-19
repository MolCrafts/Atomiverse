/// @file core/driver_template.h
/// @brief Driver -- user-facing simulation facade with fluent API.
///
/// Backend-agnostic template: DeviceTag selects CPU or CUDA at compile time.
/// All method bodies are defined inline (template class). Explicit
/// instantiations live in src/cpu/driver/driver.cpp and src/cuda/driver/driver.cu.
///
/// Plan slots hold a Component*. A nested loop that is not already a
/// Component is mounted by `on(phase).add` / `add_ref` as a registration-
/// private slot. That slot is not a named public type. `Driver` does not
/// inherit `Component`.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/core/backend_traits.h"
#include "atomiverse/core/binding.h"
#include "atomiverse/core/device.h"
#include "atomiverse/core/plan.h"
#include "atomiverse/core/stage.h"

namespace ATV {

/// @brief Instance identity of a mounted plugin (pointer, not type or name).
struct PluginIdentity {
    const void* instance = nullptr;
    bool operator==(PluginIdentity o) const noexcept {
        return instance == o.instance;
    }
};

namespace detail {

/// @brief Unresolved binding collected before `Driver::build()`.
template <typename Component>
struct PendingBinding {
    int phase = 0; ///< Target phase index.
    int order = 0; ///< Ordering within the phase (lower runs earlier).
    Component* component = nullptr; ///< Plugin occupying this slot.

    /// @brief Identity of the plugin instance on this row.
    PluginIdentity identity() const noexcept { return {component}; }
};
} // namespace detail

/// @brief Fluent phase-binding helper returned by Driver::on().
template <typename PhaseEnum, typename DeviceTag>
class PhaseBuilder {
    using Component = typename backend_traits<DeviceTag>::component_type;

  public:
    /// @brief Register a component explicitly on the current phase (owned rvalue).
    template <typename Comp,
              std::enable_if_t<
                  std::is_base_of_v<Component, std::remove_reference_t<Comp>>
                      && !std::is_lvalue_reference_v<Comp>,
                  int> = 0>
    PhaseBuilder& add(Comp&& comp, int order = 0) {
        require_mutable_();
        using C = std::remove_reference_t<Comp>;
        auto ptr = std::make_unique<C>(std::forward<Comp>(comp));
        auto* raw = ptr.get();
        owned_->push_back(std::move(ptr));
        bindings_->push_back({ATV::slot_of(phase_), order, raw});
        return *this;
    }

    /// @brief Transfer ownership from an explicit unique pointer.
    template <typename Comp,
              std::enable_if_t<std::is_base_of_v<Component, Comp>, int> = 0>
    PhaseBuilder& add(std::unique_ptr<Comp> comp, int order = 0) {
        require_mutable_();
        if (!comp)
            throw std::invalid_argument("PhaseBuilder::add: null component");
        Component* raw = comp.get();
        owned_->push_back(std::move(comp));
        bindings_->push_back({ATV::slot_of(phase_), order, raw});
        return *this;
    }

    /// @brief Borrow a component whose lifetime is owned by the caller.
    ///
    /// Prefer Component binding when the type is dual-identity
    /// (Component + cpu::DriverNode), e.g. transitional MeanFieldMethod.
    template <typename Comp,
              std::enable_if_t<std::is_base_of_v<Component, Comp>, int> = 0>
    PhaseBuilder& add_ref(Comp& comp, int order = 0) {
        require_mutable_();
        bindings_->push_back({ATV::slot_of(phase_), order, &comp});
        return *this;
    }

    /// @brief Register an owned nested loop using the backend's node protocol.
    template <typename Nested,
              std::enable_if_t<
                  std::is_base_of_v<
                      typename backend_traits<DeviceTag>::driver_node_type,
                      std::remove_reference_t<Nested>>
                      && !std::is_base_of_v<Component,
                                            std::remove_reference_t<Nested>>
                      && !std::is_lvalue_reference_v<Nested>,
                  int> = 0>
    PhaseBuilder& add(Nested&& child, int order = 0, int budget = -1) {
        require_mutable_();
        using N = std::remove_reference_t<Nested>;
        auto owned = std::make_unique<N>(std::forward<Nested>(child));
        N* node = owned.get();
        return mount_nested(std::move(owned), node, order, budget);
    }

    /// @brief Borrow a nested loop; the caller owns its lifetime.
    template <
        typename Nested,
        std::enable_if_t<
            std::is_base_of_v<
                typename backend_traits<DeviceTag>::driver_node_type, Nested>
                && !std::is_base_of_v<Component, Nested>,
            int> = 0>
    PhaseBuilder& add_ref(Nested& child, int order = 0, int budget = -1) {
        return mount_nested(std::unique_ptr<Nested>{}, &child, order, budget);
    }

    /// Implicit lvalue borrowing is forbidden. Use `add_ref(comp)`.
    template <typename Comp>
    PhaseBuilder& add(Comp&, int = 0) = delete;

  private:
    template <typename PE, typename DT>
    friend class Driver;
    PhaseBuilder(std::vector<detail::PendingBinding<Component>>* b,
                 std::vector<std::unique_ptr<Component>>* o, PhaseEnum p,
                 const bool* const* lock)
        : bindings_(b), owned_(o), phase_(p), mutation_lock_(lock) {}
    std::vector<detail::PendingBinding<Component>>* bindings_;
    std::vector<std::unique_ptr<Component>>* owned_;
    PhaseEnum phase_;
    const bool* const* mutation_lock_;
    void require_mutable_() const {
        if (mutation_lock_ && *mutation_lock_ && **mutation_lock_)
            throw std::logic_error(
                "cannot change driver bindings during dispatch");
    }

    template <typename Node>
    PhaseBuilder& mount_nested(std::unique_ptr<Node> owned, Node* node,
                               int order, int budget) {
        require_mutable_();
        const int cap = budget < 0 ? node->default_budget() : budget;
        const auto [energy, force] =
            backend_traits<DeviceTag>::node_outputs(*node);
        auto slot = backend_traits<DeviceTag>::template nested_slot<Node>(
            std::move(owned), node, cap, energy, force);
        Component* raw = slot.get();
        owned_->push_back(std::move(slot));
        bindings_->push_back({ATV::slot_of(phase_), order, raw});
        return *this;
    }
};

/// @brief Simulation driver -- user-facing facade.
///
/// @tparam PhaseEnum Phase enum type (must end with COUNT sentinel).
/// @tparam DeviceTag Device tag (cpu::device_tag or cuda::device_tag).
template <typename PhaseEnum, typename DeviceTag>
class Driver {
  protected:
    using traits = backend_traits<DeviceTag>;
    using Component = typename traits::component_type;
    using BackendT = typename traits::backend_type;
    using frame_type = typename traits::frame_type;
    using Timer = typename traits::timer_type;

  public:
    Driver() = default;
    virtual ~Driver() { shutdown(); }
    Driver(const Driver&) = delete;
    Driver& operator=(const Driver&) = delete;

    Driver(Driver&& o) { *this = std::move(o); }

    Driver& operator=(Driver&& o) {
        if (this != &o) {
            if (mutation_lock_ || o.mutation_lock_)
                throw std::logic_error(
                    "cannot move a driver bound to a Runtime");
            shutdown();
            pending_ = std::move(o.pending_);
            owned_components_ = std::move(o.owned_components_);
            plan_ = std::move(o.plan_);
            backend_ = std::move(o.backend_);
            timer_ = std::move(o.timer_);
        }
        return *this;
    }

    /// @brief Begin explicit phase binding — the only public Plan-registration
    ///        primitive (with `PhaseBuilder::add` / `add_ref`).
    ///
    /// Categorical setters (`set_evaluator`, `set_integrator`, …) call this
    /// (or protected `bind_defaults`) internally. There is no public
    /// `Driver::add` that guesses the phase from `default_bindings()`.
    /// @param phase Phase value returned by the driver's `Phase` enum.
    /// @return A `PhaseBuilder` bound to that phase.
    PhaseBuilder<PhaseEnum, DeviceTag> on(PhaseEnum phase) {
        return PhaseBuilder<PhaseEnum, DeviceTag>(&pending_, &owned_components_,
                                                  phase, &mutation_lock_);
    }

    /// @brief Read-only access to the compiled Plan (after `build()`).
    const Plan<Component>& plan() const { return plan_; }

    /// @brief Compile the execution Plan from the live mount table.
    ///
    /// `pending_` stays populated so a second `build()` / `compile()` can
    /// rewrite `plan_` without reconstructing the driver.
    void build() {
        require_mutable_();
        PlanBuilder<Component> builder;
        for (auto& pb : pending_)
            builder.add({pb.phase, pb.order, pb.component});
        plan_ = builder.build(ATV::slot_count<PhaseEnum>());
        plan_.kind = kind_of(PhaseEnum{});
    }

    /// @brief Live mount table (survives `build()`).
    const std::vector<detail::PendingBinding<Component>>& mounted() const {
        return pending_;
    }

    /// @brief True if `id` is currently mounted (instance pointer, not name).
    bool is_mounted(PluginIdentity id) const {
        for (const auto& pb : pending_) {
            if (pb.identity() == id)
                return true;
        }
        return false;
    }

    /// @brief Enable per-component timing (call before run).
    void enable_timing();

    /// @brief Timing summary (call after run).
    /// @return Human-readable per-phase timing report, or an empty string on CPU.
    std::string summary() const;

    /// @brief Shut down Backend and release all resources.
    virtual void shutdown();

  protected:
    /// @brief Backend execution primitive; public workflow entry points belong
    /// to the concrete backend driver so lifecycle and termination cannot be bypassed.
    /// @param frame Host-side frame used as input and final output.
    /// @param n_steps Number of steps to dispatch.
    void run(frame_type& frame, int n_steps);

    // Internals are protected (not private) so the backend loop node
    // (cpu::Driver / cuda::Driver) can drive the per-iteration loop itself.

    /// @brief Bind a component using its `default_bindings()` (owned rvalue).
    ///
    /// **Not public.** Categorical helpers (`set_observer`,
    /// `set_singular_default_owned_`, criteria with stage bindings) call this.
    /// Users must use `on(phase).add` or `set_evaluator` / other categorical
    /// setters.
    template <typename Comp,
              std::enable_if_t<
                  std::is_base_of_v<Component, std::remove_reference_t<Comp>>
                      && !std::is_lvalue_reference_v<Comp>
                      && !std::is_base_of_v<typename traits::driver_node_type,
                                            std::remove_reference_t<Comp>>,
                  int> = 0>
    void bind_defaults(Comp&& comp) {
        require_mutable_();
        using C = std::remove_reference_t<Comp>;
        bind_defaults(std::make_unique<C>(std::forward<Comp>(comp)));
    }

    /// @brief Transfer ownership and bind via `default_bindings()`.
    template <typename Comp,
              std::enable_if_t<std::is_base_of_v<Component, Comp>, int> = 0>
    void bind_defaults(std::unique_ptr<Comp> comp) {
        require_mutable_();
        if (!comp)
            throw std::invalid_argument(
                "Driver::bind_defaults: null component");
        auto bindings = comp->default_bindings();
        if (bindings.empty())
            throw std::runtime_error(
                "Component '" + comp->name()
                + "' has no default bindings. Use .on(phase).add().");
        validate_binding_kinds_(comp->name(), bindings);
        Component* raw = comp.get();
        owned_components_.push_back(std::move(comp));
        for (const auto& db : bindings)
            pending_.push_back({db.phase, db.order, raw});
    }

    /// @brief Borrow a component via its `default_bindings()`.
    template <typename Comp,
              std::enable_if_t<std::is_base_of_v<Component, Comp>
                                   && !std::is_base_of_v<
                                       typename traits::driver_node_type, Comp>,
                               int> = 0>
    void bind_defaults_ref(Comp& comp) {
        require_mutable_();
        auto bindings = comp.default_bindings();
        if (bindings.empty())
            throw std::runtime_error(
                "Component '" + comp.name()
                + "' has no default bindings. Use .on(phase).add_ref().");
        validate_binding_kinds_(comp.name(), bindings);
        for (const auto& db : bindings)
            pending_.push_back({db.phase, db.order, &comp});
    }

    const bool* mutation_lock_ = nullptr;
    void require_mutable_() const {
        if (mutation_lock_ && *mutation_lock_)
            throw std::logic_error(
                "cannot change driver bindings during dispatch");
    }

    std::vector<detail::PendingBinding<Component>> pending_;
    std::vector<std::unique_ptr<Component>> owned_components_;
    Plan<Component> plan_;
    std::unique_ptr<BackendT> backend_;
    std::unique_ptr<Timer> timer_;

  private:
    /// @brief Registration-time gate: reject default bindings built from
    ///        another workflow's Phase enum.
    void
    validate_binding_kinds_(const std::string& name,
                            const std::vector<DefaultBinding>& bindings) const {
        const PhaseKind expected = kind_of(PhaseEnum{});
        for (const auto& db : bindings) {
            if (db.kind != expected)
                throw std::runtime_error(
                    "Component '" + name
                    + "' declares default bindings for the "
                    + std::string(phase_kind_name(db.kind))
                    + " workflow, but this driver executes "
                    + std::string(phase_kind_name(expected))
                    + " phases. Bind explicitly with .on(phase).add().");
        }
    }
};

} // namespace ATV
