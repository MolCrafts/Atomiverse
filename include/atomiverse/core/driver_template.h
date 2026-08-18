/// @file driver.h
/// @brief Driver -- user-facing simulation facade with fluent API.
///
/// Backend-agnostic template: DeviceTag selects CPU or CUDA at compile time.
/// All method bodies are defined inline (template class). Explicit
/// instantiations live in src/cpu/driver/driver.cpp and src/cuda/driver/driver.cu.
///
/// Plan slots hold a Component*. A nested CPU loop is wrapped as
/// `cpu::NestedLoopAdapter` at `on(phase).add` / `add_ref`.
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
#include "atomiverse/cpu/driver/driver_node.h"
#include "atomiverse/cpu/driver/nested_loop_adapter.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/device/timer.h"

namespace ATV {

/// @brief GPU profiler type held by `Driver` (CUDA-free PIMPL facade).
using Timer = cuda::Timer;

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
        bindings_->push_back({ATV::slot_of(phase_), order, &comp});
        return *this;
    }

    /// @brief Own a nested driver on this phase (CPU only).
    ///
    /// Wraps `child` in `cpu::NestedLoopAdapter` (owned by this driver).
    /// `@param order` Binding order within the phase (lower runs earlier).
    /// `@param budget` Dimensionless inner-loop iterations; `< 0` uses
    /// `child.default_budget()`. Dual-identity Component+DriverNode types
    /// keep the Component overload.
    template <typename Nested,
              std::enable_if_t<
                  std::is_same_v<DeviceTag, cpu::device_tag>
                      && std::is_base_of_v<cpu::DriverNode,
                                           std::remove_reference_t<Nested>>
                      && !std::is_base_of_v<Component,
                                            std::remove_reference_t<Nested>>
                      && !std::is_lvalue_reference_v<Nested>,
                  int> = 0>
    PhaseBuilder& add(Nested&& child, int order = 0, int budget = -1) {
        using N = std::remove_reference_t<Nested>;
        auto node = std::make_unique<N>(std::forward<Nested>(child));
        auto adapter =
            std::make_unique<cpu::NestedLoopAdapter>(std::move(node), budget);
        Component* raw = adapter.get();
        owned_->push_back(std::move(adapter));
        bindings_->push_back({ATV::slot_of(phase_), order, raw});
        return *this;
    }

    /// @brief Borrow a nested driver on this phase (CPU only).
    ///
    /// The adapter is owned by this driver; `child` stays with the caller
    /// so `child.converged()` remains readable after `run`.
    /// `@param order` Binding order within the phase (lower runs earlier).
    /// `@param budget` Dimensionless inner-loop iterations; `< 0` uses
    /// `child.default_budget()`.
    template <typename Nested,
              std::enable_if_t<std::is_same_v<DeviceTag, cpu::device_tag>
                                   && std::is_base_of_v<cpu::DriverNode, Nested>
                                   && !std::is_base_of_v<Component, Nested>,
                               int> = 0>
    PhaseBuilder& add_ref(Nested& child, int order = 0, int budget = -1) {
        auto adapter = std::make_unique<cpu::NestedLoopAdapter>(child, budget);
        Component* raw = adapter.get();
        owned_->push_back(std::move(adapter));
        bindings_->push_back({ATV::slot_of(phase_), order, raw});
        return *this;
    }

    /// Implicit lvalue borrowing is forbidden. Use `add_ref(comp)`.
    template <typename Comp>
    PhaseBuilder& add(Comp&, int = 0) = delete;

  private:
    template <typename PE, typename DT>
    friend class Driver;
    PhaseBuilder(std::vector<detail::PendingBinding<Component>>* b,
                 std::vector<std::unique_ptr<Component>>* o,
                 std::vector<std::unique_ptr<cpu::DriverNode>>* d, PhaseEnum p)
        : bindings_(b), owned_(o), owned_drivers_(d), phase_(p) {}
    std::vector<detail::PendingBinding<Component>>* bindings_;
    std::vector<std::unique_ptr<Component>>* owned_;
    std::vector<std::unique_ptr<cpu::DriverNode>>* owned_drivers_;
    PhaseEnum phase_;
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

  public:
    Driver() = default;
    ~Driver() { shutdown(); }
    Driver(const Driver&) = delete;
    Driver& operator=(const Driver&) = delete;

    Driver(Driver&& o) noexcept
        : pending_(std::move(o.pending_)),
          owned_components_(std::move(o.owned_components_)),
          owned_drivers_(std::move(o.owned_drivers_)),
          plan_(std::move(o.plan_)), backend_(std::move(o.backend_)),
          timer_(std::move(o.timer_)) {}

    Driver& operator=(Driver&& o) noexcept {
        if (this != &o) {
            shutdown();
            pending_ = std::move(o.pending_);
            owned_components_ = std::move(o.owned_components_);
            owned_drivers_ = std::move(o.owned_drivers_);
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
                                                  &owned_drivers_, phase);
    }

    /// @brief Read-only access to the compiled Plan (after `build()`).
    const Plan<Component>& plan() const { return plan_; }

    /// @brief Compile the execution Plan from the live mount table.
    ///
    /// `pending_` stays populated so a second `build()` / `compile()` can
    /// rewrite `plan_` without reconstructing the driver.
    void build() {
        PlanBuilder<Component> builder;
        for (auto& pb : pending_)
            builder.add({pb.phase, pb.order, pb.component});
        plan_ = builder.build(ATV::slot_count<PhaseEnum>());
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

    /// @brief Run the full simulation lifecycle.
    /// @param frame Host-side frame used as input and final output.
    /// @param n_steps Number of time steps or SCF iterations to execute.
    void run(cpu::Frame& frame, int n_steps);

    /// @brief Enable per-component timing (call before run).
    void enable_timing();

    /// @brief Timing summary (call after run).
    /// @return Human-readable per-phase timing report, or an empty string on CPU.
    std::string summary() const;

    /// @brief Shut down Backend and release all resources.
    void shutdown();

  protected:
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
                      && !std::is_base_of_v<cpu::DriverNode,
                                            std::remove_reference_t<Comp>>,
                  int> = 0>
    void bind_defaults(Comp&& comp) {
        using C = std::remove_reference_t<Comp>;
        bind_defaults(std::make_unique<C>(std::forward<Comp>(comp)));
    }

    /// @brief Transfer ownership and bind via `default_bindings()`.
    template <typename Comp,
              std::enable_if_t<std::is_base_of_v<Component, Comp>, int> = 0>
    void bind_defaults(std::unique_ptr<Comp> comp) {
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
                                   && !std::is_base_of_v<cpu::DriverNode, Comp>,
                               int> = 0>
    void bind_defaults_ref(Comp& comp) {
        auto bindings = comp.default_bindings();
        if (bindings.empty())
            throw std::runtime_error(
                "Component '" + comp.name()
                + "' has no default bindings. Use .on(phase).add_ref().");
        validate_binding_kinds_(comp.name(), bindings);
        for (const auto& db : bindings)
            pending_.push_back({db.phase, db.order, &comp});
    }

    std::vector<detail::PendingBinding<Component>> pending_;
    std::vector<std::unique_ptr<Component>> owned_components_;
    std::vector<std::unique_ptr<cpu::DriverNode>> owned_drivers_;
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
