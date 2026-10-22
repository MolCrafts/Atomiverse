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
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/timestep_protocol.h"
#include "atomiverse/cpu/driver/driver_node.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/device/timer.h"
#include "atomiverse/cuda/driver/driver_node.h"

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

template <typename Node>
bool driver_provides_energy(const Node& node) {
    if constexpr (requires { node.provides_energy(); })
        return node.provides_energy();
    else
        return false;
}

template <typename Node>
bool driver_provides_force(const Node& node) {
    if constexpr (requires { node.provides_force(); })
        return node.provides_force();
    else
        return false;
}

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

    /// @brief Own a nested CPU driver on this phase.
    ///
    /// `@param order` Binding order within the phase (lower runs earlier).
    /// `@param budget` Dimensionless inner-loop iterations; `< 0` uses
    /// `child.default_budget()`. A type that is both Component and
    /// DriverNode keeps the Component overload.
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
        N* raw = node.get();
        return mount_nested_cpu(std::move(node), raw, order, budget);
    }

    /// @brief Borrow a nested CPU driver on this phase.
    ///
    /// The slot is owned by this driver; `child` stays with the caller
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
        return mount_nested_cpu(std::unique_ptr<Nested>{}, &child, order,
                                budget);
    }

    /// @brief Own a nested CUDA driver on this phase.
    ///
    /// `@param order` Binding order within the phase (lower runs earlier).
    /// `@param budget` Dimensionless inner-loop iterations; `< 0` uses
    /// `child.default_budget()`. A type that is both Component and
    /// DriverNode keeps the Component overload.
    template <typename Nested,
              std::enable_if_t<
                  std::is_same_v<DeviceTag, cuda::device_tag>
                      && std::is_base_of_v<cuda::DriverNode,
                                           std::remove_reference_t<Nested>>
                      && !std::is_base_of_v<Component,
                                            std::remove_reference_t<Nested>>
                      && !std::is_lvalue_reference_v<Nested>,
                  int> = 0>
    PhaseBuilder& add(Nested&& child, int order = 0, int budget = -1) {
        using N = std::remove_reference_t<Nested>;
        auto node = std::make_unique<N>(std::forward<Nested>(child));
        N* raw = node.get();
        return mount_nested_cuda(std::move(node), raw, order, budget);
    }

    /// @brief Borrow a nested CUDA driver on this phase.
    ///
    /// The slot is owned by this driver; `child` stays with the caller.
    /// `@param order` Binding order within the phase (lower runs earlier).
    /// `@param budget` Dimensionless inner-loop iterations; `< 0` uses
    /// `child.default_budget()`.
    template <
        typename Nested,
        std::enable_if_t<std::is_same_v<DeviceTag, cuda::device_tag>
                             && std::is_base_of_v<cuda::DriverNode, Nested>
                             && !std::is_base_of_v<Component, Nested>,
                         int> = 0>
    PhaseBuilder& add_ref(Nested& child, int order = 0, int budget = -1) {
        return mount_nested_cuda(std::unique_ptr<Nested>{}, &child, order,
                                 budget);
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

    template <typename Node>
    PhaseBuilder& mount_nested_cpu(std::unique_ptr<Node> owned, Node* live,
                                   int order, int budget) {
        Node* node = owned ? owned.get() : live;
        const int cap = budget < 0 ? node->default_budget() : budget;
        const bool energy = detail::driver_provides_energy(*node);
        const bool force = detail::driver_provides_force(*node);
        cpu::TimestepConsumer* step = nullptr;
        if constexpr (std::is_base_of_v<cpu::TimestepConsumer, Node>)
            step = static_cast<cpu::TimestepConsumer*>(node);

        struct Slot : Component {
            std::unique_ptr<cpu::DriverNode> owned_node;
            cpu::DriverNode* node = nullptr;
            cpu::TimestepConsumer* step = nullptr;
            int budget = 0;
            bool energy = false;
            bool force = false;

            std::string name() const override { return "nested_driver"; }
            bool provides_energy() const override { return energy; }
            bool provides_force() const override { return force; }
            bool requires_tag_index() const override {
                return node && node->requires_tag_index();
            }
            cpu::TimestepConsumer* as_timestep_consumer() override {
                return step;
            }
            void build(const cpu::BuildContext& ctx) override {
                node->build_on(require_runtime(ctx.runtime, "build"));
            }
            void launch(const cpu::LaunchContext& ctx) override {
                const auto saved_window = ctx.frame.resolution_window();
                const bool saved_forces = ctx.runtime.exec.forces_requested;
                ctx.frame.restore_resolution_window(cpu::ResolutionWindow{});
                ctx.runtime.exec.forces_requested =
                    saved_forces && ctx.forces_requested;
                node->run_nested_erased(ctx.runtime, ctx.frame, budget);
                ctx.runtime.exec.forces_requested = saved_forces;
                ctx.frame.restore_resolution_window(saved_window);
            }
            void destroy(const cpu::DestroyContext& ctx) override {
                node->destroy_on(require_runtime(ctx.runtime, "destroy"));
            }

            static cpu::Runtime& require_runtime(cpu::Context& ctx,
                                                 const char* where) {
                if (!ctx.exec.runtime) {
                    throw std::runtime_error(
                        std::string("nested driver ") + where
                        + ": no Runtime on the Context. Bind the driver under "
                          "a parent Driver that has completed setup_pass.");
                }
                return *ctx.exec.runtime;
            }
        };

        auto slot = std::make_unique<Slot>();
        slot->owned_node = std::move(owned);
        slot->node = node;
        slot->step = step;
        slot->budget = cap;
        slot->energy = energy;
        slot->force = force;
        Component* raw = slot.get();
        owned_->push_back(std::move(slot));
        bindings_->push_back({ATV::slot_of(phase_), order, raw});
        return *this;
    }

    template <typename Node>
    PhaseBuilder& mount_nested_cuda(std::unique_ptr<Node> owned, Node* live,
                                    int order, int budget) {
        Node* node = owned ? owned.get() : live;
        const int cap = budget < 0 ? node->default_budget() : budget;
        const bool energy = detail::driver_provides_energy(*node);
        const bool force = detail::driver_provides_force(*node);

        struct Slot : Component {
            std::unique_ptr<cuda::DriverNode> owned_node;
            cuda::DriverNode* node = nullptr;
            int budget = 0;
            bool energy = false;
            bool force = false;

            std::string name() const override { return "nested_driver"; }
            bool provides_energy() const override { return energy; }
            bool provides_force() const override { return force; }
            void build(const cuda::BuildContext& ctx) override {
                node->build_on(ctx.runtime, ctx.frame);
            }
            void launch(const cuda::LaunchContext& ctx) override {
                node->run_nested_erased(ctx.runtime, ctx.frame, budget);
            }
            void destroy(const cuda::DestroyContext& ctx) override {
                node->destroy_on(ctx.runtime, ctx.frame);
            }
        };

        auto slot = std::make_unique<Slot>();
        slot->owned_node = std::move(owned);
        slot->node = node;
        slot->budget = cap;
        slot->energy = energy;
        slot->force = force;
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

    /// Unmount an instance and all of its bindings, with backend resource
    /// release before ownership is erased. Repeated bindings are not namesakes.
    template <typename Release>
    void unmount_(std::string_view name, bool launching, Release&& release) {
        if (launching)
            throw std::runtime_error(
                "unfix: cannot unfix from inside launch(). Unfix between "
                "steps or run() chunks.");
        Component* found = nullptr;
        std::string mounted;
        for (const auto& binding : pending_) {
            auto* component = binding.component;
            if (!component)
                continue;
            if (!mounted.empty())
                mounted += ", ";
            mounted += component->name();
            if (component->name() != name)
                continue;
            if (found && found != component)
                throw std::runtime_error(
                    "unfix: name '" + std::string(name)
                    + "' is not unique. Give plugins distinct name() values.");
            found = component;
        }
        if (!found)
            throw std::runtime_error(
                "unfix: no plugin named '" + std::string(name) + "'. mounted: ["
                + mounted + "]. Remount with on(phase).add / set_evaluator.");
        release(found);
        std::erase_if(pending_, [found](const auto& binding) {
            return binding.component == found;
        });
        std::erase_if(owned_components_, [found](const auto& component) {
            return component.get() == found;
        });
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
