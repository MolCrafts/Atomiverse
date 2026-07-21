/// @file driver.h
/// @brief Driver -- user-facing simulation facade with fluent API.
///
/// Backend-agnostic template: DeviceTag selects CPU or CUDA at compile time.
/// All method bodies are defined inline (template class). Explicit
/// instantiations live in src/cpu/driver/driver.cpp and src/cuda/driver/driver.cu.
///
/// Plan slots hold a Component* **or** a nested-driver binding (CPU only).
/// Nested drivers are registered via `on(phase).add(driver, order, budget)` /
/// `add_ref` and erased as `cpu::Driver*` (stored as void* in pending/plan).
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
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/device/timer.h"
#include "atomiverse/facade/plan.h"
#include "atomiverse/facade/stage.h"

namespace ATV {

/// @brief GPU profiler type held by `Driver` (CUDA-free PIMPL facade).
using Timer = cuda::Timer;

namespace detail {
/// @brief Unresolved binding collected before `Driver::build()`.
///
/// Exactly one of `component` / `nested_driver` is non-null.
template <typename Component>
struct PendingBinding {
    int phase = 0;              ///< Target phase index.
    int order = 0;              ///< Ordering within the phase (lower runs earlier).
    Component* component = nullptr; ///< Component node.
    void* nested_driver = nullptr;  ///< Nested-driver node (`cpu::Driver*`).
    int nested_budget = 0;          ///< Budget for nested-driver nodes.

    /// @brief True when this binding is a nested driver.
    bool is_driver() const { return nested_driver != nullptr; }
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
        bindings_->push_back(
            {ATV::slot_of(phase_), order, raw, nullptr, 0});
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
        bindings_->push_back(
            {ATV::slot_of(phase_), order, raw, nullptr, 0});
        return *this;
    }

    /// @brief Borrow a component whose lifetime is owned by the caller.
    ///
    /// Prefer Component binding when the type is dual-identity
    /// (Component + cpu::Driver), e.g. transitional MeanFieldMethod.
    template <typename Comp,
              std::enable_if_t<std::is_base_of_v<Component, Comp>, int> = 0>
    PhaseBuilder& add_ref(Comp& comp, int order = 0) {
        bindings_->push_back(
            {ATV::slot_of(phase_), order, &comp, nullptr, 0});
        return *this;
    }

    /// @brief Own a nested driver on this phase (CPU only).
    ///
    /// Moves `child` into the parent's `owned_drivers_`. `budget < 0` uses
    /// `child.default_budget()`. Disabled for types that are also Components
    /// (e.g. transitional MeanFieldMethod = HF + CPUDriver) so they keep
    /// binding as force/energy components until 04b drops the dual identity.
    template <
        typename Nested,
        std::enable_if_t<
            std::is_same_v<DeviceTag, cpu::device_tag>
                && std::is_base_of_v<cpu::Driver, std::remove_reference_t<Nested>>
                && !std::is_base_of_v<Component, std::remove_reference_t<Nested>>
                && !std::is_lvalue_reference_v<Nested>,
            int> = 0>
    PhaseBuilder& add(Nested&& child, int order = 0, int budget = -1) {
        using N = std::remove_reference_t<Nested>;
        auto ptr = std::make_unique<N>(std::forward<Nested>(child));
        cpu::Driver* raw = ptr.get();
        const int b = (budget < 0) ? raw->default_budget() : budget;
        owned_drivers_->push_back(std::move(ptr));
        bindings_->push_back(
            {ATV::slot_of(phase_), order, nullptr, static_cast<void*>(raw), b});
        return *this;
    }

    /// @brief Borrow a nested driver on this phase (CPU only).
    ///
    /// Non-owning: the caller may inspect `child.converged()` after the parent
    /// run. `budget < 0` uses `child.default_budget()`. See rvalue `add` for
    /// the Component dual-identity exclusion.
    template <
        typename Nested,
        std::enable_if_t<std::is_same_v<DeviceTag, cpu::device_tag>
                             && std::is_base_of_v<cpu::Driver, Nested>
                             && !std::is_base_of_v<Component, Nested>,
                         int> = 0>
    PhaseBuilder& add_ref(Nested& child, int order = 0, int budget = -1) {
        cpu::Driver* raw = &child;
        const int b = (budget < 0) ? raw->default_budget() : budget;
        bindings_->push_back(
            {ATV::slot_of(phase_), order, nullptr, static_cast<void*>(raw), b});
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
                 std::vector<std::unique_ptr<cpu::Driver>>* d, PhaseEnum p)
        : bindings_(b), owned_(o), owned_drivers_(d), phase_(p) {}
    std::vector<detail::PendingBinding<Component>>* bindings_;
    std::vector<std::unique_ptr<Component>>* owned_;
    std::vector<std::unique_ptr<cpu::Driver>>* owned_drivers_;
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

    /// @brief Add a component using its `default_bindings()` (owned rvalue).
    template <typename Comp,
              std::enable_if_t<
                  std::is_base_of_v<Component, std::remove_reference_t<Comp>>
                      && !std::is_lvalue_reference_v<Comp>
                      && !std::is_base_of_v<cpu::Driver,
                                            std::remove_reference_t<Comp>>,
                  int> = 0>
    void add(Comp&& comp) {
        using C = std::remove_reference_t<Comp>;
        add(std::make_unique<C>(std::forward<Comp>(comp)));
    }

    /// @brief Transfer ownership from an explicit unique pointer.
    template <typename Comp,
              std::enable_if_t<std::is_base_of_v<Component, Comp>, int> = 0>
    void add(std::unique_ptr<Comp> comp) {
        if (!comp)
            throw std::invalid_argument("Driver::add: null component");
        auto bindings = comp->default_bindings();
        if (bindings.empty())
            throw std::runtime_error(
                "Component '" + comp->name()
                + "' has no default bindings. Use .on(phase).add().");
        validate_binding_kinds_(comp->name(), bindings);
        Component* raw = comp.get();
        owned_components_.push_back(std::move(comp));
        for (const auto& db : bindings)
            pending_.push_back({db.phase, db.order, raw, nullptr, 0});
    }

    /// @brief Borrow a component whose lifetime is owned by the caller.
    template <typename Comp,
              std::enable_if_t<std::is_base_of_v<Component, Comp>
                                   && !std::is_base_of_v<cpu::Driver, Comp>,
                               int> = 0>
    void add_ref(Comp& comp) {
        auto bindings = comp.default_bindings();
        if (bindings.empty())
            throw std::runtime_error(
                "Component '" + comp.name()
                + "' has no default bindings. Use .on(phase).add_ref().");
        validate_binding_kinds_(comp.name(), bindings);
        for (const auto& db : bindings)
            pending_.push_back({db.phase, db.order, &comp, nullptr, 0});
    }

    /// Drivers have no default bindings — use `.on(phase).add(...)`.
    template <typename Nested,
              std::enable_if_t<std::is_base_of_v<cpu::Driver,
                                                 std::remove_reference_t<Nested>>,
                               int> = 0>
    void add(Nested&&) = delete;

    /// Drivers have no default bindings — use `.on(phase).add_ref(...)`.
    template <typename Nested,
              std::enable_if_t<std::is_base_of_v<cpu::Driver, Nested>, int> = 0>
    void add_ref(Nested&) = delete;

    /// Implicit lvalue borrowing is forbidden. Use `add_ref(comp)`.
    template <typename Comp>
    void add(Comp&) = delete;

    /// @brief Begin explicit phase binding.
    /// @param phase Phase value returned by the driver's `Phase` enum.
    /// @return A `PhaseBuilder` bound to that phase.
    PhaseBuilder<PhaseEnum, DeviceTag> on(PhaseEnum phase) {
        return PhaseBuilder<PhaseEnum, DeviceTag>(
            &pending_, &owned_components_, &owned_drivers_, phase);
    }

    /// @brief Read-only access to the compiled Plan (after `build()`).
    const Plan<Component>& plan() const { return plan_; }

    /// @brief Compile the execution Plan.
    void build() {
        PlanBuilder<Component> builder;
        for (auto& pb : pending_) {
            if (pb.is_driver())
                builder.add({pb.phase, pb.order, nullptr, pb.nested_driver,
                             pb.nested_budget});
            else
                builder.add(
                    {pb.phase, pb.order, pb.component, nullptr, 0});
        }
        pending_.clear();
        plan_ = builder.build(ATV::slot_count<PhaseEnum>());
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
    // (cpu::CPUDriver / cuda::CUDADriver) can drive the per-iteration loop itself.
    std::vector<detail::PendingBinding<Component>> pending_;
    std::vector<std::unique_ptr<Component>> owned_components_;
    std::vector<std::unique_ptr<cpu::Driver>> owned_drivers_;
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
