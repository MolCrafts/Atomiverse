/// @file plan.h
/// @brief Plan + PlanBuilder -- execution plan compilation. CUDA-free.
///
/// The Plan is the compiled schedule that bridges Facade and Backend.
/// PlanBuilder resolves bindings into sorted StageSlots. Built once,
/// executed many times.
///
/// Nodes are either a Component* or a nested-driver binding
/// (`nested_driver` + `nested_budget`). Nested drivers are type-erased as
/// `void*` here so this header stays free of `cpu::Driver`; the CPU backend
/// casts to `cpu::Driver*`. CUDA never populates driver nodes.
///
/// @see Driver::build() for how Plans are created
/// @see Backend::step() for how Plans are executed
#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/types.h"

namespace ATV {

/// @brief One resolved operation within a StageSlot.
/// @tparam Component Component base type (cpu::Component or cuda::Component).
template <typename Component>
struct Operation {
    Component* component = nullptr; ///< Non-null for a component node.
    void* nested_driver = nullptr;  ///< Non-null for a nested-driver node.
    int nested_budget = 0;          ///< Budget for a nested-driver node.
    Scheduling scheduling =
        Scheduling::blocking; ///< Cached scheduling (component nodes).
    int frequency = 0;        ///< Cached frequency (0 = every step).

    /// @brief True when this op is a nested driver (not a Component).
    bool is_driver() const { return nested_driver != nullptr; }
};

/// @brief All operations for one phase, sorted by order.
/// @tparam Component Component base type.
template <typename Component>
struct StageSlot {
    int phase; ///< Phase index within the driver's Phase enum.
    std::vector<Operation<Component>>
        ops; ///< Operations sorted by registration order.
};

/// @brief Compiled execution plan -- the schedule between Facade and Backend.
///
/// Contains one StageSlot per phase. Built once by PlanBuilder, then
/// executed many times by Backend::step().
///
/// @tparam Component Component base type.
template <typename Component>
struct Plan {
    std::vector<StageSlot<Component>> slots; ///< One StageSlot per phase.
};

/// @brief Compiles bindings into a Plan.
///
/// Collects binding entries (phase + order + component|driver), then sorts
/// operations within each phase by order value (stable sort).
///
/// @tparam Component Component base type.
/// @see Driver::build() which uses PlanBuilder internally
template <typename Component>
class PlanBuilder {
  public:
    /// @brief Binding entry: associates a node with a phase and order.
    struct Entry {
        int phase;                    ///< Target phase index.
        int order;                    ///< Ordering within phase (lower = earlier).
        Component* component = nullptr; ///< Component node (xor nested_driver).
        void* nested_driver = nullptr;  ///< Nested-driver node (cpu::Driver*).
        int nested_budget = 0;          ///< Budget for nested-driver nodes.
    };

    /// @brief Register a binding entry.
    /// @param entry Binding to register.
    void add(Entry entry) { entries_.push_back(entry); }

    /// @brief Compile all registered bindings into a Plan.
    /// @param phase_count Number of phases (typically PhaseEnum::COUNT).
    /// @return Compiled Plan with sorted operations per phase.
    /// @throws std::runtime_error If a binding targets an invalid phase.
    Plan<Component> build(int phase_count) {
        Plan<Component> plan;
        plan.slots.resize(phase_count);
        for (int p = 0; p < phase_count; ++p)
            plan.slots[p].phase = p;

        struct SE {
            int order;
            Operation<Component> op;
        };
        std::vector<std::vector<SE>> slot_entries(phase_count);

        for (auto& e : entries_) {
            if (e.phase < 0 || e.phase >= phase_count) {
                const std::string who = e.component
                                            ? e.component->name()
                                            : std::string("nested_driver");
                throw std::runtime_error("Node '" + who
                                         + "' binds to invalid phase "
                                         + std::to_string(e.phase));
            }
            Operation<Component> op;
            if (e.nested_driver) {
                op.nested_driver = e.nested_driver;
                op.nested_budget = e.nested_budget;
                op.scheduling = Scheduling::blocking;
                op.frequency = 0;
            } else {
                op.component = e.component;
                op.scheduling = e.component->scheduling();
                op.frequency = e.component->frequency();
            }
            slot_entries[e.phase].push_back({e.order, op});
        }

        for (int p = 0; p < phase_count; ++p) {
            std::stable_sort(
                slot_entries[p].begin(), slot_entries[p].end(),
                [](const SE& a, const SE& b) { return a.order < b.order; });
            for (auto& se : slot_entries[p])
                plan.slots[p].ops.push_back(se.op);
        }

        return plan;
    }

  private:
    std::vector<Entry> entries_; ///< Accumulated binding entries.
};

} // namespace ATV
