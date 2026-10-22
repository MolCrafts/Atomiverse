/// @file core/plan.h
/// @brief Plan + PlanBuilder -- execution plan compilation. CUDA-free.
///
/// The Plan is the compiled schedule that bridges Facade and Backend.
/// PlanBuilder resolves bindings into sorted StageSlots. Built once,
/// executed many times.
///
/// Every slot holds a Component*. A nested loop enters through the private
/// slot built by `on(phase).add`; this header never names a loop-node type.
///
/// @see Driver::build() for how Plans are created
/// @see Backend::step() for how Plans are executed
#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/types.h"

namespace ATV {

/// @brief One resolved operation within a StageSlot.
/// @tparam Component Component base type (cpu::Component or cuda::Component).
template <typename Component>
struct Operation {
    Component* component = nullptr; ///< Plugin occupying this slot.
    Scheduling scheduling = Scheduling::blocking; ///< Cached scheduling.
    int frequency = 0; ///< Cached frequency (0 = every step).
    bool operator==(const Operation&) const = default;
};

/// @brief All operations for one phase, sorted by order.
/// @tparam Component Component base type.
template <typename Component>
struct StageSlot {
    int phase; ///< Phase index within the driver's Phase enum.
    std::vector<Operation<Component>>
        ops; ///< Operations sorted by registration order.

    /// @brief Compiled per-phase force policy — the phase runs energy-only
    /// when false.
    ///
    /// Stamped once by `cpu::Driver::build()` from its
    /// `phase_wants_forces(PhaseEnum)` hook, so the decision costs a build-time
    /// pass and never a per-step branch. The Runtime ANDs it with the Context's
    /// nesting channel (`ExecServices::forces_requested`) into every
    /// `LaunchContext` this slot launches, and a force-writing component skips
    /// its gradient when the result is false (MC's trial `energy` phase — the
    /// Metropolis test scores an energy and discards the move).
    /// Default true: every other phase asks for forces.
    bool forces_requested = true;
    /// @brief Whether this phase requests stress/virial output.
    bool stress_requested = true;
    bool operator==(const StageSlot&) const = default;
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
    /// Objects with lifecycle but no scheduled launch (e.g. polled criteria).
    std::vector<Component*> lifecycle_only;
    /// @brief Workflow this Plan was compiled for. MD uses the Verlet
    ///        `nflag` comm block in `Runtime::step`; other kinds walk slots
    ///        linearly. Stamped by `Driver::build()` from `kind_of(Phase)`.
    PhaseKind kind = PhaseKind::Evaluate;

    /// Initial force preparation schedule for MD and minimization. Force-slot evaluators run normally;
    /// other operations opt in through initialize_forces. Integration and
    /// ordinary observer/recorder launches are never part of preparation.
    template <typename BeforePhase, typename Visit>
    void prepare_forces(BeforePhase&& before_phase, Visit&& visit) const {
        if (kind == PhaseKind::Minimizer) {
            for (const auto& slot : slots) {
                if (slot.phase != slot_of(minimizer::Phase::force))
                    continue;
                before_phase(slot.phase);
                for (const auto& op : slot.ops)
                    visit(slot, op, true);
            }
            return;
        }
        if (kind != PhaseKind::MD)
            return;
        for (const auto& slot : slots) {
            if (slot.phase < slot_of(md::Phase::post_integrate)
                || slot.phase == slot_of(md::Phase::final_integrate))
                continue;
            before_phase(slot.phase);
            for (const auto& op : slot.ops)
                visit(slot, op, slot.phase == slot_of(md::Phase::force));
        }
    }

    bool requires_full_frame = false;
    /// Some component of this plan reads the system stress (a barostat, a
    /// pressure criterion). Expensive stress sources (electronic structure)
    /// compute their virial only when this is set.
    bool stress_consumed = false;
    bool operator==(const Plan&) const = default;

    /// Unique objects in construction order; phase bindings remain untouched.
    std::vector<Component*> components() const {
        std::vector<Component*> objects;
        std::unordered_set<Component*> seen;
        for (const auto& slot : slots)
            for (const auto& op : slot.ops)
                if (op.component && seen.insert(op.component).second)
                    objects.push_back(op.component);
        for (auto* c : lifecycle_only)
            if (c && seen.insert(c).second)
                objects.push_back(c);
        return objects;
    }
};

namespace detail {

/// Runtime-owned lifecycle ledger shared by CPU and CUDA. Nested plans
/// acquire references to instances; bindings never acquire resources.
template <typename Component>
class ComponentLifecycle {
  public:
    bool empty() const noexcept { return entries_.empty(); }
    template <typename Build, typename Destroy>
    void build(const Plan<Component>& plan, Build&& build_one,
               Destroy&& destroy_one) {
        std::vector<Component*> acquired;
        acquired.reserve(plan.components().size());
        try {
            for (auto* component : plan.components()) {
                auto it = find_(component);
                if (it != entries_.end()) {
                    ++it->references;
                } else {
                    try {
                        build_one(*component);
                    } catch (...) {
                        // build may have acquired only part of its resources.
                        destroy_one(*component);
                        throw;
                    }
                    entries_.push_back({component, 1});
                }
                acquired.push_back(component);
            }
        } catch (...) {
            for (auto it = acquired.rbegin(); it != acquired.rend(); ++it)
                release_(*it, destroy_one);
            throw;
        }
    }

    template <typename Destroy>
    void destroy(const Plan<Component>& plan, Destroy&& destroy_one) {
        auto components = plan.components();
        for (auto it = components.rbegin(); it != components.rend(); ++it)
            release_(*it, destroy_one);
    }

    template <typename Destroy>
    void destroy_all(Destroy&& destroy_one) {
        // Pop before the callback: destroying a nested driver may itself
        // release child plans in this same ledger.
        while (!entries_.empty()) {
            auto* component = entries_.back().component;
            entries_.pop_back();
            destroy_one(*component);
        }
    }

  private:
    struct Entry {
        Component* component;
        std::size_t references;
    };
    auto find_(Component* component) {
        return std::find_if(
            entries_.begin(), entries_.end(),
            [component](const Entry& e) { return e.component == component; });
    }
    template <typename Destroy>
    void release_(Component* component, Destroy& destroy_one) {
        auto it = find_(component);
        if (it == entries_.end() || --it->references != 0)
            return;
        entries_.erase(it);
        destroy_one(*component);
    }
    std::vector<Entry> entries_;
};

} // namespace detail

/// @brief Compiles bindings into a Plan.
///
/// Collects binding entries (phase + order + component), then sorts
/// operations within each phase by order value (stable sort).
///
/// @tparam Component Component base type.
/// @see Driver::build() which uses PlanBuilder internally
template <typename Component>
class PlanBuilder {
  public:
    /// @brief Binding entry: associates a node with a phase and order.
    struct Entry {
        int phase; ///< Target phase index.
        int order; ///< Ordering within phase (lower = earlier).
        Component* component = nullptr; ///< Plugin occupying the slot.
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
                const std::string who =
                    e.component ? e.component->name() : std::string("plugin");
                throw std::runtime_error("Node '" + who
                                         + "' binds to invalid phase "
                                         + std::to_string(e.phase));
            }
            if (!e.component) {
                throw std::runtime_error("PlanBuilder: null component at phase "
                                         + std::to_string(e.phase));
            }
            Operation<Component> op;
            op.component = e.component;
            op.scheduling = e.component->scheduling();
            op.frequency = e.component->frequency();
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
