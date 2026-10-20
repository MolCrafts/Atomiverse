/// @file cpu/component/component.h
/// @brief cpu::Component — OOP base for CPU simulation components.
///
/// Lifecycle: build() → optional initial force evaluation → launch() × N → destroy().
/// Role is a RoleFlag set (see facade/component_role.h).
///
/// Intermediate kinds (Potential, HF, Integrator, …) live in their own
/// headers under this directory — one class per file. Concrete components
/// inherit at most one kind: Concrete → Kind → Component.
#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/types.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

class DriverNode;
class Runtime;
class ConstraintProvider;
class NeighborConsumer;
class NeighborProvider;
class LongRangeParticipant;
class TimestepProvider;
class TimestepConsumer;

struct BuildContext {
    cpu::Context& runtime;
    cpu::Frame& frame;
};

struct LaunchContext {
    cpu::Context& runtime;
    cpu::Frame& frame;
    int stage;
    int step;
    /// @brief When false, force-writing components should skip force writes
    ///        (MC energy-only trials).
    bool forces_requested = true;
    /// @brief When false, skip stress/virial writes independently of forces.
    bool stress_requested = true;

    template <typename PhaseEnum>
    bool at(PhaseEnum phase) const {
        return stage == ATV::slot_of(phase);
    }
};

struct DestroyContext {
    cpu::Context& runtime;
    cpu::Frame& frame;
};

/// @brief Abstract base for CPU simulation components.
///
/// Deliberately NOT a `ForwardPacker`. The extra-field forward path is
/// self-initiated — a plugin calls `ctx.md.comm->forward_comm(*this)` when
/// it has dirtied owned rows (`fix_settle.cpp`, `fix_rattle.cpp`,
/// `ChargePacker::forward`) — so nothing in the engine scans for a packer
/// and no probe accessor is needed. Inheriting it here put three empty
/// virtuals in every component's vtable for the benefit of three classes.
/// @brief Liveness flag of one Component object.
///
/// A borrowed component (`add_ref`, `set_*_ref`) may be destroyed before the
/// driver that borrows it; the driver's teardown must then not call into it.
/// The Runtime records this flag when it builds the component and checks it
/// before `destroy()`. Copies and moves get a fresh flag: they are different
/// objects with their own lifetime.
class ComponentLifetime {
  public:
    ComponentLifetime() = default;
    ComponentLifetime(const ComponentLifetime&) {}
    ComponentLifetime(ComponentLifetime&&) noexcept {}
    ComponentLifetime& operator=(const ComponentLifetime&) { return *this; }
    ComponentLifetime& operator=(ComponentLifetime&&) noexcept { return *this; }
    ~ComponentLifetime() { *alive_ = false; }

    /// @brief Shared view of the flag; false once the object is destroyed.
    /// @return Flag that outlives the object.
    std::shared_ptr<const bool> token() const noexcept { return alive_; }

  private:
    std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);
};

class Component {
    friend class Runtime; // reads lifetime_ when it takes the object in

  public:
    virtual ~Component() = default;

    virtual std::string name() const = 0;

    /// @brief Behavioural RoleFlag set (may combine several flags).
    /// Default 0: main-loop required, may mutate the frame.
    virtual Role role() const { return 0; }

    /// Nested loop boundary for lifecycle refresh and active-plan contracts.
    virtual DriverNode* nested_node() const { return nullptr; }

    virtual const NeighborConsumer* as_neighbor_consumer() const {
        return nullptr;
    }
    virtual const NeighborProvider* as_neighbor_provider() const {
        return nullptr;
    }
    /// @brief Mutable supply side, for a driver that must CONFIGURE the
    /// neighbor structure it was handed (Monte Carlo forbids row reordering).
    /// Mirrors `as_timestep_consumer()`.
    virtual NeighborProvider* neighbor_provider() { return nullptr; }
    virtual const LongRangeParticipant* as_long_range_participant() const {
        return nullptr;
    }
    virtual const ConstraintProvider* as_constraint() const { return nullptr; }
    /// @brief This component owns the MD clock (integrators).
    /// @see component/timestep_protocol.h
    virtual const TimestepProvider* as_timestep_provider() const {
        return nullptr;
    }
    /// @brief This component must be handed the nuclear timestep [fs].
    /// @see component/timestep_protocol.h
    virtual TimestepConsumer* as_timestep_consumer() { return nullptr; }
    /// @brief True for CommBrick / Neighbor engine ops the MD Verlet
    ///        coordinator calls by name. The Plan walk skips them.
    virtual bool verlet_engine() const { return false; }

    /// @brief Plan needs a system-stress Observer (NPT / barostat).
    virtual bool requires_system_stress() const { return false; }
    /// @brief Writes system stress meta (kinetic and/or assembled virial).
    virtual bool provides_system_stress() const { return false; }

    /// @brief True when this plugin contributes a system energy term.
    /// @return Dimensionless predicate. Default false; energy kinds override.
    virtual bool provides_energy() const { return false; }
    /// @brief True when this plugin contributes a force / gradient term.
    /// @return Dimensionless predicate. Default false; force kinds override.
    virtual bool provides_force() const { return false; }
    /// @brief True when this plugin assembles Hessian rows.
    /// @return Dimensionless predicate. Default false.
    virtual bool provides_hessian() const { return false; }
    /// @brief Rewrites owned per-atom charges mid-step (constant-potential
    /// electrode solvers). After writing `kAtom_charge` the solver calls
    /// `ChargePacker::forward(ctx)` so ghost rows see the new charges
    /// [e] before the force stages read them.
    virtual bool updates_owned_charges() const { return false; }
    /// Requires all atoms on each executing rank (serial or full replica).
    /// Spatial decomposition must reject this component before partitioning.
    virtual bool requires_full_frame() const { return false; }

    /// @brief Resolves global tags through the Frame tag index at run time.
    ///
    /// `DomainRuntime` frees the Frame tag index (and the DD mirror, ~50 B per
    /// atom each) when nothing needs it. "Nothing needs it" used to mean
    /// `!frame.has_bonded()`, which is wrong: a component can name atoms by
    /// **tag** and carry no topology at all. `FixSettle` does exactly that — it
    /// is handed water triples as tags, and rows are not identity (the spatial
    /// sort renumbers them at its own cadence). Declaring the need here keeps
    /// the index alive for the whole run, and under domain decomposition gets
    /// it rebuilt for the owned rows after the initial partition instead of
    /// dropped.
    ///
    /// A payload-free declaration, so it is a predicate here rather than an
    /// `as_*` protocol object — same shape as `requires_system_stress()`.
    virtual bool requires_tag_index() const { return false; }

    virtual Scheduling scheduling() const { return Scheduling::blocking; }
    virtual int frequency() const { return 0; }
    virtual std::size_t memory_bytes() const { return 0; }
    virtual std::vector<DefaultBinding> default_bindings() const { return {}; }

    /// @brief What this component needs of the ghost copies it uses.
    ///
    /// A **static property of the type**, in the same family as
    /// `default_bindings()`: hand back a `static constexpr` table and the
    /// Driver folds every mounted component's answer once at `build()`.
    /// Nothing here runs during a step and nothing allocates.
    ///
    /// @code{.cpp}
    /// static constexpr std::array kGhosts = {
    ///     GhostField::read_stable(kAtom_charge, "CoulLong"),
    ///     GhostField::read_each_step(kAtom_vx, "PairDpd"),
    ///     GhostField::reduce_to_owner(kAtom_fx, "PairX"),
    /// };
    /// std::span<const GhostField> declare_ghosts() const override {
    ///     return kGhosts;
    /// }
    /// @endcode
    ///
    /// This is a **Component-author** surface, like `launch()` — no user
    /// script ever writes a `GhostField`. Owned rows need no declaration;
    /// this is only about what has to travel and what guarantee it carries.
    ///
    /// `read_each_step` is a **freshness level**, not a lifecycle event: it
    /// says the value must be current when this component reads it, not that
    /// anything happens per step. To *act* at a point in the loop, bind to a
    /// `Phase` (`on(phase).add(...)`) — that is the engine's event system.
    ///
    /// @return This component's declarations; empty by default, because most
    ///         components only touch rows they already own.
    virtual std::span<const GhostField> declare_ghosts() const { return {}; }

    virtual void build(const BuildContext& ctx) { (void)ctx; }
    virtual void launch(const LaunchContext& ctx) = 0;
    /// Initialization hook for pre/post-force and scalar-finalization bindings.
    /// Default excludes observers and time propagation; force-slot evaluators
    /// use launch() with runtime.exec.initializing_forces set.
    /// Opt in only to establish forces at the current configuration.
    virtual void initialize_forces(const LaunchContext& ctx) { (void)ctx; }
    virtual void destroy(const DestroyContext& ctx) { (void)ctx; }

  private:
    ComponentLifetime lifetime_;
};

} // namespace cpu
} // namespace ATV
