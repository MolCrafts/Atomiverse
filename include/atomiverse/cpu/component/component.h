/// @file cpu/component/component.h
/// @brief cpu::Component — OOP base for CPU simulation components.
///
/// Lifecycle: build() → launch() × N → destroy().
/// Role is a RoleFlag set (see facade/component_role.h).
///
/// Intermediate kinds (Potential, HF, Integrator, …) live in their own
/// headers under this directory — one class per file. Concrete components
/// inherit at most one kind: Concrete → Kind → Component.
#pragma once

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
    /// @brief When true, force-writing components should skip force writes
    ///        (MC energy-only trials).
    bool forces_requested = true;

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
class Component {
  public:
    virtual ~Component() = default;

    virtual std::string name() const = 0;

    /// @brief Behavioural RoleFlag set (may combine several flags).
    /// Default 0: main-loop required, may mutate the frame.
    virtual Role role() const { return 0; }

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
    virtual void destroy(const DestroyContext& ctx) { (void)ctx; }
};

} // namespace cpu
} // namespace ATV
