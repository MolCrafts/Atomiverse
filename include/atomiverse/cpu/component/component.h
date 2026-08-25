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

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/schema.h"
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

    /// @brief Fields this component needs the system to carry on GHOST rows.
    ///
    /// Same shape as `default_bindings()`: the component states a fact about
    /// itself and the Driver folds every component's answer into the system
    /// `Schema` at `build()`. Owned rows need no declaration — they are
    /// always there; this is about what has to travel.
    ///
    /// A pair style that indexes a neighbour (`q[j]`, `v[j]`) declares those
    /// fields here. Default: nothing, because most components only touch
    /// rows they already own.
    /// @param system Schema being accumulated. Add to it; never clear it.
    virtual void declare_fields(Schema& system) const { (void)system; }

    /// @brief Fields this component ACCUMULATES onto ghost rows, for the
    ///        reverse pass to fold back onto the owners.
    /// @param system Schema being accumulated.
    virtual void declare_reverse_fields(Schema& system) const { (void)system; }

    /// @brief Fields that CHANGE every step and must be refreshed on ghost
    ///        rows between neighbour rebuilds.
    ///
    /// The distinction `declare_fields` does not make, and which LAMMPS
    /// keeps as two lists (`fields_comm` vs `fields_border`,
    /// `atom_vec.cpp:36-39`): a field read on a ghost but constant between
    /// rebuilds rides the border wire once; a field that moves every step
    /// has to ride the per-step forward wire as well. Positions are the base
    /// case; a component that mutates some other ghost-visible field per
    /// step declares it here.
    ///
    /// Declaring here does NOT imply `declare_fields` — say both if the
    /// field must also be present when the ghost is first created.
    /// @param system Schema being accumulated.
    virtual void declare_refresh_fields(Schema& system) const { (void)system; }

    virtual void build(const BuildContext& ctx) { (void)ctx; }
    virtual void launch(const LaunchContext& ctx) = 0;
    virtual void destroy(const DestroyContext& ctx) { (void)ctx; }
};

} // namespace cpu
} // namespace ATV
