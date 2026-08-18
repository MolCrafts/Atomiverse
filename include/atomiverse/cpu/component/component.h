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
#include "atomiverse/core/stage.h"
#include "atomiverse/core/types.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

class HaloParticipant;
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
    /// Mirrors `halo_participant()` / `as_timestep_consumer()`.
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
    virtual HaloParticipant* halo_participant() { return nullptr; }

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
    /// electrode solvers). Under MPI DD the MD driver auto-injects a
    /// `comm::HaloRefresh{Field::charges}` after the charge-solve slot so
    /// ghost rows see the updated charges before the force stages read them
    /// (otherwise ghost q is one solve stale — zero at step 0).
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

    virtual void build(const BuildContext& ctx) { (void)ctx; }
    virtual void launch(const LaunchContext& ctx) = 0;
    virtual void destroy(const DestroyContext& ctx) { (void)ctx; }
};

} // namespace cpu
} // namespace ATV
