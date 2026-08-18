/// @file cuda/component/component.cuh
/// @brief Component -- OOP base class for all simulation components.
///
/// Components follow a strict 3-phase lifecycle:
///   build()   -- one-time resource preparation (after Runtime + DeviceFrame exist)
///   launch()  -- per-step GPU work submission (pure GPU, zero D2H/sync/malloc)
///   destroy() -- resource cleanup
///
/// Two registration paths:
/// - `sim.set_observer(comp)` -- uses default_bindings().
/// - `sim.on(phase).add(comp)` -- explicit phase override.
///
/// CUDA-free header.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/types.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/frame/frame.cuh"

namespace ATV {
namespace cuda {

class HaloParticipant;  // opt-in halo protocol (halo_participant.cuh)
class NeighborConsumer; // opt-in neighbor protocols (neighbor_protocol.cuh)
class NeighborProvider;
class
    LongRangeParticipant; // opt-in long-range protocol (long_range_protocol.cuh)

/// @brief Context for one-time component build (after Runtime + DeviceFrame exist).
///
/// Passed to Component::build(). Components may perform cudaMalloc,
/// cudaMemcpy, cudaStreamSynchronize during build -- any host/device work.
///
/// @see Component::build()
struct BuildContext {
    cuda::Context& runtime; ///< CUDA execution context (streams).
    cuda::Frame& frame;     ///< Device-resident system state + host metadata.
};

/// @brief Context for per-step GPU work submission.
///
/// Passed to Component::launch() every simulation step. launch() must
/// remain GPU-pure: only kernel launches, cudaMemsetAsync, cudaMemcpyAsync D2D.
///
/// @see Component::launch()
struct LaunchContext {
    cuda::Context& runtime; ///< CUDA execution context (streams).
    cuda::Frame& frame;     ///< Device-resident system state + host metadata.
    int stage; ///< Current Plan slot index (erased — compare via `at()`).
    int step;  ///< Current simulation step index (0-based).

    /// @brief Is the component running in `phase` right now? @see cpu::LaunchContext::at
    /// @tparam PhaseEnum The workflow Phase enum the component is bound in.
    /// @param phase The phase to test against.
    /// @return True when the current stage is that phase.
    template <typename PhaseEnum>
    bool at(PhaseEnum phase) const {
        return stage == ATV::slot_of(phase);
    }
};

/// @brief Context for component resource cleanup.
///
/// Passed to Component::destroy() after the simulation loop completes.
/// Components must release all GPU resources allocated in build().
///
/// @see Component::destroy()
struct DestroyContext {
    cuda::Context& runtime; ///< CUDA execution context (streams).
    cuda::Frame& frame;     ///< Device-resident system state + host metadata.
};

/// @brief Abstract base for simulation components.
///
/// A Component is a CPU-side facade for staged algorithmic behavior.
/// It owns configuration, persistent state, and lifecycle, but not the
/// global loop. Components follow a strict **build / launch / destroy**
/// lifecycle managed by the Backend.
///
/// Two registration paths:
/// @code{.cpp}
/// sim.set_observer(comp);                  // uses default_bindings()
/// sim.on(phase).add(comp, order); // explicit phase override
/// @endcode
///
/// @see Driver for the user-facing registration API
/// @see Backend for lifecycle dispatch
class Component {
  public:
    virtual ~Component() = default;

    /// @brief Human-readable component name (used in timing reports).
    /// @return Component identifier string.
    virtual std::string name() const = 0;

    /// @brief RoleFlag set (mirrors cpu::Component). @see facade/component_role.h
    virtual Role role() const { return 0; }

    virtual bool requires_system_stress() const { return false; }
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

    /// @brief Per-run hint: does any component downstream consume this
    ///        evaluator's stress/virial contribution?
    ///
    /// `cuda::md::MD::install_defaults()` calls this on every pending
    /// component with the plan-level verdict (`needs_stress` — true iff
    /// some component `requires_system_stress()`). Virial-producing
    /// evaluators (pair / bonded) override it to gate their per-step virial
    /// accumulation: in a plain NVE run (no barostat/stress observer) the
    /// pair kernel then skips the 6 inner-loop virial products and the
    /// stress reduction, and frees the accumulator registers under its
    /// launch-bounds budget. Default no-op: components that produce no
    /// virial ignore it. No `dynamic_cast` / `name()` matching.
    /// @param required true iff a downstream consumer needs the virial.
    virtual void set_stress_required(bool /*required*/) {}

    /// @brief Typed accessor for the consumer side of the opt-in
    ///        neighbor protocol (mirrors
    ///        `cpu::Component::as_neighbor_consumer()`).
    ///
    /// Pair evaluators (`cuda::Pair` subclasses) return `this`;
    /// `cuda::md::MD::install_infra_` probes it for the missing-NL gate —
    /// no `dynamic_cast`, no `name()` matching. Default: not a consumer.
    /// @return The consumer view of this component, or `nullptr`.
    virtual const NeighborConsumer* as_neighbor_consumer() const {
        return nullptr;
    }

    /// @brief Typed accessor for the supply side of the opt-in neighbor
    ///        protocol. `cuda::NeighborList` returns `this`. Default:
    ///        not a provider.
    /// @return The provider view of this component, or `nullptr`.
    virtual const NeighborProvider* as_neighbor_provider() const {
        return nullptr;
    }
    /// @brief Typed accessor for split long-range electrostatics.
    ///        `cuda::CoulLong` / `cuda::KSpace` return `this`. Default: none.
    virtual const LongRangeParticipant* as_long_range_participant() const {
        return nullptr;
    }

    /// @brief Scheduling trait (blocking or deferrable).
    /// @return Scheduling::blocking (default) or Scheduling::deferrable.
    virtual Scheduling scheduling() const { return Scheduling::blocking; }

    /// @brief Execution frequency: 0 = every step, N = every Nth step.
    /// @return Step frequency. 0 means execute on every step.
    virtual int frequency() const { return 0; }

    /// @brief Default stage bindings. Empty = must bind explicitly via on().add().
    /// @return Vector of DefaultBinding (phase index + order pairs).
    virtual std::vector<DefaultBinding> default_bindings() const { return {}; }

    /// @brief One-time resource preparation.
    ///
    /// Called once after Runtime and DeviceFrame are materialized.
    /// May perform cudaMalloc, D2H reads, sync -- any host/device work.
    /// All persistent GPU resources for repeated launch() must be created here.
    ///
    /// @param ctx Build context with Runtime and DeviceFrame references.
    virtual void build(const BuildContext& ctx) {}

    /// @brief Per-step GPU work submission.
    ///
    /// Called every step during the simulation loop.
    /// Must remain GPU-pure: only kernel launches, cudaMemsetAsync,
    /// cudaMemcpyAsync D2D. Zero cudaMalloc, zero D2H, zero sync.
    ///
    /// @param ctx Launch context with Runtime, DeviceFrame, phase, and step.
    virtual void launch(const LaunchContext& ctx) = 0;

    /// @brief Resource cleanup.
    ///
    /// Called once after the simulation loop completes.
    /// Must release all resources allocated in build().
    /// Executed in reverse Plan order.
    ///
    /// @param ctx Destroy context with Runtime and DeviceFrame references.
    virtual void destroy(const DestroyContext& ctx) {}

    /// @brief Typed accessor for the opt-in halo protocol (mirrors
    ///        `cpu::Component::halo_participant()`). A participating
    ///        component inherits `cuda::HaloParticipant` and overrides this
    ///        to return `this`; probes stay free of `dynamic_cast` and
    ///        `name()` matching. Default: not a participant.
    /// @return The participant view of this component, or `nullptr`.
    virtual HaloParticipant* halo_participant() { return nullptr; }

    /// @brief Heap/device bytes this component owns (workspaces, rings,
    ///        comm buffers). Mirrors `cpu::Component::memory_bytes()`
    ///        so both backends expose the same core reporting set; the
    ///        driver-side memory summary consumption lands with
    ///        arch-refactor-04. Default 0 for stateless components.
    virtual std::size_t memory_bytes() const { return 0; }

    // The former in-vtable CommField trio moved to the opt-in
    // `cuda::HaloParticipant` protocol (halo_participant.cuh), shared
    // shape with the CPU backend. See arch-refactor-02-halo.
};

} // namespace cuda
} // namespace ATV
