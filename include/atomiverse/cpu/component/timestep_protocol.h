/// @file cpu/component/timestep_protocol.h
/// @brief TimestepProvider / TimestepConsumer — opt-in protocols that
///        single-source the nuclear MD timestep. CUDA-free header.
///
/// The integrator owns the clock: `VelocityVerlet{0.5}` already knows the
/// step, and every other component that must run on the SAME clock (a
/// Car–Parrinello electronic propagator, an extended-Lagrangian density
/// propagator) used to re-declare it as its own parameter. That duplication
/// is unverifiable by construction — "must match the nuclear dt" was a
/// docs-only contract, so a typo produced a silently wrong trajectory rather
/// than an error.
///
/// The two halves of the contract are disjoint populations (integrators
/// supply, methods demand), so they are two protocols, following the
/// `NeighborProvider` / `NeighborConsumer` precedent: a participant inherits
/// the protocol alongside `Component` and overrides the matching `as_*`
/// accessor to return `this`. `cpu::md::MD::install_defaults()` probes those
/// typed accessors — no `dynamic_cast` on components, no `name()` matching —
/// and pushes the one agreed dt to every consumer before any component
/// `build()` runs. Loops with no integrator (Evaluate, Minimize) push
/// nothing; a consumer that cannot run without a dt then fails in its own
/// `build()`.
#pragma once

namespace ATV {
namespace cpu {

/// @brief Supply side: a component that owns the MD integration step.
///
/// Implemented by the integrators (`VelocityVerlet`, `NVELimit`,
/// `NoseHooverNPT`, `NphIntegrator`, `ParrinelloRahmanNPT`, `Brownian`).
/// Two providers publishing different values make the clock ambiguous and
/// `MD::build()` throws naming both.
class TimestepProvider {
  public:
    virtual ~TimestepProvider() = default;

    /// @brief Nuclear MD timestep [fs]; must be finite and > 0.
    /// @return The integrator's own step size [fs] (its constructor `dt`).
    virtual double timestep() const = 0;
};

/// @brief Demand side: a component/method that must know the nuclear dt.
///
/// Implemented by methods whose internal propagation shares the nuclear
/// clock (e.g. `cpu::gpw::GPWEvaluator` under `cpmd = true`).
class TimestepConsumer {
  public:
    virtual ~TimestepConsumer() = default;

    /// @brief True when this consumer cannot run without a pushed dt
    ///        (e.g. `cpmd == true`). Gates the missing-provider throw.
    /// @return True to require a `TimestepProvider` in the same driver.
    virtual bool timestep_required() const = 0;

    /// @brief Receive the integrator timestep [fs] (before component build).
    /// @param dt_fs Nuclear MD timestep [fs].
    /// @throws std::invalid_argument when dt_fs is not finite and > 0.
    virtual void set_timestep(double dt_fs) = 0;
};

} // namespace cpu
} // namespace ATV
