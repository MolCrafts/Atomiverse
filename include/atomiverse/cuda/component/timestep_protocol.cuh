/// @file cuda/component/timestep_protocol.cuh
/// @brief TimestepProvider / TimestepConsumer — the CUDA mirror of
///        `cpu/component/timestep_protocol.h`. CUDA-free header.
///
/// The integrator owns the clock. A component that must stamp or advance on
/// the same clock (the `*.mrec` recorder's `time` axis) never re-declares a
/// dt of its own: it implements `TimestepConsumer` and `cuda::md::MD` pushes
/// the integrator's dt to it before any component `build()`. Two providers
/// on different clocks make `MD::build()` throw naming both — the same rule
/// and the same messages as the CPU driver.
#pragma once

namespace ATV {
namespace cuda {

/// @brief Supply side: a component that owns the MD integration step.
///
/// Implemented by the CUDA integrators (`VelocityVerlet`, `NVELimit`,
/// `NoseHooverNPT`, `NvtSllod`, `Brownian`).
class TimestepProvider {
  public:
    virtual ~TimestepProvider() = default;

    /// @brief Nuclear MD timestep [fs].
    /// @return The integrator's own step size [fs] (its constructor `dt`).
    virtual double timestep() const = 0;
};

/// @brief Demand side: a component that must know the nuclear dt.
class TimestepConsumer {
  public:
    virtual ~TimestepConsumer() = default;

    /// @brief True when this consumer cannot run without a pushed dt.
    /// @return True to require a `TimestepProvider` in the same driver.
    virtual bool timestep_required() const = 0;

    /// @brief Receive the integrator timestep [fs] (before component build).
    /// @param dt_fs Nuclear MD timestep [fs].
    /// @throws std::invalid_argument when dt_fs is not finite and > 0.
    virtual void set_timestep(double dt_fs) = 0;
};

} // namespace cuda
} // namespace ATV
