/// @file cpu/device/device_runtime.h
/// @brief cpu::device::DeviceRuntime — owns the CPU "device" setup surface:
///        OpenMP thread configuration and the per-component TimerRegistry.
///
/// backend-mirror-05-dissolve pulled this out of the former CPU execution
/// engine. On CPU the "device" is the host: thread count + timing. The mirror
/// of `cuda/device/` (streams, device manager) on the CPU backend.
#pragma once

#include "atomiverse/core/timer.h"

namespace ATV {
namespace cpu {

class Context;

namespace device {

/// @brief Owns CPU thread configuration and the timing registry.
class DeviceRuntime {
  public:
    /// @brief Apply the OpenMP thread count (honouring `ctx.n_threads` when set,
    ///        else adopting the max) and warn once on unbound NUMA threads.
    void configure_threads(Context& ctx);

    /// @brief Publish the timing registry on the context and apply the remembered
    ///        enable preference (overridable by the `ATV_TIMER` env var).
    void init_timing(Context& ctx);

    /// @brief The per-component timing registry (opt-in). @see core/timer.h
    TimerRegistry& timing() { return timing_; }
    const TimerRegistry& timing() const { return timing_; }

    /// @brief Enable/disable per-launch timing; the preference sticks across
    ///        re-init.
    void enable_timing(bool on) {
        timing_pref_ = on;
        timing_.set_enabled(on);
    }

    /// @brief Clear accumulated step timing.
    void reset_timing() { timing_.reset(); }

    /// @brief Cap the OpenMP thread count. Must be called before configure_threads.
    void set_threads(Context& ctx, int n);

  private:
    TimerRegistry timing_{};
    bool timing_pref_ = true;
};

} // namespace device
} // namespace cpu
} // namespace ATV
