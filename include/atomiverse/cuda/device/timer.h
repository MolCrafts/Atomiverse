/// @file cuda/device/timer.h
/// @brief cuda::Timer -- CUDA-free PIMPL facade for the GPU profiler.
///
/// The actual CUDA-event-based profiler lives in `Timer::Impl`, defined
/// in `src/cuda/device/timer.cu` together with `cuda/timer.cuh`. This header is
/// deliberately CUDA-free so that non-CUDA translation units (notably
/// `src/cpu/driver/driver.cpp`, which instantiates `Driver<PhaseEnum,
/// cpu::device_tag>`) can hold and destroy a `std::unique_ptr<Timer>`
/// member without a complete CUDA type. All special members are declared
/// here and defined out-of-line in `src/cuda/device/timer.cu`, where `Impl` is
/// complete.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ATV {
namespace cuda {

/// @brief Per-component GPU profiler (CUDA-free PIMPL facade).
///
/// Attach to a `cuda::Backend` via `backend.set_timer(&timer)`. Records
/// start/stop CUDA events around component launches and accumulates
/// elapsed times after each slot sync. Produces a LAMMPS-style summary
/// table. Enabled by `Driver::enable_timing()`, results via
/// `Driver::summary()`.
class Timer {
  public:
    /// @brief Construct an inert profiler. Call `init()` before use.
    Timer();

    /// @brief Destroy the profiler, releasing any CUDA events.
    ~Timer();

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    /// @brief Move-construct, transferring ownership of the impl.
    /// @param other Source profiler (left empty).
    Timer(Timer&& other) noexcept;

    /// @brief Move-assign, transferring ownership of the impl.
    /// @param other Source profiler (left empty).
    /// @return Reference to `*this`.
    Timer& operator=(Timer&& other) noexcept;

    /// @brief Allocate CUDA events.
    void init();

    /// @brief Mark one simulation step completed.
    void step_done();

    /// @brief Format a LAMMPS-style timing summary.
    /// @param phase_names Phase name strings indexed by phase number.
    /// @return Formatted timing table string [no units — text].
    std::string summary(const std::vector<std::string>& phase_names) const;

    /// @brief Destroy CUDA events and reset all accumulated state.
    void shutdown();

    /// @brief Access the underlying CUDA-dependent implementation.
    ///
    /// Used by `cuda::Backend` (a `.cu` translation unit) to record
    /// per-component events; never called from CUDA-free code.
    /// @return Non-owning pointer to the implementation (never null after
    ///         construction).
    class Impl;
    Impl* impl() { return impl_.get(); }

  private:
    std::unique_ptr<Impl> impl_; ///< CUDA-dependent profiler state.
};

} // namespace cuda
} // namespace ATV
