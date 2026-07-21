/// @file torch/device/stream_guard.cuh
/// @brief StreamGuard -- RAII guard for ATV-to-Torch CUDA stream interop.
///
/// Sets the current Torch CUDA stream to an ATV stream (void*),
/// restoring the previous stream on destruction.
#pragma once

#ifdef ATV_HAS_TORCH

#include <c10/cuda/CUDAStream.h>

#include "atomiverse/torch/frame/frame.cuh"

namespace ATV {
namespace torch {

/// @brief CUDA stream type alias (moved here to keep frame.cuh CUDA-free).
using Stream = ::c10::cuda::CUDAStream;

/// @brief RAII guard that sets the current Torch CUDA stream to an ATV stream.
///
/// Usage inside a cross-device component's launch():
/// @code{.cpp}
/// void launch(const LaunchContext& ctx) override {
///     ATV::torch::StreamGuard guard(ctx.runtime.stream);
///     // All Torch ops now execute on the simulation's stream
/// }
/// @endcode
class StreamGuard {
  public:
    /// @brief Set the current Torch CUDA stream.
    /// @param atv_stream ATV stream handle (void*, cast to cudaStream_t internally).
    explicit StreamGuard(void* atv_stream);

    /// @brief Restore the previous Torch CUDA stream.
    ~StreamGuard();

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

  private:
    Stream prev_stream_; ///< Previous stream to restore on destruction.
};

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH
