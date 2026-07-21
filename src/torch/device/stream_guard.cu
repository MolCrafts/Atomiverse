/// @file torch/device/stream_guard.cu
/// @brief StreamGuard implementation.

#ifdef ATV_HAS_TORCH

#include "atomiverse/torch/device/stream_guard.cuh"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace ATV {
namespace torch {

StreamGuard::StreamGuard(void* atv_stream)
    : prev_stream_(::c10::cuda::getCurrentCUDAStream()) {
    auto stream = ::c10::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(atv_stream), prev_stream_.device_index());
    ::c10::cuda::setCurrentCUDAStream(stream);
}

StreamGuard::~StreamGuard() { ::c10::cuda::setCurrentCUDAStream(prev_stream_); }

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH
