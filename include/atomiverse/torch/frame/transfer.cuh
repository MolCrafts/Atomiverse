/// @file torch/frame/transfer.cuh
/// @brief Transfer functions between torch::Frame and cpu::Frame / cuda::Frame.
///
/// Symmetric with cpu::to_cuda() and cuda::to_cpu().
///
/// @see torch::Frame for the Tensor-based system state
/// @see cuda::Frame for the raw-pointer CUDA counterpart
/// @see cpu::Frame for the host-side counterpart
#pragma once

#ifdef ATV_HAS_TORCH

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/torch/frame/frame.cuh"

namespace ATV {
namespace torch {

/// @brief Wrap a cuda::Frame as a torch::Frame (zero-copy borrow).
///
/// Tensors wrap existing device pointers via torch::from_blob. The system
/// scalars (energy, kinetic, conserved, stress, stress_kinetic) are wrapped
/// as views into the cuda::Frame's device-resident `SystemScalars` struct.
/// The cuda::Frame owns the memory; the torch::Frame is a non-owning view.
/// Caller must ensure the cuda::Frame outlives the returned torch::Frame.
///
/// @param src CUDA Frame to wrap.
/// @return torch::Frame with non-owning Tensor views.
Frame to_torch(const cuda::Frame& src);

/// @brief Extract pointers from a torch::Frame into a cuda::Frame.
///
/// Validates that all tensors are CUDA, contiguous, and correct dtype.
/// Non-owning: the cuda::Frame borrows pointers from the torch::Frame.
/// The system-scalar Tensors are copied (D2D) into the cuda::Frame's
/// device-resident `SystemScalars` struct, which must already be allocated.
///
/// @param src  torch::Frame to extract from.
/// @param dst  cuda::Frame to write pointers into.
void to_cuda(const Frame& src, cuda::Frame& dst);

/// @brief Copy torch::Frame to cpu::Frame (D2H transfer).
///
/// @param src torch::Frame (CUDA tensors) to read from.
/// @param dst cpu::Frame to write into (must be pre-sized).
void to_cpu(const Frame& src, cpu::Frame& dst);

/// @brief Copy cpu::Frame to a new torch::Frame (H2D transfer).
///
/// Allocates new CUDA Tensors and copies from host xtensor arrays.
///
/// @param src CPU Frame to transfer.
/// @return Newly allocated torch::Frame with data on CUDA device.
Frame from_cpu(const cpu::Frame& src);

/// @brief Reset all Tensors in a torch::Frame to empty.
///
/// @param frame torch::Frame to clear.
void destroy(Frame& frame);

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH
