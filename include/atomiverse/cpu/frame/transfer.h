/// @file cpu/frame/transfer.h
/// @brief cpu::to_cuda() -- transfer CPU Frame to CUDA Frame.
///
/// CUDA-free declaration (implementation in src/cuda/frame/transfer.cu).
///
/// @see cpu::Frame for the CPU device data
/// @see cuda::Frame for the CUDA device data
/// @see cuda::to_cpu() for the reverse direction
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/frame/frame.cuh"

namespace ATV {
namespace cpu {

/// @brief Transfer CPU Frame to a new CUDA Frame (allocate + H2D copy).
///
/// Allocates device memory for all per-atom arrays, scalars, box, and topology.
/// PairList starts empty (nullptr) -- allocated later by NeighborList::build().
///
/// @param src CPU Frame to transfer.
/// @param capacity  If greater than `src.nlocal()`, per-atom arrays are allocated to
///        `capacity` slots (only size() values copied), leaving headroom for
///        multi-GPU halo ghosts appended in [n_local, capacity). 0 ⇒ exact.
/// @return Newly allocated cuda::Frame with all data on device.
cuda::Frame to_cuda(const Frame& src, int capacity = 0);

} // namespace cpu
} // namespace ATV
