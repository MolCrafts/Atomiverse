/// @file torch/component/context.cuh
/// @brief torch::Context -- torch-specific execution context.
///
/// Holds autograd configuration for the torch backend.
/// Parallel to cpu::Context and cuda::Context.
#pragma once

#ifdef ATV_HAS_TORCH

namespace ATV {
namespace torch {

/// @brief Torch-specific execution context.
///
/// Controls autograd behavior for torch backend operations.
/// Future: model registry, cached JIT modules, custom allocator config.
struct Context {
    bool grad_enabled = true; ///< Whether autograd graph is constructed.
};

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH
