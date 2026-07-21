/// @file cuda/scf/internal/solver_context.cuh
/// @brief SolverContext — the CUDA SCF inner solver's private execution
///        context. Deliberately CUDA-free header.
///
/// The solver layer's whole external surface is a CUDA stream plus the
/// device frame (arch-refactor-04-engines): the former component-layer
/// contexts carried workflow state (pair lists, exclusion CSR, outer
/// status) the inner solver must never see. Owning a minimal context type
/// makes that isolation a property of the type system — enforced by the
/// `test_cuda_scf_solver_isolation` grep gate — instead of a comment-level
/// promise.
#pragma once

#include "atomiverse/cuda/frame/frame.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

/// @brief Minimal execution context threaded through the SCF step helpers.
struct SolverContext {
    void* stream;       ///< CUDA stream (cudaStream_t, opaque to keep the
                        ///< header CUDA-free; cast in .cu files).
    cuda::Frame& frame; ///< Device-resident system state.
    int step;           ///< SCF iteration index (0 for build/destroy).
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
