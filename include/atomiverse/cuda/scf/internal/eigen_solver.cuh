/// @file cuda/scf/internal/eigen_solver.cuh
/// @brief internal::EigenSolver -- device symmetric eigendecomposition.
///
/// **Internal SCF helper.** Does NOT inherit `cuda::Component` and is
/// never registered into a `Driver` Plan. Owned by `the concrete method (cuda::scf::RHF) private cycle`,
/// which drives its `build` / `step` / `destroy` lifecycle directly.
///
/// Diagonalizes the effective Hamiltonian H each SCF iteration:
///   H * C = C * diag(eig)
///
/// Uses sorted FP64 cuSOLVER batched Jacobi through order 64, and an
/// explicitly device-controlled cyclic Jacobi kernel for orders 65..215.
/// Both evolving Jacobi matrices use shared memory when device capacity
/// permits; larger matrices retain the same algorithm in device memory.
/// Only mapped integer success/failure state is inspected by the host on
/// these paths. Larger generic/synthetic matrices retain legacy syevd and
/// its internal norm readback; molecular HF rejects those dimensions before
/// this helper, so that branch is outside the CUDA-only HF product envelope.
///
/// @see the concrete method (cuda::scf::RHF) private cycle for the inner solver that owns this helper
/// @see internal::Hamiltonian for the helper that builds H
#pragma once

#include <cstddef>

#include "atomiverse/cuda/scf/internal/solver_context.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

/// @brief Symmetric eigendecomposition; CUDA-only molecular envelope 1..215.
///
/// In `build()`, reserves resources for the selected device solver.
/// In `step()`, diagonalizes H's lower triangle with device-owned scratch.
/// Eigenvalues written to the SCF workspace `ws.eig`, eigenvectors to `ws.C`.
class EigenSolver {
  public:
    /// @brief Reserve library/global scratch or opt in to shared matrices.
    /// @param ctx Build context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void build(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Diagonalize H, writing sorted eigenvectors to C and values to eig.
    /// @param ctx Launch context (CUDA stream + device frame + step index).
    /// @param ws SCF electronic-structure workspace.
    void step(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Free cuSOLVER handle and workspace.
    /// @param ctx Destroy context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Global device workspace reserved in setup [bytes].
    /// @return Library or Jacobi scratch, excluding SCF and shared matrices.
    std::size_t device_workspace_bytes() const { return device_work_bytes_; }
    /// @brief Host workspace requested by the selected library algorithm [bytes].
    /// @return Zero when the library requires no host workspace.
    std::size_t host_workspace_bytes() const { return 0; }

  private:
    void* cusolver_handle_ = nullptr; ///< cuSOLVER handle.
    void* jacobi_params_ = nullptr; ///< Parameters for the small cuSOLVER path.
    void* d_work_ = nullptr; ///< Library scratch or custom Jacobi matrix.
    std::size_t device_work_bytes_ = 0; ///< Reserved device workspace [bytes].
    std::size_t shared_matrix_bytes_ = 0; ///< Jacobi dynamic shared bytes/CTA.
    int library_work_elements_ = 0; ///< Small-matrix library scratch length.
    int n_ = 0;                     ///< Matrix order reserved in build.
    int* d_info_ = nullptr; ///< Device alias of the mapped control status.
    int* h_info_ = nullptr; ///< Host-mapped integer eigensolver status.
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
