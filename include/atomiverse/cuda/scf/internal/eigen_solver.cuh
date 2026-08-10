/// @file cuda/scf/internal/eigen_solver.cuh
/// @brief internal::EigenSolver -- dense symmetric eigendecomposition via cuSOLVER.
///
/// **Internal SCF helper.** Does NOT inherit `cuda::Component` and is
/// never registered into a `Driver` Plan. Owned by `the concrete method (cuda::scf::RHF) private cycle`,
/// which drives its `build` / `step` / `destroy` lifecycle directly.
///
/// Diagonalizes the effective Hamiltonian H each SCF iteration:
///   H * C = C * diag(eig)
///
/// Uses cusolverDnSsyevd (float) or cusolverDnDsyevd (double) depending
/// on compile-time precision.
///
/// @see the concrete method (cuda::scf::RHF) private cycle for the inner solver that owns this helper
/// @see internal::Hamiltonian for the helper that builds H
#pragma once

#include "atomiverse/cuda/scf/internal/solver_context.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

/// @brief Dense symmetric eigendecomposition via cuSOLVER syevd.
///
/// In `build()`, allocates the cuSOLVER workspace.
/// In `step()`, copies H -> C then solves in-place.
/// Eigenvalues written to the SCF workspace `ws.eig`, eigenvectors to `ws.C`.
class EigenSolver {
  public:
    /// @brief Allocate cuSOLVER handle and workspace.
    /// @param ctx Build context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void build(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Diagonalize H: copy H -> C, solve C in-place, eigenvalues to eig.
    /// @param ctx Launch context (CUDA stream + device frame + step index).
    /// @param ws SCF electronic-structure workspace.
    void step(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Free cuSOLVER handle and workspace.
    /// @param ctx Destroy context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    void* cusolver_handle_ = nullptr; ///< cuSOLVER handle.
    double* d_work_ = nullptr;        ///< cuSOLVER workspace [device].
    int work_size_ = 0;               ///< Workspace size in elements.
    int* d_info_ = nullptr; ///< cuSOLVER convergence info [device, 1 element].
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
