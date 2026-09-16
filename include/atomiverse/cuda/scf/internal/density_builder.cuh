/// @file cuda/scf/internal/density_builder.cuh
/// @brief internal::DensityBuilder -- rebuild density matrix P = C_occ * C_occ^T.
///
/// **Internal SCF helper.** Does NOT inherit `cuda::Component` and is
/// never registered into a `Driver` Plan. Owned by `the concrete method (cuda::scf::RHF) private cycle`,
/// which drives its `build` / `step` / `destroy` lifecycle directly.
///
/// Each SCF iteration, saves the current density to P_old, then rebuilds P
/// from the first n_occ eigenvectors using cuBLAS syrk.
///
/// @see the concrete method (cuda::scf::RHF) private cycle for the inner solver that owns this helper
/// @see internal::EigenSolver for the helper that produces eigenvectors C
#pragma once

#include "atomiverse/cuda/scf/internal/solver_context.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

/// @brief Rebuild density matrix P from occupied eigenvectors.
///
/// P = C[:, 0:n_occ] * C[:, 0:n_occ]^T  (cuBLAS ssyrk/dsyrk)
///
/// Before rebuilding, copies P -> P_old for convergence checking.
class DensityBuilder {
  public:
    /// @brief Allocate cuBLAS handle.
    /// @param ctx Build context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void build(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Copy P -> P_old, then rebuild P = C_occ * C_occ^T.
    /// @param ctx Launch context (CUDA stream + device frame + step index).
    /// @param ws SCF electronic-structure workspace.
    void step(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Free cuBLAS handle.
    /// @param ctx Destroy context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    void* cublas_handle_ = nullptr; ///< cuBLAS handle.
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
