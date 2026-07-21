/// @file cuda/scf/internal/hamiltonian.cuh
/// @brief internal::Hamiltonian -- construct effective Hamiltonian H = H0 + alpha * P.
///
/// **Internal SCF helper.** Does NOT inherit `cuda::Component` and is
/// never registered into a `Driver` Plan. Owned by `cuda::scf::SCFSolver`,
/// which drives its `build` / `step` / `destroy` lifecycle directly.
///
/// Allocates and initializes the SCF workspace matrices (H0, H, P, P_old,
/// C, eig) in `build()`. Constructs the effective Hamiltonian each SCF
/// iteration in `step()`.
///
/// H0 is a deterministic diagonally-dominant symmetric matrix seeded from
/// atom indices. The coupling term alpha*P creates a self-consistent cycle.
///
/// @see cuda::scf::SCFSolver for the inner solver that owns this helper
/// @see internal::DensityBuilder for the helper that rebuilds P
#pragma once

#include "atomiverse/cuda/scf/internal/solver_context.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

/// @brief Construct effective Hamiltonian H(P) = H0 + alpha * P.
///
/// Also responsible for SCF workspace allocation in `build()`:
/// allocates H0, H, P, P_old, C, eig, work on device and initializes H0
/// as a deterministic symmetric matrix.
class Hamiltonian {
  public:
    /// @brief Construct Hamiltonian builder.
    /// @param n_occ Number of occupied orbitals [dimensionless].
    /// @param alpha Coupling parameter for H = H0 + alpha*P [dimensionless].
    Hamiltonian(int n_occ, double alpha) : n_occ_(n_occ), alpha_(alpha) {}

    /// @brief Allocate SCF workspace and initialize H0 on device.
    /// @param ctx Build context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void build(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Compute H = H0 + alpha * P via cuBLAS.
    /// @param ctx Launch context (CUDA stream + device frame + step index).
    /// @param ws SCF electronic-structure workspace.
    void step(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Free cuBLAS handle.
    /// @param ctx Destroy context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    int n_occ_;    ///< Number of occupied orbitals.
    double alpha_; ///< Density coupling parameter.
    void* cublas_handle_ =
        nullptr; ///< cuBLAS handle (cast to cublasHandle_t in .cu).
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
