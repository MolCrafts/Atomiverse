/// @file cuda/scf/internal/convergence_check.cuh
/// @brief internal::ConvergenceCheck -- SCF convergence via Frobenius norm
///        of P - P_old.
///
/// **Internal SCF helper.** Does NOT inherit `cuda::Component` and is
/// never registered into a `Driver` Plan. Owned by `the concrete method (cuda::scf::RHF) private cycle`,
/// which drives its `build` / `step` / `destroy` lifecycle directly.
///
/// Computes ||P - P_old||_F using cuBLAS (axpy + nrm2) and compares against
/// the specified tolerance. Sets the SCF workspace `ws.converged` when
/// satisfied.
///
/// When compute_energy is true (default), also computes E = dot(H, P) and
/// writes it to frame.scalars->energy. Set to false when another helper (e.g.
/// internal::RHFDensity) computes the proper energy.
///
/// @see the concrete method (cuda::scf::RHF) private cycle for the inner solver that owns this helper
/// @see internal::DensityBuilder for the helper that produces P and P_old
#pragma once

#include "atomiverse/cuda/scf/internal/solver_context.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

/// @brief SCF convergence check: ||P_new - P_old||_F < tol.
///
/// Optionally computes SCF energy E = trace(H * P) via cuBLAS dot.
class ConvergenceCheck {
  public:
    /// @brief Construct convergence checker.
    /// @param tol Frobenius-norm tolerance for convergence [dimensionless].
    /// @param compute_energy If true, compute E=dot(H,P) and write to
    ///        frame.scalars->energy.
    explicit ConvergenceCheck(double tol, bool compute_energy = true)
        : tol_(tol), compute_energy_(compute_energy) {}

    /// @brief Allocate cuBLAS handle.
    /// @param ctx Build context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void build(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Compute ||P - P_old||_F, check convergence, optionally compute
    ///        energy.
    /// @param ctx Launch context (CUDA stream + device frame + step index).
    /// @param ws SCF electronic-structure workspace.
    void step(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Free cuBLAS handle.
    /// @param ctx Destroy context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    double tol_;                    ///< Convergence tolerance (Frobenius norm).
    bool compute_energy_;           ///< Whether to compute and write energy.
    void* cublas_handle_ = nullptr; ///< cuBLAS handle.
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
