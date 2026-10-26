/// @file cuda/scf/internal/convergence_check.cuh
/// @brief Density convergence and physical HF convergence measurements.
///
/// **Internal SCF helper.** Does NOT inherit `cuda::Component` and is
/// never registered into a `Driver` Plan. Owned by `the concrete method (cuda::scf::RHF) private cycle`,
/// which drives its `build` / `step` / `destroy` lifecycle directly.
///
/// Synthetic step() measures ||P-P_old||_F. Physical step_hf() also requires
/// a small energy change and pre-extrapolation Fock commutator. All HF norms,
/// energy history and comparisons stay on device; only a mapped control flag
/// is observed by the owning Driver's criterion.
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

/// @brief Density convergence for synthetic paths; physical dual check for HF.
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

    /// @brief HF acceptance requires density, energy and unmixed orbital error.
    ///
    /// The error matrices are the pre-extrapolation CDIIS commutators
    /// X^T(FPS-SPF)X. Energy must change by less than tol [Hartree], the
    /// combined error norm by less than sqrt(tol), matching the CPU default,
    /// and the density change must also remain below the CUDA density tol.
    /// All measurements/history remain on device.
    void step_hf(const SolverContext& ctx, SCFWorkspace& ws,
                 const prec_t* error_alpha, const prec_t* error_beta = nullptr,
                 const prec_t* density_beta = nullptr,
                 const prec_t* old_density_beta = nullptr);

    /// @brief Reset device energy history before an independent HF solve.
    void reset(void* stream);

    /// @brief Free cuBLAS handle.
    /// @param ctx Destroy context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    double tol_;                    ///< Convergence tolerance (Frobenius norm).
    bool compute_energy_;           ///< Whether to compute and write energy.
    void* cublas_handle_ = nullptr; ///< cuBLAS handle.
    prec_t* d_difference_ = nullptr; ///< Reused alpha/beta density difference.
    prec_t* d_metrics_ =
        nullptr; ///< Previous E[eV], density/error norms, dE[Ha].
    int n_ = 0;
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
