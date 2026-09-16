/// @file cuda/scf/internal/synthetic_scf_cycle.cuh
/// @brief internal::SyntheticScfCycle — toy H(P)=H0+αP cycle for unit tests.
///
/// Owned **only** by transitional `cuda::scf::SCF`. Not molecular RHF and not
/// a Driver. Loop order: fock(H) → diagonalize → density → check (no DIIS mix).
#pragma once

#include <memory>

#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "atomiverse/cuda/scf/internal/density_builder.cuh"
#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"
#include "atomiverse/cuda/scf/internal/hamiltonian.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

/// @brief Synthetic SCF cycle for `cuda::scf::SCF` tests only.
class SyntheticScfCycle {
  public:
    /// @param n_occ Occupied orbitals [dimensionless].
    /// @param alpha Density coupling [dimensionless].
    /// @param tol Frobenius residual tolerance [dimensionless].
    /// @param max_iter Max SCF iterations [dimensionless].
    SyntheticScfCycle(int n_occ, double alpha, double tol, int max_iter);

    void build(void* stream, cuda::Frame& frame);
    void drive(void* stream, cuda::Frame& frame);
    void destroy(void* stream, cuda::Frame& frame);

    int iterations() const { return iterations_; }
    const SCFWorkspace& workspace() const { return workspace_; }

  private:
    double tol_;
    int max_iter_;
    bool built_ = false;
    int iterations_ = 0;

    SCFWorkspace workspace_;
    std::unique_ptr<EigenSolver> eigen_;
    std::unique_ptr<ConvergenceCheck> convergence_;
    std::unique_ptr<Hamiltonian> hamiltonian_;
    std::unique_ptr<DensityBuilder> density_;
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
