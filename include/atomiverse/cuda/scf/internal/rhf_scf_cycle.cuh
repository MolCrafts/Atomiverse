/// @file cuda/scf/internal/rhf_scf_cycle.cuh
/// @brief internal::RhfScfCycle — private RHF self-consistency cycle.
///
/// Owned **only** by `cuda::scf::RHF`. Not a public type and not a
/// `Driver` subclass. Drives helpers in `scf::Phase` order:
///   fock → mix → diagonalize → density → check
/// matching MeanFieldMethod / GpwDft (arch-cuda-scf-solver-fold).
/// External surface is `(void* stream, cuda::Frame&)`.
#pragma once

#include <memory>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "atomiverse/cuda/scf/internal/diis_mixer.cuh"
#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"
#include "atomiverse/cuda/scf/internal/rhf_density.cuh"
#include "atomiverse/cuda/scf/internal/rhf_fock.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

/// @brief Private CUDA RHF SCF cycle (not a Driver; owned by RHF only).
///
/// Construction configures host-matrix or device-integral path →
/// `build(stream, frame)` → `drive(stream, frame)`. Density is zeroed at
/// the start of `drive()`. Prefer `cuda::scf::RHF` for user code.
class RhfScfCycle {
  public:
    /// @brief RHF from discrete host matrices (no integral pack/bag).
    /// @param S Overlap [n×n].
    /// @param Hcore Core Hamiltonian [n×n, Hartree].
    /// @param ERI Dense ERI [n⁴, Hartree].
    /// @param V_nuc Nuclear repulsion [Hartree].
    /// @param n_electrons Electron count (must be even, ≥0).
    /// @param tol Frobenius residual tolerance [dimensionless].
    /// @param max_iter Max SCF iterations [dimensionless].
    /// @param level_shift Virtual level shift λ [Hartree]; default 0.
    RhfScfCycle(xt::xtensor<double, 2> S, xt::xtensor<double, 2> Hcore,
                xt::xtensor<double, 4> ERI, double V_nuc, int n_electrons,
                double tol, int max_iter, double level_shift = 0.0);

    /// @brief RHF from GPU-native s-shell kernels (one s shell / atom).
    RhfScfCycle(int n_prim, const double* exponents, const double* coefficients,
                int n_occ, double tol, int max_iter, double level_shift = 0.0);

    /// @brief RHF from GPU-native multi-shell pure-s.
    RhfScfCycle(int n_shells_per_atom, int n_prim, const double* alpha_atom,
                const double* coeff_atom, int n_occ, double tol, int max_iter,
                double level_shift = 0.0);

    /// @brief RHF from GPU-native molecular s+p shells.
    RhfScfCycle(int n_shells, int n_ao, int n_prim, const double* alpha_mol,
                const double* coeff_mol, const int* L_mol, const int* atom_of_mol,
                int n_occ, double tol, int max_iter, double level_shift = 0.0);

    /// @brief One-time device workspace allocation.
    void build(void* stream, cuda::Frame& frame);

    /// @brief Drive SCF in scf::Phase order to convergence or max_iter.
    void drive(void* stream, cuda::Frame& frame);

    /// @brief Free device workspaces and library handles.
    void destroy(void* stream, cuda::Frame& frame);

    /// @brief Iterations executed by the most recent `drive()`.
    int iterations() const { return iterations_; }

    /// @brief Device SCF workspace (after `build()`).
    const SCFWorkspace& workspace() const { return workspace_; }

  private:
    enum class Path { RHFHostIntegrals, RHFDeviceIntegrals };

    Path path_;
    double tol_;
    int max_iter_;
    double level_shift_ = 0.0;
    bool built_ = false;
    int iterations_ = 0;

    SCFWorkspace workspace_;

    std::unique_ptr<EigenSolver> eigen_;
    std::unique_ptr<ConvergenceCheck> convergence_;
    std::unique_ptr<RHFFock> rhf_fock_;
    std::unique_ptr<RHFFockDevice> rhf_fock_device_;
    std::unique_ptr<RHFDensity> rhf_density_;
    std::unique_ptr<DIISMixer> diis_;
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
