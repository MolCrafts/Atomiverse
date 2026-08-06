/// @file cuda/scf/internal/scf_solver.cuh
/// @brief cuda::scf::SCFSolver — **internal** SCF self-consistency loop.
///
/// Not a public user type. Concrete method is `cuda::scf::RHF` (Frame in/out,
/// parameters as members). Unit tests and the transitional synthetic
/// `cuda::scf::SCF` component may include this header.
///
/// **NOT a `Driver<...>` subclass** — owns `internal::*` step helpers with a
/// `build` / `step` / `destroy` lifecycle. External surface is
/// `(void* stream, cuda::Frame&)`. Isolation grep-gated by
/// `tests/cuda/scf_solver_isolation.sh` when present.
///
/// Paths: Synthetic `H(P)=H0+alpha*P`; RHF from host S/Hcore/ERI matrices;
/// RHF device integrals (s-type only). No integral pack/bag type.
#pragma once

#include <memory>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "atomiverse/cuda/scf/internal/density_builder.cuh"
#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"
#include "atomiverse/cuda/scf/internal/hamiltonian.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

#ifdef ATV_HAS_HF
#include "atomiverse/cuda/scf/internal/diis_mixer.cuh"
#include "atomiverse/cuda/scf/internal/rhf_density.cuh"
#include "atomiverse/cuda/scf/internal/rhf_fock.cuh"
#endif

namespace ATV {
namespace cuda {
namespace scf {

/// @brief Inner CUDA SCF self-consistency loop (internal).
///
/// Construction → `build(stream, frame)` → `run(stream, frame)`. Density is
/// zeroed at the start of `run()`. Prefer `cuda::scf::RHF` for user code.
class SCFSolver {
  public:
    /// @brief Synthetic-path solver `H(P) = H0 + alpha*P`.
    /// @param n_occ Occupied orbitals [dimensionless].
    /// @param alpha Density coupling [dimensionless].
    /// @param tol Frobenius residual tolerance [dimensionless].
    /// @param max_iter Max SCF iterations [dimensionless].
    SCFSolver(int n_occ, double alpha, double tol, int max_iter);

#ifdef ATV_HAS_HF
    /// @brief RHF from discrete host matrices (no integral pack/bag).
    /// @param S Overlap [n×n].
    /// @param Hcore Core Hamiltonian [n×n, Hartree].
    /// @param ERI Dense ERI [n⁴, Hartree].
    /// @param V_nuc Nuclear repulsion [Hartree].
    /// @param n_electrons Electron count (must be even, ≥0).
    /// @param tol Frobenius residual tolerance [dimensionless].
    /// @param max_iter Max SCF iterations [dimensionless].
    /// @param level_shift Virtual level shift λ [Hartree]; default 0.
    SCFSolver(xt::xtensor<double, 2> S, xt::xtensor<double, 2> Hcore,
              xt::xtensor<double, 4> ERI, double V_nuc, int n_electrons,
              double tol, int max_iter, double level_shift = 0.0);

    /// @brief RHF from GPU-native s-shell kernels (one s shell / atom).
    /// @param level_shift Virtual level shift λ [Hartree]; default 0.
    SCFSolver(int n_prim, const double* exponents, const double* coefficients,
              int n_occ, double tol, int max_iter, double level_shift = 0.0);

    /// @brief RHF from GPU-native multi-shell pure-s (Wave 3).
    /// @param n_shells_per_atom s shells per atom from the basis table.
    /// @param n_prim Primitives per shell (uniform).
    /// @param alpha_atom [n_shells_per_atom * n_prim] unnormalized α.
    /// @param coeff_atom [n_shells_per_atom * n_prim] unnormalized c.
    /// @param level_shift Virtual level shift λ [Hartree]; default 0.
    SCFSolver(int n_shells_per_atom, int n_prim, const double* alpha_atom,
              const double* coeff_atom, int n_occ, double tol, int max_iter,
              double level_shift = 0.0);

    /// @brief RHF from GPU-native molecular s+p shells (Wave 4).
    /// @param n_shells Shell count for the whole molecule.
    /// @param n_ao Cartesian AO count (Σ n_cart(L)).
    /// @param n_prim Primitives per shell (uniform).
    /// @param alpha_mol [n_shells * n_prim] unnormalized α.
    /// @param coeff_mol [n_shells * n_prim] unnormalized c.
    /// @param L_mol [n_shells] angular momentum (0 or 1).
    /// @param atom_of_mol [n_shells] atom index per shell.
    /// @param level_shift Virtual level shift λ [Hartree]; default 0.
    SCFSolver(int n_shells, int n_ao, int n_prim, const double* alpha_mol,
              const double* coeff_mol, const int* L_mol, const int* atom_of_mol,
              int n_occ, double tol, int max_iter, double level_shift = 0.0);
#endif

    /// @brief One-time device workspace allocation.
    /// @param stream CUDA stream (`cudaStream_t` as `void*`).
    /// @param frame Device-resident system state.
    void build(void* stream, cuda::Frame& frame);

    /// @brief Run the SCF cycle to convergence or `max_iter`.
    /// @param stream CUDA stream (`cudaStream_t` as `void*`).
    /// @param frame Device frame; energy scalar updated in place.
    void run(void* stream, cuda::Frame& frame);

    /// @brief Free device workspaces and library handles.
    /// @param stream CUDA stream (`cudaStream_t` as `void*`).
    /// @param frame Device frame.
    void destroy(void* stream, cuda::Frame& frame);

    /// @brief Iterations executed by the most recent `run()`.
    int iterations() const { return iterations_; }

    /// @brief Device SCF workspace (after `build()`).
    const SCFWorkspace& workspace() const { return workspace_; }

  private:
    enum class Path {
        Synthetic,
        RHFHostIntegrals,
        RHFDeviceIntegrals,
    };

    Path path_;
    double tol_;
    int max_iter_;
    double level_shift_ = 0.0;
    bool built_ = false;
    int iterations_ = 0;

    SCFWorkspace workspace_;

    std::unique_ptr<internal::EigenSolver> eigen_;
    std::unique_ptr<internal::ConvergenceCheck> convergence_;

    std::unique_ptr<internal::Hamiltonian> hamiltonian_;
    std::unique_ptr<internal::DensityBuilder> density_;

#ifdef ATV_HAS_HF
    std::unique_ptr<internal::RHFFock> rhf_fock_;
    std::unique_ptr<internal::RHFFockDevice> rhf_fock_device_;
    std::unique_ptr<internal::RHFDensity> rhf_density_;
    /// Device CDIIS + level shift (CPU MixStage twin). RHF paths only.
    std::unique_ptr<internal::DIISMixer> diis_;
#endif
};

} // namespace scf
} // namespace cuda
} // namespace ATV
