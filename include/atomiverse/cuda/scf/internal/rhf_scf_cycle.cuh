/// @file cuda/scf/internal/rhf_scf_cycle.cuh
/// @brief internal::RhfScfCycle — private RHF workspace assembly.
///
/// Owned **only** by `cuda::scf::RHF`. Not a Driver and not a loop: the
/// SCF iteration is `cuda::Driver<scf::Phase>` (`fock → mix → diagonalize
/// → density → check`). This type builds the device workspace and runs
/// exactly one stage per call. PySCF `scf.hf.kernel` is the loop shape;
/// `scf.diis.CDIIS.update` is the mix stage (device CDIIS).
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

/// @brief Private CUDA RHF workspace (not a Driver; owned by RHF only).
///
/// Construction configures the host-matrix or device-integral path.
/// `build` allocates; `begin` zeros the density; each `step_*` is one
/// `scf::Phase`. The Driver owns iteration and convergence.
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
                const double* coeff_mol, const int* L_mol,
                const int* atom_of_mol, int n_occ, double tol, int max_iter,
                double level_shift = 0.0);

    /// @brief Device RHF from ragged Cartesian shells (l ≤ 3).
    ///
    /// Raw table coefficients. See `RHFFockDevice`'s ragged constructor.
    /// @param n_prim_total Length of the flat `alpha` / `coeff` lists.
    /// @param spherical Sandwich Cartesian integrals to the spherical AO basis.
    RhfScfCycle(int n_shells, int n_ao, const double* alpha,
                const double* coeff, int n_prim_total, const int* nprim,
                const int* prim_off, const int* L_mol, const int* atom_of_mol,
                int n_occ, double tol, int max_iter, double level_shift = 0.0,
                bool spherical = false);

    /// @brief One-time device workspace allocation.
    void build(void* stream, cuda::Frame& frame);

    /// @brief Zero the density and DIIS history before the first stage.
    /// @param stream CUDA stream (cudaStream_t).
    void begin(void* stream);

    /// @brief `scf::Phase::fock` — build the unshifted AO Fock.
    void step_fock(void* stream, cuda::Frame& frame, int step);
    /// @brief `scf::Phase::mix` — CDIIS extrapolation + level shift.
    void step_mix(void* stream, cuda::Frame& frame, int step);
    /// @brief `scf::Phase::diagonalize` — generalized eigenproblem.
    void step_diagonalize(void* stream, cuda::Frame& frame, int step);
    /// @brief `scf::Phase::density` — occupied density + energy.
    void step_density(void* stream, cuda::Frame& frame, int step);
    /// @brief `scf::Phase::check` — density residual (device flag or host).
    void step_check(void* stream, cuda::Frame& frame, int step);

    /// @brief Copy the device convergence flag onto the host workspace.
    /// @param stream CUDA stream the check kernel was launched on.
    /// @param step SCF iteration index [dimensionless].
    /// @return True when this iteration met `tol`.
    bool poll_converged(void* stream, int step);

    /// @brief Post-loop virtual-orbital de-shift on device.
    /// @param stream CUDA stream.
    void finish(void* stream);

    /// @brief Free device workspaces and library handles.
    void destroy(void* stream, cuda::Frame& frame);

    /// @brief Device SCF workspace (after `build()`).
    const SCFWorkspace& workspace() const { return workspace_; }

    /// @brief True on the device s/p integral path (no host ERI).
    bool device_integrals() const { return path_ == Path::RHFDeviceIntegrals; }

  private:
    enum class Path { RHFHostIntegrals, RHFDeviceIntegrals };

    Path path_;
    double tol_;
    double level_shift_ = 0.0;
    bool built_ = false;
    bool polishing_ = false;
    bool canonicalized_ = false;

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
