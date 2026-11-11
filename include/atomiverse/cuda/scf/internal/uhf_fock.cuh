/// @file cuda/scf/internal/uhf_fock.cuh
/// @brief internal::UHFFockDevice — device 1e/ERI and unrestricted Fock.
///
/// F^σ = Hcore + J[Pα+Pβ] − K[P^σ], PySCF `scf.uhf.get_veff`.
/// Integrals are `launch_1e_integrals_sp` / `launch_eri_sp` (l ≤ 1).
/// No host ERI. Owned by `UhfScfCycle` only.
#pragma once

#include <vector>

#include "atomiverse/cuda/scf/internal/solver_context.cuh"
#include "atomiverse/cuda/scf/internal/uhf_scf_cycle.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

#ifdef ATV_HAS_HF

/// @brief Device s/p UHF Fock (setup uploads shells; step contracts ERI).
class UHFFockDevice {
  public:
    /// @param n_shells Molecular contracted shells.
    /// @param n_ao Cartesian AO count.
    /// @param n_prim Primitives per shell (STO-3G = 3).
    /// @param alpha_mol Unnormalized exponents [n_shells * n_prim].
    /// @param coeff_mol Unnormalized coefficients [n_shells * n_prim].
    /// @param L_mol Angular momentum per shell (0 or 1).
    /// @param atom_of_mol Atom index of each shell.
    /// @param n_alpha Alpha electrons.
    /// @param n_beta Beta electrons.
    UHFFockDevice(int n_shells, int n_ao, int n_prim, const double* alpha_mol,
                  const double* coeff_mol, const int* L_mol,
                  const int* atom_of_mol, int n_alpha, int n_beta);

    void build(const SolverContext& ctx, SCFWorkspace& ws, UhfBeta& beta);
    void step(const SolverContext& ctx, SCFWorkspace& ws, UhfBeta& beta);
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    int n_shells_ = 0;
    int n_ao_ = 0;
    int n_prim_ = 0;
    int n_alpha_ = 0;
    int n_beta_ = 0;
    std::vector<double> alpha_mol_;
    std::vector<double> coeff_mol_;
    std::vector<int> L_mol_;
    std::vector<int> atom_of_mol_;
    void* cublas_handle_ = nullptr;
    void* cusolver_handle_ = nullptr;
    double* d_temp_ = nullptr;
    double* d_T_ = nullptr;
    double* d_V_ = nullptr;
};

#endif

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
