/// @file cuda/scf/internal/rhf_fock.cuh
/// @brief internal::RHFFock — F = Hcore + G(P), F' = Xᵀ F X.
///
/// Takes **discrete host matrices** (S, Hcore, ERI), not an integral pack/bag.
/// Owned by `the concrete method (cuda::scf::RHF) private cycle` only.
#pragma once

#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cuda/scf/internal/solver_context.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

#ifdef ATV_HAS_HF

/// @brief Build Fock from host S/Hcore/ERI uploads (discrete matrices only).
class RHFFock {
  public:
    /// @param S Overlap [n×n].
    /// @param Hcore Core Hamiltonian [n×n, Hartree].
    /// @param ERI Dense (ij|kl) [n⁴, Hartree].
    /// @param V_nuc Nuclear repulsion [Hartree].
    /// @param n_occ Doubly-occupied orbitals.
    /// @param level_shift Virtual shift λ [Hartree] (0 = off).
    RHFFock(xt::xtensor<double, 2> S, xt::xtensor<double, 2> Hcore,
            xt::xtensor<double, 4> ERI, double V_nuc, int n_occ,
            double level_shift = 0.0)
        : S_(std::move(S)), Hcore_(std::move(Hcore)), ERI_(std::move(ERI)),
          V_nuc_(V_nuc), n_basis_(static_cast<int>(S_.shape()[0])),
          n_occ_(n_occ), level_shift_(level_shift) {}

    void build(const SolverContext& ctx, SCFWorkspace& ws);
    void step(const SolverContext& ctx, SCFWorkspace& ws);
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    xt::xtensor<double, 2> S_;
    xt::xtensor<double, 2> Hcore_;
    xt::xtensor<double, 4> ERI_;
    double V_nuc_ = 0.0;
    int n_basis_ = 0;
    int n_occ_ = 0;
    double level_shift_ = 0.0;
    void* cublas_handle_ = nullptr;
    double* d_temp_ = nullptr;
};

/// @brief Fock build with device integral kernels (s + p, l ≤ 1).
///
/// Wave 4: molecular shell list (mixed Z OK). Host provides per-shell α/c/L
/// and atom ownership; geometry centers expand on device from atom positions.
/// AO count is `n_ao` (s→1, p→3 cart AOs per shell), not n_shells.
class RHFFockDevice {
  public:
    /// @brief Molecular s/p shell basis → device 1e + ERI + Löwdin X.
    ///
    /// @param n_shells Number of contracted shells in the molecule.
    /// @param n_ao Total Cartesian AOs (Σ n_cart(L)).
    /// @param n_prim Primitives per shell (uniform; STO-3G = 3).
    /// @param alpha_mol Host [n_shells * n_prim] unnormalized α.
    /// @param coeff_mol Host [n_shells * n_prim] unnormalized c.
    /// @param L_mol Host [n_shells] angular momentum (0 or 1).
    /// @param atom_of_mol Host [n_shells] atom index for each shell center.
    /// @param n_occ Doubly occupied MOs.
    /// @param level_shift Virtual shift λ [Hartree] (0 = off).
    RHFFockDevice(int n_shells, int n_ao, int n_prim, const double* alpha_mol,
                  const double* coeff_mol, const int* L_mol,
                  const int* atom_of_mol, int n_occ, double level_shift = 0.0);

    /// @brief Multi-shell pure-s convenience (Wave 3): one element type, all L=0.
    /// @param n_shells_per_atom s shells per atom (table order).
    /// @param n_prim Primitives per shell.
    /// @param alpha_atom Host [n_shells_per_atom * n_prim].
    /// @param coeff_atom Host [n_shells_per_atom * n_prim].
    /// @param n_occ Doubly occupied MOs.
    /// @param natm Atom count (builds atom_of = shell / n_shells_per_atom).
    /// @param level_shift Virtual shift λ [Hartree] (0 = off).
    RHFFockDevice(int n_shells_per_atom, int n_prim, const double* alpha_atom,
                  const double* coeff_atom, int n_occ, int natm,
                  double level_shift = 0.0);

    /// @brief Legacy single-shell pure-s (one s shell / atom).
    RHFFockDevice(int n_prim, const double* exponents,
                  const double* coefficients, int n_occ, int natm,
                  double level_shift = 0.0)
        : RHFFockDevice(/*n_shells_per_atom=*/1, n_prim, exponents,
                        coefficients, n_occ, natm, level_shift) {}

    /// @brief Legacy Wave-3 signature without natm — natm taken at build().
    RHFFockDevice(int n_shells_per_atom, int n_prim, const double* alpha_atom,
                  const double* coeff_atom, int n_occ,
                  double level_shift = 0.0);

    void build(const SolverContext& ctx, SCFWorkspace& ws);
    void step(const SolverContext& ctx, SCFWorkspace& ws);
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    int n_shells_ = 0;
    int n_ao_ = 0;
    int n_prim_ = 0;
    int n_occ_ = 0;
    double level_shift_ = 0.0;
    /// When true, rebuild molecular shell list from pure-s atom block at build.
    bool pure_s_broadcast_ = false;
    int n_shells_per_atom_ = 1;

    std::vector<double> alpha_mol_;
    std::vector<double> coeff_mol_;
    std::vector<int> L_mol_;
    std::vector<int> atom_of_mol_;

    void* cublas_handle_ = nullptr;
    void* cusolver_handle_ = nullptr;
    double* d_temp_ = nullptr;
    /// Overlap lives on `SCFWorkspace::S` (kept for CDIIS after Löwdin X).
    double* d_T_ = nullptr;
    double* d_V_ = nullptr;
};

#endif // ATV_HAS_HF

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
