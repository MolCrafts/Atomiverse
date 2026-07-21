/// @file cuda/scf/internal/rhf_fock.cuh
/// @brief internal::RHFFock -- build Fock matrix F = Hcore + G(P),
///        transform F' = X^T F X.
///
/// **Internal SCF helper.** Does NOT inherit `cuda::Component` and is
/// never registered into a `Driver` Plan. Owned by `cuda::scf::SCFSolver`,
/// which drives its `build` / `step` / `destroy` lifecycle directly.
///
/// In `build()`, uploads Hcore, ERI, and X = S^{-1/2} to device.
/// In `step()`, builds two-electron matrix G(P) from ERI and current
/// density, forms Fock matrix F, and transforms to orthogonal basis F'.
///
/// @see cuda::scf::SCFSolver for the inner solver that owns this helper
/// @see internal::RHFDensity for back-transform and density rebuild
#pragma once

#include <vector>

#include "atomiverse/cuda/scf/internal/solver_context.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

#ifdef ATV_HAS_HF
#include "atomiverse/cpu/scf/internal/integrals.h"
#endif

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

#ifdef ATV_HAS_HF

/// @brief Build Fock matrix and transform to orthogonal basis (CPU integral path).
///
/// build(): compute X = S^{-1/2} on host, upload Hcore/ERI/X to device.
/// step():  G(P) from ERI, F = Hcore + G, F' = X^T F X.
class RHFFock {
  public:
    /// @brief Construct with precomputed CPU integrals.
    /// @param integrals Molecular integral pack (Hcore, ERI, S, V_nuc).
    /// @param level_shift Virtual-orbital level shift λ [Hartree] applied in
    ///        the orthonormal basis (F' += λ(I − P'_occ)); default 0.0 is a
    ///        strict no-op. Mirrors the CPU SCFConfig::level_shift
    ///        (hf-robust-03) for the CUDA RHF path.
    explicit RHFFock(const cpu::scf::IntegralPack& integrals,
                     double level_shift = 0.0)
        : integrals_(integrals), level_shift_(level_shift) {}

    /// @brief Upload Hcore/ERI/X to device, allocate workspace.
    /// @param ctx Build context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void build(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Build G(P), form F = Hcore + G, transform F' = X^T F X.
    /// @param ctx Launch context (CUDA stream + device frame + step index).
    /// @param ws SCF electronic-structure workspace.
    void step(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Free cuBLAS handle and scratch buffers.
    /// @param ctx Destroy context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    cpu::scf::IntegralPack integrals_; ///< Precomputed molecular integrals.
    double level_shift_ = 0.0;         ///< Level shift λ [Hartree].
    void* cublas_handle_ = nullptr;    ///< cuBLAS handle.
    double* d_temp_ = nullptr;         ///< Scratch buffer [device, n*n].
};

/// @brief Build Fock matrix — CUDA-only path (GPU integral kernels).
///
/// build(): upload basis params to device, launch GPU integral kernels to
///          compute S, T, V, ERI directly on device, then Hcore = T + V
///          and X = S^{-1/2} via cuSOLVER — zero CPU integral computation.
/// step():  same as RHFFock (G(P) from ERI, F = Hcore + G, F' = X^T F X).
class RHFFockDevice {
  public:
    /// @brief Construct for CUDA-only integral computation.
    /// @param n_prim Primitives per shell (3 for STO-3G).
    /// @param exponents Gaussian exponents [n_prim] [Bohr^-2].
    /// @param coefficients Raw contraction coefficients [n_prim]
    ///        (NOT normalized).
    /// @param n_occ Number of doubly-occupied orbitals [dimensionless].
    RHFFockDevice(int n_prim, const double* exponents,
                  const double* coefficients, int n_occ);

    /// @brief Upload basis, launch GPU integral kernels, build Hcore + X.
    /// @param ctx Build context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void build(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Build G(P), form F = Hcore + G, transform F' = X^T F X.
    /// @param ctx Launch context (CUDA stream + device frame + step index).
    /// @param ws SCF electronic-structure workspace.
    void step(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Free cuBLAS/cuSOLVER handles and scratch buffers.
    /// @param ctx Destroy context (CUDA stream + device frame).
    /// @param ws SCF electronic-structure workspace.
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

  private:
    int n_prim_;                       ///< Primitives per shell.
    int n_occ_;                        ///< Doubly-occupied orbital count.
    std::vector<double> exponents_;    ///< Gaussian exponents [Bohr^-2].
    std::vector<double> coefficients_; ///< Raw contraction coefficients.
    void* cublas_handle_ = nullptr;    ///< cuBLAS handle.
    void* cusolver_handle_ = nullptr;  ///< cuSOLVER handle.
    double* d_temp_ = nullptr;         ///< Scratch buffer [device, n*n].
    double* d_S_ = nullptr;            ///< Overlap (device, for X computation).
    double* d_T_ = nullptr;            ///< Kinetic (device, temp).
    double* d_V_ = nullptr;            ///< Nuclear (device, temp).
};

#endif // ATV_HAS_HF

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
