/// @file cuda/scf/internal/diis_mixer.cuh
/// @brief internal::DIISMixer — device CDIIS (Pulay) Fock mixer + level shift.
///
/// CUDA twin of CPU `MixStage` / `internal::DIISMixer` for **restricted** RHF.
/// Runs after the Fock build and before diagonalization:
///   1. DIIS error e = Xᵀ (F P S − S P F) X from the **unshifted** AO Fock
///   2. Fock-space Pulay extrapolation of F (history on device)
///   3. F′ = Xᵀ F X, then Saunders–Hillier level shift on F′ (EigenSolver input)
///
/// Energy path keeps the unshifted (DIIS-extrapolated) F in `ws.work`. No host
/// Fock materialization: history, error, and linear combination stay on device.
/// The tiny (m+1)×(m+1) Pulay system is solved on host (m ≤ max_vectors ≤ 8).
///
/// References: Pulay, Chem. Phys. Lett. 73, 393 (1980); J. Comp. Chem. 3, 556
/// (1982). Level shift: Saunders & Hillier, IJQC 7, 699 (1973).
#pragma once

#include "atomiverse/cuda/scf/internal/solver_context.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

/// @brief Device CDIIS mixer for closed-shell RHF (single spin channel).
class DIISMixer {
  public:
    /// @param max_vectors Maximum stored (F, e) pairs (CPU default 8).
    /// @param start_iter First iteration index at which history is pushed
    ///                   (CPU `diis_start_iter`, default 1).
    /// @param level_shift Virtual-orbital shift λ [Hartree]; |λ| ≤ 1e-4 no-op.
    explicit DIISMixer(int max_vectors = 8, int start_iter = 1,
                       double level_shift = 0.0);

    /// @brief Allocate device history + scratch sized from `ws.n`.
    void build(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief CDIIS extrapolate `ws.work` (AO F) + form level-shifted F′ in `ws.H`.
    ///
    /// Requires `ws.S`, `ws.X`, `ws.P`, and unshifted F in `ws.work` (as left by
    /// RHFFock / RHFFockDevice). Writes extrapolated F back to `ws.work` and
    /// the EigenSolver input into `ws.H` / `ws.F_prime`.
    void step(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Free device buffers and the cuBLAS handle.
    void destroy(const SolverContext& ctx, SCFWorkspace& ws);

    /// @brief Clear DIIS history (call once at the start of each SCF `run()`).
    void reset();

  private:
    int max_vectors_ = 8;
    int start_iter_ = 1;
    double level_shift_ = 0.0;
    int hist_count_ = 0;
    int hist_start_ = 0; ///< Circular index of the oldest vector.

    void* cublas_handle_ = nullptr;
    prec_t* d_F_hist_ = nullptr; ///< [max_vectors * n * n] AO Fock history.
    prec_t* d_e_hist_ = nullptr; ///< [max_vectors * n * n] DIIS error history.
    prec_t* d_err_ = nullptr;    ///< Current error matrix [n * n].
    prec_t* d_temp1_ = nullptr;  ///< Scratch [n * n].
    prec_t* d_temp2_ = nullptr;  ///< Scratch [n * n].
    prec_t* d_B_ = nullptr;      ///< Device B block [max_vectors * max_vectors].
    prec_t* d_c_ = nullptr;      ///< Extrapolation coefficients [max_vectors].
    int n_ = 0;
    bool built_ = false;
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
