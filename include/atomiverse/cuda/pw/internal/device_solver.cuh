/// @file cuda/pw/internal/device_solver.cuh
/// @brief Device-resident plane-wave band solver (block Davidson) and
///        density accumulator used by `cuda::pw::pw_scf`.
///
/// Owned by the CUDA PW SCF driver, never registered in any Driver Plan
/// (same internal-helper contract as `cuda::scf::internal::*`). All
/// per-band hot loops run on the GPU: H·ψ via cuFFT Z2Z + V_eff(r)
/// multiply + Kleinman-Bylander projector GEMVs, subspace algebra via
/// cuBLAS ZGEMM. Only the small dense subspace eigenproblem (Jacobi)
/// and the Davidson bookkeeping stay on the host, reusing the exact CPU
/// helpers from `cpu/pw/internal/pw_subspace.h` for numerical parity.
#pragma once

#include "atomiverse/cpu/pw/internal/g_vector.h"
#include "atomiverse/cpu/pw/internal/pw_workspace.h"
#include <complex>
#include <cublas_v2.h>
#include <cufft.h>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief GPU band solver for one plane-wave SCF run.
///
/// Lifecycle per SCF iteration (mirrors the CPU loop in pw_scf.cpp):
/// @code
///   solver.set_veff(ws.V_eff_r.data());          // once per iteration
///   solver.zero_density();
///   for each k-point:
///     // ws.gvec / ws.n_g / ws.T_G bound to the k-sphere by the caller
///     solver.set_kpoint(ws);
///     solver.solve(ws, tol, max_dav);            // ws.eigenvalues filled
///     solver.accumulate_density(weight, ws.omega);
///   solver.fetch_density_g(gv0, rho_g_out);      // ρ(G) on the Γ-sphere
/// @endcode
///
/// Internally all math is double precision (matching the CPU path, which
/// runs FFTW in double regardless of ATV_REAL).
class DeviceBandSolver {
  public:
    /// @param fft_nx,fft_ny,fft_nz Dense c2c FFT grid dims (ws.fft_n*).
    /// @param n_bands Number of bands to solve per k-point.
    DeviceBandSolver(int fft_nx, int fft_ny, int fft_nz, int n_bands);
    ~DeviceBandSolver();
    DeviceBandSolver(const DeviceBandSolver&) = delete;
    DeviceBandSolver& operator=(const DeviceBandSolver&) = delete;

    /// @brief Upload V_eff(r) (size nr) for the current SCF iteration.
    void set_veff(const Real* v_eff_r);

    /// @brief Bind the k-point currently selected in `ws` (gvec/n_g/T_G):
    /// uploads T(G), G-vectors, the G→FFT-grid index map and rebuilds the
    /// KB projector matrix β (ng × n_proj_columns) on device.
    void set_kpoint(const cpu::pw::PWWorkspace& ws);

    /// @brief Block-Davidson diagonalization for the bound k-point,
    /// algorithmically identical to `cpu::pw::PW_Davidson::launch`.
    /// Writes ws.eigenvalues; band wavefunctions stay device-resident.
    void solve(cpu::pw::PWWorkspace& ws, Real tol, int max_iter);

    /// @brief Reset the device real-space density accumulator.
    void zero_density();

    /// @brief rho_r += weight·2·|ψ_b(r)|²/Ω over the bands of the most
    /// recent solve() (occupations = 2, as on CPU).
    void accumulate_density(Real weight, Real omega);

    /// @brief Forward-FFT the accumulated density and gather ρ(G) on the
    /// Γ-centred sphere `gv0` (host output, gv0.count() entries).
    void fetch_density_g(const cpu::pw::internal::GVectorList& gv0,
                         std::complex<Real>* rho_g);

    /// @brief Test hook: H·ψ for one host vector on the bound k-point
    /// (H2D → apply_h → D2H). Used by the band-parity diagnostics to
    /// compute residual norms of candidate eigenpairs.
    void apply_h_host(const std::complex<Real>* psi, std::complex<Real>* hpsi);

  private:
    /// H·ψ for one device vector (cuFFT + V_eff + KB projectors).
    void apply_h(const cufftDoubleComplex* psi, cufftDoubleComplex* hpsi);
    void ensure_subspace_capacity(int ng, int cap);

    int nx_, ny_, nz_, nr_;
    int n_bands_;
    int ng_ = 0;       ///< G-sphere size of the bound k-point.
    int n_proj_ = 0;   ///< KB projector columns (Σ_atoms Σ_proj (2l+1)).
    int cap_sub_ = 0;  ///< Allocated subspace capacity (vectors).
    int ng_cap_ = 0;   ///< Allocated per-vector length (G-sphere size).
    double omega_ = 0; ///< Cell volume of the bound workspace [Bohr³].

    cufftHandle fft_plan_ = 0;
    cublasHandle_t blas_ = nullptr;

    // Dense-grid buffers
    cufftDoubleComplex* d_grid_ = nullptr; ///< nr c2c work grid.
    double* d_veff_ = nullptr;             ///< nr V_eff(r).
    double* d_rho_ = nullptr;              ///< nr density accumulator.

    // Per-k-point buffers (sized to ng_)
    int* d_idx_ = nullptr;                   ///< G → flat FFT-grid index.
    double* d_tg_ = nullptr;                 ///< T(G) kinetic diagonal.
    cufftDoubleComplex* d_beta_ = nullptr;   ///< β matrix (ng × n_proj_).
    double* d_betad_ = nullptr;              ///< per-column KB strength D.
    cufftDoubleComplex* d_proj_c_ = nullptr; ///< ⟨β|ψ⟩ workspace (n_proj_).

    // Davidson buffers
    cufftDoubleComplex* d_sub_ = nullptr;   ///< subspace V (ng × cap_sub).
    cufftDoubleComplex* d_hsub_ = nullptr;  ///< H·V (ng × cap_sub).
    cufftDoubleComplex* d_ritz_ = nullptr;  ///< Ritz vectors (ng × nb).
    cufftDoubleComplex* d_hritz_ = nullptr; ///< H·Ritz (ng × nb).
    cufftDoubleComplex* d_hs_ = nullptr;    ///< subspace H (cap × cap).
    cufftDoubleComplex* d_x_ = nullptr;     ///< Ritz rotation (cap × nb).
    cufftDoubleComplex* d_work_ = nullptr;  ///< ng scratch (R / δ).
    cufftDoubleComplex* d_ovlp_ = nullptr;  ///< cap scratch (⟨V|δ⟩).
};

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
