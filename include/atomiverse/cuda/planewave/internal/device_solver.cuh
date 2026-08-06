/// @file cuda/planewave/internal/device_solver.cuh
/// @brief Device-resident plane-wave band solver (block Davidson) and
///        density accumulator used by `cuda::pw::PW`.
///
/// Owned by the CUDA PW method object, never registered in any Driver Plan
/// (same internal-helper contract as `cuda::scf::internal::*`). All
/// per-band hot loops run on the GPU: H·ψ via cuFFT Z2Z + V_eff(r)
/// multiply + Kleinman-Bylander projector GEMVs, subspace algebra via
/// cuBLAS ZGEMM. Only the small dense subspace eigenproblem (complex
/// Hermitian Jacobi) and the Davidson bookkeeping stay on the host,
/// reusing the exact CPU helpers from `cpu/planewave/internal/pw_subspace.h`
/// for numerical parity.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
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
/// The real-space V_eff(r) it reads and the density accumulator it writes are
/// **owned by `internal::DeviceDensity`** and borrowed through `bind_grids`,
/// so neither crosses PCIe during the SCF. Lifecycle:
/// @code
///   solver.bind_grids(den.density_accumulator(),      // once per run
///                     den.effective_potential());
///   for each SCF iteration:
///     den.hartree(); den.exchange_correlation(); den.assemble_veff();
///     den.zero_accumulator();
///     for each k-point:
///       // ws.gvec / ws.n_g / ws.T_G bound to the k-sphere by the caller
///       solver.set_kpoint(ws);
///       solver.solve(ws, max_dav);                 // ws.eigenvalues filled
///       solver.accumulate_density(weight, ws.omega);
///     den.collect(); den.mix(iter); den.reduce(vol);
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

    /// @brief Bind the device grids owned by `internal::DeviceDensity`.
    ///
    /// The solver reads V_eff(r) and writes the density accumulator in place;
    /// it owns neither buffer and never copies them across PCIe. Must be
    /// called once per SCF run, before the first `solve()`.
    ///
    /// @param d_rho_accum Real-space density accumulator, nr doubles (device).
    /// @param d_veff_r    Real-space effective potential, nr doubles (device)
    ///                    [Hartree].
    void bind_grids(double* d_rho_accum, const double* d_veff_r);

    /// @brief Bind the k-point currently selected in `ws` (gvec/n_g/T_G):
    /// uploads T(G), G-vectors, the G→FFT-grid index map and rebuilds the
    /// KB projector matrix β (ng × n_proj_columns) on device.
    void set_kpoint(const cpu::pw::internal::PWWorkspace& ws);

    /// @brief Block-Davidson diagonalization for the bound k-point,
    /// algorithmically identical to `cpu::pw::internal::DavidsonSolver::step`,
    /// including its fixed |R|² residual threshold. Writes ws.eigenvalues;
    /// band wavefunctions stay device-resident.
    /// @param ws       Workspace with the k-sphere bound (gvec/n_g/T_G).
    /// @param max_iter Maximum Davidson expansion rounds (capped at 20).
    void solve(cpu::pw::internal::PWWorkspace& ws, int max_iter);

    /// @brief rho_accum += weight·2·|ψ_b(r)|²/Ω over the bands of the most
    /// recent solve() (occupations = 2, as on CPU). Writes the borrowed
    /// accumulator; the buffer is zeroed by
    /// `DeviceDensity::zero_accumulator()`.
    void accumulate_density(AccReal weight, AccReal omega);

    /// @brief Test hook: H·ψ for one host vector on the bound k-point
    /// (H2D → apply_h → D2H). Used by the band-parity diagnostics to
    /// compute residual norms of candidate eigenpairs.
    void apply_h_host(const std::complex<AccReal>* psi,
                      std::complex<AccReal>* hpsi);

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
    cufftDoubleComplex* d_grid_ = nullptr; ///< nr c2c work grid (owned).
    // Borrowed from internal::DeviceDensity — never allocated or freed here.
    double* rho_accum_ = nullptr;     ///< nr density accumulator (device).
    const double* veff_r_ = nullptr;  ///< nr V_eff(r) [Hartree] (device).

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
