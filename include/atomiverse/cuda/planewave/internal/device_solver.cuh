/// @file cuda/planewave/internal/device_solver.cuh
/// @brief Device-resident plane-wave band solver (block Davidson) and
///        density accumulator used by `cuda::pw::PW`.
///
/// Owned by the CUDA PW method object, never registered in any Driver Plan
/// (same internal-helper contract as `cuda::scf::internal::*`). All
/// per-band hot loops run on the GPU: H·ψ via cuFFT Z2Z + V_eff(r)
/// multiply + Kleinman-Bylander projector GEMVs, subspace algebra via
/// cuBLAS ZGEMM, and the dense subspace eigenproblem via cuSOLVER
/// `Zheevd` in place. What still reaches the host per Davidson round is
/// the `devInfo` flag, the n_bands lowest eigenvalues (kernel scalar
/// arguments) and the host-pointer `cublasDznrm2` residual norms.
///
/// **Not bit-identical to the CPU solver, by construction.** The CPU uses a
/// complex Hermitian Jacobi rotation sweep; `Zheevd` is divide-and-conquer.
/// Eigenvalues agree to roundoff, but eigenvectors differ by a phase and,
/// inside a degenerate multiplet, by an arbitrary unitary rotation. The
/// density is invariant under both **provided a degenerate multiplet is not
/// split across the n_bands cut** — with fixed closed-shell occupations and a
/// gapped system it is not. Individual residual vectors do differ, so the
/// Davidson trajectory can diverge from the CPU one; the converged
/// eigenvalues are what is compared, not the path.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include <complex>
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {
struct PWProblem; ///< Defined in cpu/planewave/internal/pw_setup.h.
} // namespace internal
} // namespace pw
} // namespace cpu

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
///   solver.build_kpoints(prob);                       // once per run
///   for each SCF iteration:
///     den.hartree(); den.exchange_correlation(); den.assemble_veff();
///     den.zero_accumulator();
///     for each k-point ik:
///       solver.bind_kpoint(ik);   // pointer swap + one β launch
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

    /// @brief Build every per-k-point table on device, once per run.
    ///
    /// Everything this uploads is a pure function of the geometry — the
    /// G→FFT index map, T(G), q = k+G, the KB projector descriptors and
    /// radial f_G, and the initial Davidson subspace (`init_subspace`, which
    /// reads only T_G / atom_wfc / gvec / atom_pos_bohr). None of it depends
    /// on V_eff or ρ, so rebuilding it per SCF iteration — as the loop did
    /// before — was repeating identical work `max_iter` times.
    ///
    /// Device cost is \f$\sum_k\f$ over the k-spheres; `bind_kpoint` then
    /// allocates nothing and copies nothing across PCIe. The total is checked
    /// against free device memory and throws a message naming the figure
    /// rather than surfacing a bare cudaMalloc failure.
    ///
    /// @param prob Problem assembled by `setup_pw_problem`. `prob.ws` is
    ///        temporarily rebound to each k-sphere and left bound to the
    ///        density sphere (`prob.gv0`) on return.
    void build_kpoints(cpu::pw::internal::PWProblem& prob);

    /// @brief Select a k-point built by `build_kpoints`.
    ///
    /// Moves pointers into the packed tables and relaunches the β build
    /// kernel from already-resident inputs. No allocation, no host↔device
    /// copy, no synchronization. β itself is recomputed rather than cached
    /// because it is the one \f$O(n_G n_\mathrm{proj})\f$ array here, and
    /// caching it over all k would dominate device memory on any real cell.
    ///
    /// @param ik k-point index into `prob.gv_k` (0-based).
    void bind_kpoint(int ik);

    /// @brief Block-Davidson diagonalization for the bound k-point,
    /// algorithmically identical to `cpu::pw::internal::DavidsonSolver::step`,
    /// including its fixed |R|² residual threshold. Writes ws.eigenvalues;
    /// band wavefunctions stay device-resident.
    ///
    /// The starting subspace is copied device→device from the table
    /// `build_kpoints` produced, so no O(n_G) array crosses PCIe here.
    ///
    /// @param ws       Workspace, used only to receive `eigenvalues`; the
    ///                 k-sphere comes from `bind_kpoint`, not from `ws`.
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
    /// Release the packed per-k tables and null the views into them.
    void free_tables();

    int nx_, ny_, nz_, nr_;
    int n_bands_;
    int ng_ = 0;       ///< G-sphere size of the bound k-point.
    int n_proj_ = 0;   ///< KB projector columns (Σ_atoms Σ_proj (2l+1)).
    int n_sub0_ = 0;   ///< Initial subspace vectors of the bound k-point.
    int bound_k_ = -1; ///< k-point index bound by bind_kpoint(), −1 = none.
    int cap_sub_ = 0;  ///< Allocated subspace capacity (vectors).
    int ng_cap_ = 0;   ///< Allocated per-vector length (G-sphere size).

    cufftHandle fft_plan_ = 0;
    cublasHandle_t blas_ = nullptr;
    /// Second handle pinned to CUBLAS_POINTER_MODE_DEVICE for the residual
    /// norms and their reciprocal scales. A separate handle rather than
    /// toggling `blas_`: the mode also governs GEMM's alpha/beta, and a
    /// throw between set and reset would leave the shared handle wrong.
    cublasHandle_t blas_dev_ = nullptr;
    cusolverDnHandle_t solver_ = nullptr;

    // Dense-grid buffers
    cufftDoubleComplex* d_grid_ = nullptr; ///< nr c2c work grid (owned).
    // Borrowed from internal::DeviceDensity — never allocated or freed here.
    double* rho_accum_ = nullptr;     ///< nr density accumulator (device).
    const double* veff_r_ = nullptr;  ///< nr V_eff(r) [Hartree] (device).

    // ── Views into the packed tables below, moved by bind_kpoint(). ──
    // NOT owned: never pass these to device_free().
    const int* d_idx_ = nullptr;    ///< G → flat FFT-grid index (bound k).
    const double* d_tg_ = nullptr;  ///< T(G) kinetic diagonal (bound k).
    const double* d_betad_ = nullptr; ///< per-column KB strength D (bound k).
    const cufftDoubleComplex* d_sub0_ = nullptr; ///< initial subspace (bound k).

    // Owned, sized to the largest k-sphere; contents rebuilt per bind_kpoint.
    cufftDoubleComplex* d_beta_ = nullptr;   ///< β matrix (ng × n_proj_).
    cufftDoubleComplex* d_proj_c_ = nullptr; ///< ⟨β|ψ⟩ workspace (n_proj_).

    // ── Packed per-k tables (owned), built once by build_kpoints(). ──
    // Indexed via the offset vectors; k_goff_ addresses per-(k,G) arrays,
    // k_coff_ per-(k,β-column), k_fgoff_ the concatenated radial f_G, and
    // k_soff_ the concatenated initial subspaces.
    int n_kpts_ = 0;
    double pref_ = 0; ///< 4π/√Ω, constant over k.
    std::vector<int> k_ng_, k_ncol_, k_nsub_;
    std::vector<std::size_t> k_goff_, k_coff_, k_fgoff_, k_soff_;
    int* d_idx_all_ = nullptr;
    double* d_tg_all_ = nullptr;
    double* d_qx_all_ = nullptr; ///< q = k+G, x component [Bohr⁻¹].
    double* d_qy_all_ = nullptr;
    double* d_qz_all_ = nullptr;
    int* d_cl_all_ = nullptr;  ///< β column: l.
    int* d_cm_all_ = nullptr;  ///< β column: m.
    int* d_co_all_ = nullptr;  ///< β column: offset into this k's f_G block.
    int* d_cn_all_ = nullptr;  ///< β column: length of its f_G run.
    double* d_cd_all_ = nullptr; ///< β column: KB strength D.
    double* d_cp_all_ = nullptr; ///< β column: atom position (3 per column).
    double* d_fg_all_ = nullptr; ///< concatenated radial f_G.
    cufftDoubleComplex* d_sub0_all_ = nullptr; ///< concatenated subspaces.

    // Davidson buffers
    cufftDoubleComplex* d_sub_ = nullptr;   ///< subspace V (ng × cap_sub).
    cufftDoubleComplex* d_hsub_ = nullptr;  ///< H·V (ng × cap_sub).
    cufftDoubleComplex* d_ritz_ = nullptr;  ///< Ritz vectors (ng × nb).
    cufftDoubleComplex* d_hritz_ = nullptr; ///< H·Ritz (ng × nb).
    /// Subspace H (cap × cap, column-major). `cusolverDnZheevd` overwrites it
    /// in place with the eigenvectors, whose first n_bands columns ARE the
    /// Ritz rotation — so no separate rotation buffer exists.
    cufftDoubleComplex* d_hs_ = nullptr;
    double* d_eig_ = nullptr;               ///< subspace eigenvalues (cap).
    cufftDoubleComplex* d_evd_work_ = nullptr; ///< Zheevd workspace.
    int* d_evd_info_ = nullptr;             ///< Zheevd convergence flag.
    int evd_lwork_ = 0;                     ///< Allocated Zheevd workspace.
    cufftDoubleComplex* d_work_ = nullptr;  ///< ng scratch (R / δ).
    // Residual-expansion scalars, device-resident for a whole band step.
    double* d_nrm_ = nullptr;  ///< [3]: ‖R‖, ‖precond‖, ‖δ‖.
    double* d_inv_ = nullptr;  ///< [2]: guarded 1/‖precond‖, 1/‖δ‖.
    int* d_accept_ = nullptr;  ///< 1 = append this band's δ to the subspace.
    cufftDoubleComplex* d_ovlp_ = nullptr;  ///< cap scratch (⟨V|δ⟩).
};

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
