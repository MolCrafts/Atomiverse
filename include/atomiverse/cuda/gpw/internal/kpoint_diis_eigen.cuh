/// @file cuda/gpw/internal/kpoint_diis_eigen.cuh
/// @brief Device multi-k Pulay DIIS + cuSOLVER complex Hermitian eigen.
///
/// CUDA twin of host `cpu::gpw::internal::KDiis` + `diagonalize_k` +
/// closed-shell density used inside `cuda::gpw::GpwDft` multi-k RKS mid-SCF
/// (`run_cuda_kpoints_rks`). One shared Pulay system
/// \(B_{ij}=\sum_k w_k \mathrm{Re}\,\mathrm{Tr}((e_i^k)^\dagger e_j^k)\) with
/// residual \(e^k=X^\dagger(HPS-SPH)X\); per-k advance does \(F'=X^\dagger H X\),
/// `cusolverDnZheevd`, \(C=X C'\), \(P=2\,C_{\mathrm{occ}} C_{\mathrm{occ}}^\dagger\).
///
/// Host matrices are **row-major** `std::complex<double>` matching
/// `cpu::linalg::CplxMatrix`; device work is **column-major** cuBLAS layout.
/// Only the tiny (depth+1) Pulay Lagrange system is solved on host
/// (depth ≤ 8). Domain-local to `cuda::gpw::internal`.
#pragma once

#include <complex>
#include <cuda_runtime.h>
#include <vector>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Multi-k device Pulay DIIS + complex eigen + density (closed-shell).
class KpointDiisEigenDevice {
  public:
    KpointDiisEigenDevice() = default;
    ~KpointDiisEigenDevice() { destroy(); }

    KpointDiisEigenDevice(const KpointDiisEigenDevice&) = delete;
    KpointDiisEigenDevice& operator=(const KpointDiisEigenDevice&) = delete;

    /// @brief Allocate multi-k device buffers and cuBLAS/cuSOLVER handles.
    ///
    /// @param stream      Non-default SCF stream (owned by caller).
    /// @param nk          Number of k-points (≥ 1).
    /// @param n           AO dimension (rows/cols of H, P, S).
    /// @param max_m       Max orthonormal MO count over k (m_k ≤ max_m ≤ n).
    /// @param max_vectors Pulay history depth (≥ 1; 0 disables DIIS).
    /// @param n_occ       Occupied orbitals for density.
    /// @param occ_factor  Density occupancy (2 closed-shell RKS).
    /// @param weights     Length-nk k weights (copied; used in B_ij).
    void build(cudaStream_t stream, int nk, int n, int max_m, int max_vectors,
               int n_occ, double occ_factor,
               const std::vector<double>& weights);

    /// @brief Upload geometry-fixed S (n×n) and X (n×m_k), **row-major host**.
    ///
    /// @param ik  k-index in [0, nk).
    /// @param S_host Row-major Hermitian overlap, length n*n.
    /// @param X_host Row-major Löwdin transform, length n*m_k.
    /// @param m_k Orthonormal column count at this k (1 ≤ m_k ≤ max_m).
    void upload_SX_k(int ik, const std::complex<double>* S_host,
                     const std::complex<double>* X_host, int m_k);

    /// @brief Upload / refresh density P_k (n×n, row-major host Hermitian).
    void upload_P_k(int ik, const std::complex<double>* P_host);

    /// @brief Download current density P_k (n×n, row-major host).
    void download_P_k(int ik, std::complex<double>* P_host) const;

    /// @brief Clear DIIS history (call once per multi-k `run`).
    void reset();

    /// @brief Upload all H_k (row-major host) and form DIIS residuals.
    ///
    /// @param H_hosts Length-nk pointers to n×n row-major KS matrices [Ha].
    /// @return max |e_ij^k| over all k and elements [Hartree].
    double residual(const std::vector<const std::complex<double>*>& H_hosts);

    /// @brief Ingest device-resident H_k (row-major, integrate layout) into
    ///        per-k DIIS column-major `d_H_[ik]` via device transpose (D2D).
    ///
    /// Used by product mid-SCF after `KpointIntegrateDevice::assemble_H` so
    /// residual does not D2H full \(H_k\) (cuda-gpw-kpoint-h-residency).
    ///
    /// @param ik       k-index in [0, nk).
    /// @param d_H_rm   Device pointer to n×n complex row-major H [Ha]
    ///                 (same layout as `KpointIntegrateDevice::d_H()`).
    void ingest_H_k_device_rm(int ik, const void* d_H_rm);

    /// @brief Form DIIS residuals from already-resident device H_k buffers.
    ///
    /// Call after `ingest_H_k_device_rm` for every k (or after host
    /// `residual` which also fills `d_H_`). No host H traffic.
    ///
    /// @return max |e_ij^k| over all k and elements [Hartree].
    double residual_from_device();

    /// @brief DIIS extrapolate + eigen + rebuild density for every k.
    ///
    /// Call after `residual` / `residual_from_device` when the SCF iteration
    /// is not yet converged.
    void advance();

    /// @brief residual() then advance() — convenience for tests.
    double step(const std::vector<const std::complex<double>*>& H_hosts);

    /// @brief Free device memory and handles.
    void destroy();

    bool built() const { return built_; }
    int nk() const { return nk_; }
    int n() const { return n_; }
    int m_k(int ik) const;

    /// @brief Device pointer to current P_k (n×n col-major complex).
    const void* d_P(int ik) const;

    // ── Test / oracle surface ────────────────────────────────────────────

    /// @brief Device e^k into host buffer (m_k×m_k **row-major**).
    void error_to_host(int ik, const std::complex<double>* H_host,
                       std::complex<double>* e_host_mm);

    /// @brief Device eigen of F' = X† H X (no DIIS) → host row-major C + ε.
    void eigen_to_host(int ik, const std::complex<double>* H_host,
                       double* eig_host, std::complex<double>* C_host);

  private:
    void ensure_built_() const;
    void ensure_ik_(int ik) const;
    void form_error_k_(int ik);
    void diis_extrapolate_();
    void diagonalize_and_density_k_(int ik);
    double err_norm_from_device_() const;

    void host_rm_to_dev_cm_(const std::complex<double>* rm, void* d_cm,
                            int rows, int cols) const;
    void dev_cm_to_host_rm_(const void* d_cm, std::complex<double>* rm,
                            int rows, int cols) const;

    cudaStream_t stream_ = nullptr;
    void* cublas_ = nullptr;
    void* cusolver_ = nullptr;

    int nk_ = 0;
    int n_ = 0;
    int max_m_ = 0;
    int max_vectors_ = 0;
    int n_occ_ = 0;
    double occ_factor_ = 2.0;
    bool use_diis_ = false;
    bool built_ = false;

    int hist_count_ = 0;
    int hist_start_ = 0;

    std::vector<double> weights_;
    std::vector<int> m_k_;

    // Per-k device buffers (void* = cuDoubleComplex* storage).
    std::vector<void*> d_S_;   ///< n×n
    std::vector<void*> d_X_;   ///< n×max_m
    std::vector<void*> d_P_;   ///< n×n
    std::vector<void*> d_H_;   ///< n×n work / extrapolated H
    std::vector<void*> d_C_;   ///< n×max_m AO MOs
    std::vector<void*> d_Cp_;  ///< max_m×max_m orthonormal eigenvectors
    std::vector<void*> d_err_; ///< max_m×max_m DIIS residual
    std::vector<double*> d_eig_;

    void* d_t1_ = nullptr; ///< scratch max(n×n, n×max_m, max_m×max_m)
    void* d_t2_ = nullptr;
    void* d_t3_ = nullptr;
    void* d_H_hist_ = nullptr; ///< max_vectors × nk × n×n
    void* d_e_hist_ = nullptr; ///< max_vectors × nk × max_m×max_m
    double* d_c_ = nullptr;
    void* d_sy_work_ = nullptr;
    int* d_info_ = nullptr;
    int sy_lwork_ = 0;

    // Host staging for layout convert (reused).
    mutable std::vector<std::complex<double>> stage_;
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
