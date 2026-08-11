/// @file cuda/gpw/internal/ao_diis_eigen.cuh
/// @brief Device Pulay DIIS + cuSOLVER dense eigen for GPW Γ RKS/UKS mid-SCF.
///
/// CUDA twin of host `cpu::gpw::internal::GpwDiis` + `diagonalize_fock` /
/// `diagonalize_fock_shifted` + `build_density_matrix` used inside
/// `cuda::gpw::GpwDft::run` after the device KS product stage builds H.
///
/// Work stays on device: commutator residual
/// `e = Xᵀ (H P S − S P H) X`, history of (H, e), extrapolated H, F′ = Xᵀ H X
/// (optional Saunders–Hillier level shift), `dsyevd`, C = X C′, and
/// P = factor · C_occ C_occᵀ. Only the tiny (depth+1) Pulay Lagrange system
/// is solved on host (depth ≤ 8). Matrices are column-major doubles matching
/// `cpu::linalg::RealMatrix`.
///
/// One instance is one spin channel: RKS uses a single device with
/// `occ_factor=2`; collinear UKS composes two (`n_alpha`/`n_beta`,
/// `occ_factor=1` each) — cuda-gpw-uks-device-diis.
///
/// Domain-local to `cuda::gpw::internal` — does not consume `cuda::scf`
/// helpers (GPW isolation). Public method surface stays `GpwDft`.
#pragma once

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Device AO DIIS + eigen + density for one spin channel (RKS or UKS).
class AoDiisEigenDevice {
  public:
    AoDiisEigenDevice() = default;
    ~AoDiisEigenDevice() { destroy(); }

    AoDiisEigenDevice(const AoDiisEigenDevice&) = delete;
    AoDiisEigenDevice& operator=(const AoDiisEigenDevice&) = delete;

    /// @brief Allocate device buffers and cuBLAS/cuSOLVER handles.
    ///
    /// @param stream     Non-default SCF stream (owned by caller).
    /// @param n          AO dimension (rows/cols of H, P, S).
    /// @param m          Orthonormal MO count (cols of X; m ≤ n).
    /// @param max_vectors Pulay history depth (≥ 1; 0 disables DIIS).
    /// @param n_occ      Occupied orbitals for density / level-shift projector.
    /// @param occ_factor Density occupancy (2 closed-shell RKS; 1 UKS channel).
    /// @param level_shift Virtual shift λ [Hartree]; ≤ 0 disables.
    void build(cudaStream_t stream, int n, int m, int max_vectors, int n_occ,
               double occ_factor, double level_shift);

    /// @brief Upload geometry-fixed S (n×n) and X (n×m), column-major.
    void upload_SX(const double* S_host, const double* X_host);

    /// @brief Upload / refresh density P (n×n, column-major).
    void upload_P(const double* P_host);

    /// @brief Upload previous MO coefficients C (n×m) for level-shift projector.
    ///
    /// Sets the have-C flag so Saunders–Hillier shift applies from the first
    /// advance (CPU twin seeds C_prev from the initial diagonalize).
    void upload_C(const double* C_host);

    /// @brief Download current density P (n×n, column-major).
    void download_P(double* P_host) const;

    /// @brief Download MO coefficients C (n×m, column-major). Valid after eigen.
    void download_C(double* C_host) const;

    /// @brief Download eigenvalues ε (length m). Valid after eigen.
    void download_eig(double* eig_host) const;

    /// @brief Clear DIIS history (call once per `GpwDft::run`).
    void reset();

    /// @brief Upload H from **host** and form DIIS residual (tests / oracle).
    ///
    /// @param H_host Column-major KS matrix [Hartree], length n*n.
    /// @return max |e_ij| of `e = Xᵀ (HPS−SPH) X` [Hartree].
    /// @note Production mid-SCF must use `residual_device` (no H2D of H).
    double residual(const double* H_host);

    /// @brief Form DIIS residual from a **device-resident** H (D2D into `d_H_`).
    ///
    /// CUDA-only iron law path: mid-SCF never H2Ds the KS matrix.
    ///
    /// @param d_H Device column-major H [Hartree], length n*n (may alias `d_H_`).
    /// @return max |e_ij| of the commutator residual [Hartree] (device max-reduce).
    double residual_device(const double* d_H);

    /// @brief DIIS extrapolate current `d_H_` + eigen + rebuild density.
    ///
    /// Call after `residual` when the SCF iteration is not yet converged.
    /// Pushes (H, e) into history when DIIS is enabled.
    void advance();

    /// @brief residual() then advance() — convenience for tests.
    double step(const double* H_host);

    /// @brief Free device memory and handles.
    void destroy();

    /// @return true after successful `build`.
    bool built() const { return built_; }

    int n() const { return n_; }
    int m() const { return m_; }

    /// @brief Device pointer to current P (n×n); valid after build/upload/step.
    const double* d_P() const { return d_P_; }
    /// @brief Device pointer to current C (n×m); valid after first step.
    const double* d_C() const { return d_C_; }
    /// @brief Device eigenvalues (m); valid after first step.
    const double* d_eig() const { return d_eig_; }

    // ── Test / oracle surface (host ↔ device single-shot) ───────────────

    /// @brief Device `e = Xᵀ (HPS−SPH) X` into host buffer (m×m col-major).
    /// Requires `build` + `upload_SX` + live `d_P_` (via `upload_P`).
    void error_to_host(const double* H_host, double* e_host_mm);

    /// @brief Device eigen of F′ = Xᵀ H X (no DIIS, no level shift) → host.
    ///
    /// @param H_host  n×n column-major.
    /// @param eig_host length m eigenvalues [Hartree] ascending.
    /// @param C_host  n×m column-major MO coefficients in AO basis.
    void eigen_to_host(const double* H_host, double* eig_host, double* C_host);

    /// @brief Device eigen of **device-resident** H (no DIIS, no level shift).
    ///
    /// CUDA-only payload path: final Γ RKS (`cuda-gpw-device-payload`) and
    /// dual-spin Γ UKS (`cuda-gpw-uks-payload-device`) C/ε from device
    /// `dsyevd` without host `diagonalize_fock` re-diag. D2D into `d_H_`,
    /// then boundary D2H of eig + C (and densified `d_P_`).
    ///
    /// @param d_H     Device column-major H [Hartree], length n*n.
    /// @param eig_host length m eigenvalues [Hartree] ascending (may be null).
    /// @param C_host  n×m column-major MOs (may be null).
    void eigen_from_device(const double* d_H, double* eig_host, double* C_host);

  private:
    void ensure_built_() const;
    void form_error_();                ///< d_err_ = Xᵀ(HPS−SPH)X using d_H_
    void diis_extrapolate_();          ///< d_H_ ← extrapolated H (or unchanged)
    void diagonalize_and_density_();   ///< d_H_ → d_C_, d_eig_, d_P_
    double err_norm_from_device_() const;

    cudaStream_t stream_ = nullptr;
    void* cublas_ = nullptr;
    void* cusolver_ = nullptr;

    int n_ = 0;
    int m_ = 0;
    int max_vectors_ = 0;
    int n_occ_ = 0;
    double occ_factor_ = 2.0;
    double level_shift_ = 0.0;
    bool use_diis_ = false;
    bool built_ = false;
    bool have_C_ = false;

    int hist_count_ = 0;
    int hist_start_ = 0;

    double* d_S_ = nullptr;      ///< n×n
    double* d_X_ = nullptr;      ///< n×m
    double* d_P_ = nullptr;      ///< n×n
    double* d_H_ = nullptr;      ///< n×n work / extrapolated H
    double* d_C_ = nullptr;      ///< n×m AO MOs
    double* d_Cp_ = nullptr;     ///< m×m orthonormal eigenvectors (syevd)
    double* d_eig_ = nullptr;    ///< m
    double* d_err_ = nullptr;    ///< m×m DIIS residual
    double* d_t1_ = nullptr;     ///< scratch max(n×n, n×m, m×m)
    double* d_t2_ = nullptr;     ///< scratch
    double* d_t3_ = nullptr;     ///< scratch n×n (PH copy / commutator aux)
    double* d_H_hist_ = nullptr; ///< max_vectors × n×n
    double* d_e_hist_ = nullptr; ///< max_vectors × m×m
    double* d_c_ = nullptr;      ///< max_vectors coefficients
    double* d_sy_work_ = nullptr;
    int* d_info_ = nullptr;
    int sy_lwork_ = 0;
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
