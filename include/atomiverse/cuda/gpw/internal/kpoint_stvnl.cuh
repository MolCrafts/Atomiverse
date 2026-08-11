/// @file cuda/gpw/internal/kpoint_stvnl.cuh
/// @brief Device multi-k phased analytic STVnl (\(S_k\), \(T_k\), \(V_{\mathrm{nl}}^k\)).
///
/// CUDA twin of host `build_overlap_kinetic_k` + `add_nonlocal_kb_k` setup
/// (once per geometry / multi-k `run`). Contracted McMurchie–Davidson / s–s
/// ST over lattice images with Bloch phases; Kleinman–Bylander nonlocal when
/// atoms expose channels (H GTH-PADE-q1 has none → device-zero Vnl). Domain-
/// local to `cuda::gpw::internal`.
#pragma once

#include <complex>
#include <cuda_runtime.h>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h" // GpwAtom, GridBasisFunction

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Device phased STVnl workspace (one multi-k geometry setup).
///
/// Matrices are nb×nb row-major complex per k:
///   out[ik * nb * nb + u * nb + t] = M_k(u,t)
/// Hermiticity: upper triangle computed, lower filled by conjugate.
class KpointStvnlDevice {
  public:
    KpointStvnlDevice() = default;
    ~KpointStvnlDevice() { destroy(); }

    KpointStvnlDevice(const KpointStvnlDevice&) = delete;
    KpointStvnlDevice& operator=(const KpointStvnlDevice&) = delete;

    /// @brief Free device buffers (idempotent).
    void destroy();

    /// @brief Build phased \(S_k\), \(T_k\), \(V_{\mathrm{nl}}^k\) on device.
    ///
    /// All scientific ST work is device-side (no host pair×image loop).
    /// Nonlocal KB: device-zero when every atom has empty `nonlocal`; otherwise
    /// host `add_nonlocal_kb_k` once then H2D (Si-class residual depth — dual-
    /// gate H₂ has zero channels).
    ///
    /// @param stream  Non-default SCF stream (caller-owned; may be null).
    /// @param basis   Contracted AO basis (geometry-fixed for the run).
    /// @param radii   Per-function eps-radii [Bohr].
    /// @param atoms   GTH centers (nonlocal channels).
    /// @param cell    Orthorhombic cell edges [Bohr].
    /// @param mimg    Per-axis image half-counts.
    /// @param screen  Pair Gaussian-product screen threshold.
    /// @param k_frac  nk×3 fractional κ (row-major).
    /// @param nk      Number of k-points.
    void compute(cudaStream_t stream,
                 const std::vector<cpu::gpw::GridBasisFunction>& basis,
                 const std::vector<double>& radii,
                 const std::vector<cpu::gpw::GpwAtom>& atoms,
                 const double cell[3], const int mimg[3], double screen,
                 const double* k_frac, int nk);

    /// @brief Download \(S_k\) (length nk·nb·nb complex, row-major per k).
    void download_S(std::complex<double>* out) const;
    /// @brief Download \(T_k\) [Hartree].
    void download_T(std::complex<double>* out) const;
    /// @brief Download \(V_{\mathrm{nl}}^k\) [Hartree].
    void download_Vnl(std::complex<double>* out) const;
    /// @brief Download \(H0_k = T_k + V_{\mathrm{nl}}^k\).
    void download_H0(std::complex<double>* out) const;

    /// @brief Device pointer to \(S_k\) pack (nk·nb·nb complex).
    const void* d_S() const { return d_S_; }
    /// @brief Device pointer to \(T_k\) pack.
    const void* d_T() const { return d_T_; }
    /// @brief Device pointer to \(V_{\mathrm{nl}}^k\) pack.
    const void* d_Vnl() const { return d_Vnl_; }
    /// @brief Device pointer to \(H0_k = T+Vnl\) pack.
    const void* d_H0() const { return d_H0_; }

    int nb() const { return nb_; }
    int nk() const { return nk_; }
    bool built() const { return built_; }

  private:
    void ensure_built_() const;

    cudaStream_t stream_ = nullptr;
    int nb_ = 0;
    int nk_ = 0;
    void* d_S_ = nullptr;
    void* d_T_ = nullptr;
    void* d_Vnl_ = nullptr;
    void* d_H0_ = nullptr;
    bool built_ = false;
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
