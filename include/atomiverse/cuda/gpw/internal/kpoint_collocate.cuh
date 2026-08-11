/// @file cuda/gpw/internal/kpoint_collocate.cuh
/// @brief Device multi-k density collocate from χ^k · P_k · χ^{k†}.
///
/// CUDA twin of host `cpu::gpw::internal::collocate_density_k`. Geometry-
/// fixed Bloch basis values χ_u^k(g) are uploaded once per `run`; each SCF
/// iteration only refreshes the small Hermitian P_k and accumulates
///   ρ(g) += w_k · Re[ Σ_{uv} P_uv χ_u(g) conj(χ_v(g)) ]
/// into a device density buffer (no host grid loop / mid-SCF H2D ρ).
///
/// Domain-local to `cuda::gpw::internal` — not a public method API.
#pragma once

#include <complex>
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Device workspace for multi-k χ^k density collocation.
///
/// Owns χ buffers (nk · npts · nb complex) and a P_k scratch (nb · nb).
/// Caller owns `d_rho` and zeros it before the first `collocate_k` of an
/// SCF iteration (or use `collocate_zero` to clear).
class KpointCollocateDevice {
  public:
    KpointCollocateDevice() = default;
    ~KpointCollocateDevice() { destroy(); }

    KpointCollocateDevice(const KpointCollocateDevice&) = delete;
    KpointCollocateDevice& operator=(const KpointCollocateDevice&) = delete;

    /// @brief Allocate device χ stack + P scratch.
    ///
    /// @param stream Non-default SCF stream (caller-owned; may be null).
    /// @param nk     Number of k-points.
    /// @param nb     AO basis size.
    /// @param npts   Real-space grid points.
    void build(cudaStream_t stream, int nk, int nb, long npts);

    /// @brief Upload one k's χ (length npts·nb, row-major g-major then u).
    ///
    /// Layout matches host `KpointBasisGrid::chi[ik][g * nb + u]`.
    ///
    /// @param ik        k index in [0, nk).
    /// @param chi_host  Host complex buffer, length npts·nb.
    void upload_chi(int ik, const std::complex<double>* chi_host);

    /// @brief Zero device density (async on the bound stream).
    void zero_rho(double* d_rho) const;

    /// @brief Accumulate w_k · ρ_k into `d_rho` from host row-major P_k.
    ///
    /// Scientific work is on-device: one thread per grid point, O(nb²)
    /// matvec matching host `collocate_density_k`. No host loop over g.
    ///
    /// @param ik     k index.
    /// @param P_host Host Hermitian density, row-major nb×nb
    ///               (`CplxMatrix` / xtensor default).
    /// @param wk     k-point quadrature weight.
    /// @param d_rho  Device density, length npts (accumulated).
    /// @note Prefer `collocate_k_device_cm` mid-SCF (no H2D of P).
    void collocate_k(int ik, const std::complex<double>* P_host, double wk,
                     double* d_rho);

    /// @brief Accumulate w_k · ρ_k from **device col-major** P (DIIS layout).
    ///
    /// Transposes CM→RM into internal scratch (D2D only), then same kernel
    /// as `collocate_k`. CUDA-only iron law path for multi-k mid-SCF.
    ///
    /// @param ik       k index.
    /// @param d_P_cm   Device Hermitian density, **column-major** nb×nb
    ///                 complex (matches `KpointDiisEigenDevice::d_P(ik)`).
    /// @param wk       k-point weight.
    /// @param d_rho    Device density, length npts (accumulated).
    void collocate_k_device_cm(int ik, const void* d_P_cm, double wk,
                               double* d_rho);

    /// @brief Free device memory.
    void destroy();

    bool built() const { return built_; }
    int nk() const { return nk_; }
    int nb() const { return nb_; }
    long npts() const { return npts_; }
    cudaStream_t stream() const { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
    int nk_ = 0;
    int nb_ = 0;
    long npts_ = 0;
    /// Packed χ: [ik][g * nb + u] as interleaved double2 / cuDoubleComplex.
    double2* d_chi_ = nullptr;
    double2* d_P_ = nullptr; ///< nb×nb row-major scratch.
    bool built_ = false;
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
