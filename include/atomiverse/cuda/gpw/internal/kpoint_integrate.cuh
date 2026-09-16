/// @file cuda/gpw/internal/kpoint_integrate.cuh
/// @brief Device multi-k V→H_k integrate + H_k assemble with resident H0.
///
/// CUDA twin of host `cpu::gpw::internal::integrate_potential_k` plus mid-SCF
/// `H_k = T_k + Vnl_k + V_μν^k` fuse. Geometry-fixed pre-summed Bloch basis
/// χ^k and H0_k = T_k + Vnl_k live on device for the run; each SCF iteration
/// integrates a device-resident V_eff into Hermitian V_μν^k and assembles H_k
/// without mid-SCF host matrix add after Vmat D2H. Domain-local to
/// `cuda::gpw::internal`.
#pragma once

#include <complex>
#include <cuda_runtime.h>
#include <vector>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Device χ^k integrate + H_k assemble workspace (one multi-k run).
///
/// Layout of device χ (per k): dense row-major over grid points,
///   chi[g * nb + μ] = χ_μ^k(g)  (std::complex<double> / cuDoubleComplex)
/// Matches host `KpointBasisGrid::chi[ik]`.
///
/// H0 / H / Vmat are nb×nb row-major complex: out[u*nb+t] = M(u,t).
class KpointIntegrateDevice {
  public:
    KpointIntegrateDevice() = default;
    ~KpointIntegrateDevice() { destroy(); }

    KpointIntegrateDevice(const KpointIntegrateDevice&) = delete;
    KpointIntegrateDevice& operator=(const KpointIntegrateDevice&) = delete;

    /// @brief Allocate per-k χ + H0 buffers and nb×nb Vmat / H workspace.
    ///
    /// @param stream  Non-default SCF stream (caller-owned).
    /// @param nb      AO dimension.
    /// @param npts    Total real-space grid points.
    /// @param nk      Number of k-points.
    void build(cudaStream_t stream, int nb, long npts, int nk);

    /// @brief Free device buffers (idempotent).
    void destroy();

    /// @brief Upload pre-summed χ^k for one k (once per geometry / run).
    ///
    /// @param ik   k index in [0, nk).
    /// @param chi  Host array length npts·nb, layout g*nb+μ.
    void upload_chi(int ik, const std::complex<double>* chi);

    /// @brief Upload H0_k = T_k + Vnl_k once per k (geometry-fixed for the run).
    ///
    /// @param ik  k index in [0, nk).
    /// @param H0  Host Hermitian nb×nb row-major complex [Hartree].
    void upload_H0(int ik, const std::complex<double>* H0);

    /// @brief Integrate V_eff into Hermitian V_μν for one k on device.
    ///
    /// V(μ,ν) = Σ_g w · V(g) · conj(χ_μ) · χ_ν. Overwrites internal d_Vmat.
    /// All scientific work is on device; no host loop over grid points.
    ///
    /// @param d_V  Device potential length npts [Hartree].
    /// @param w    Grid integration weight V_cell/npts [Bohr³].
    /// @param ik   k index.
    void integrate(const double* d_V, double w, int ik);

    /// @brief Fuse H_k = H0_k + Vmat on device (after integrate for that k).
    ///
    /// Overwrites internal d_H. Requires prior `upload_H0(ik, …)` and
    /// `integrate(…, ik)` for the same ik. Device-only elementwise add.
    ///
    /// @param ik  k index matching the last integrate.
    void assemble_H(int ik);

    /// @brief Download last integrated Vmat (nb×nb row-major: out[u*nb+t]=V(u,t)).
    ///
    /// @param out  Host buffer length nb·nb.
    void download_Vmat(std::complex<double>* out) const;

    /// @brief Download last assembled H_k (nb×nb row-major).
    ///
    /// @param out  Host buffer length nb·nb.
    void download_H(std::complex<double>* out) const;

    /// @brief Device pointer to last Vmat (nb·nb complex), for advanced use.
    const void* d_Vmat() const { return d_Vmat_; }

    /// @brief Device pointer to last assembled H (nb·nb complex).
    const void* d_H() const { return d_H_; }

    int nb() const { return nb_; }
    long npts() const { return npts_; }
    int nk() const { return nk_; }
    bool built() const { return built_; }

  private:
    void ensure_built_() const;

    cudaStream_t stream_ = nullptr;
    int nb_ = 0;
    long npts_ = 0;
    int nk_ = 0;
    /// d_chi_[ik] length npts·nb complex.
    std::vector<void*> d_chi_;
    /// d_H0_[ik] length nb·nb complex (T_k + Vnl_k, once per run).
    std::vector<void*> d_H0_;
    void* d_Vmat_ = nullptr; ///< nb·nb complex, overwritten each integrate.
    void* d_H_ = nullptr;    ///< nb·nb complex, overwritten each assemble_H.
    bool built_ = false;
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
