/// @file cuda/gpw/internal/kpoint_integrate.cuh
/// @brief Device multi-k V→H_k integrate: χ^k† V χ^k with resident V_eff.
///
/// CUDA twin of host `cpu::gpw::internal::integrate_potential_k`. Geometry-
/// fixed pre-summed Bloch basis χ^k lives on device for the run; each SCF
/// iteration integrates a device-resident V_eff into Hermitian V_μν^k without
/// mid-SCF D2H of the full grid. Domain-local to `cuda::gpw::internal`.
#pragma once

#include <complex>
#include <cuda_runtime.h>
#include <vector>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Device χ^k integrate workspace (one multi-k product run).
///
/// Layout of device χ (per k): dense row-major over grid points,
///   chi[g * nb + μ] = χ_μ^k(g)  (std::complex<double> / cuDoubleComplex)
/// Matches host `KpointBasisGrid::chi[ik]`.
class KpointIntegrateDevice {
  public:
    KpointIntegrateDevice() = default;
    ~KpointIntegrateDevice() { destroy(); }

    KpointIntegrateDevice(const KpointIntegrateDevice&) = delete;
    KpointIntegrateDevice& operator=(const KpointIntegrateDevice&) = delete;

    /// @brief Allocate per-k χ buffers + nb×nb Vmat workspace.
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

    /// @brief Integrate V_eff into Hermitian V_μν for one k on device.
    ///
    /// V(μ,ν) = Σ_g w · V(g) · conj(χ_μ) · χ_ν. Overwrites internal d_Vmat.
    /// All scientific work is on device; no host loop over grid points.
    ///
    /// @param d_V  Device potential length npts [Hartree].
    /// @param w    Grid integration weight V_cell/npts [Bohr³].
    /// @param ik   k index.
    void integrate(const double* d_V, double w, int ik);

    /// @brief Download last integrated Vmat (nb×nb column-major complex).
    ///
    /// @param out  Host buffer length nb·nb.
    void download_Vmat(std::complex<double>* out) const;

    /// @brief Device pointer to last Vmat (nb·nb complex), for advanced use.
    const void* d_Vmat() const { return d_Vmat_; }

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
    void* d_Vmat_ = nullptr; ///< nb·nb complex, overwritten each integrate.
    bool built_ = false;
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
