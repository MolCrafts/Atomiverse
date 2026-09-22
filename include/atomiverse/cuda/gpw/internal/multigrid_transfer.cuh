/// @file cuda/gpw/internal/multigrid_transfer.cuh
/// @brief Device spectral prolong / restrict between GPW multigrid levels.
///
/// Mirrors CPU `prolong_fft` / `restrict_fft` (c2c band-limited embedding with
/// Nyquist half/half split; prolong scales by N_fine/N_coarse; restrict is the
/// exact adjoint w.r.t. grid inner products, inverse-normalized by 1/N_fine).
/// Orthorhombic + periodic only. Stage foothold for multi-grid product
/// collocate/integrate — not a full GPW method product.
#pragma once

#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

#include <cuda_runtime.h>
#include <cufft.h>

#include <vector>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Multigrid level list (finest first), matching CPU `MultiGrid::levels`.
///
/// All levels share the same orthorhombic box `L` and origin; coarser levels
/// have fewer points. Level 0 is the reference grid for Poisson + XC.
struct MultigridDesc {
    std::vector<CollocateGridDesc> levels; ///< finest first.

    int nlevels() const { return static_cast<int>(levels.size()); }
    const CollocateGridDesc& fine() const { return levels.at(0); }
    const CollocateGridDesc& level(int l) const {
        return levels.at(static_cast<std::size_t>(l));
    }
};

/// @brief Device spectral transfer between one coarse and one fine grid.
///
/// Owns c2c cuFFT plans + complex scratch for both shapes. Caller owns real
/// field buffers. All scientific work stays on device after construction.
class MultigridTransferDevice {
    int nc0_ = 0, nc1_ = 0, nc2_ = 0;
    int nf0_ = 0, nf1_ = 0, nf2_ = 0;
    long ntot_c_ = 0, ntot_f_ = 0;
    double scale_prolong_ = 1.0; ///< N_fine / N_coarse
    double inv_nf_ = 1.0;        ///< 1 / N_fine (prolong inv + restrict inv)

    cufftHandle fwd_c_ = 0;
    cufftHandle bwd_c_ = 0;
    cufftHandle fwd_f_ = 0;
    cufftHandle bwd_f_ = 0;
    cudaStream_t stream_ = nullptr;

    cufftDoubleComplex* d_c_c_ = nullptr; ///< coarse complex buffer
    cufftDoubleComplex* d_c_f_ = nullptr; ///< fine complex buffer

  public:
    /// @brief Build c2c plans + scratch for a coarse→fine pair (same box L).
    /// @param coarse Coarse collocate grid (even n, n_coarse ≤ n_fine per axis).
    /// @param fine   Fine collocate grid (even n).
    /// @param stream CUDA stream (default stream if null).
    MultigridTransferDevice(const CollocateGridDesc& coarse,
                            const CollocateGridDesc& fine,
                            cudaStream_t stream = nullptr);
    ~MultigridTransferDevice();

    MultigridTransferDevice(const MultigridTransferDevice&) = delete;
    MultigridTransferDevice& operator=(const MultigridTransferDevice&) = delete;

    /// @brief Spectral prolong: coarse real field → fine real field.
    /// @param d_coarse Device coarse field, length coarse.npts().
    /// @param d_fine   Device fine field out, length fine.npts() (overwritten).
    void prolong(const double* d_coarse, double* d_fine);

    /// @brief Spectral restrict: fine real field → coarse real field (adjoint).
    /// @param d_fine   Device fine field, length fine.npts().
    /// @param d_coarse Device coarse field out, length coarse.npts() (overwritten).
    void restrict(const double* d_fine, double* d_coarse);

    long ntot_coarse() const { return ntot_c_; }
    long ntot_fine() const { return ntot_f_; }
    cudaStream_t stream() const { return stream_; }
};

/// @brief Add two device fields: y[i] += x[i] for i in [0, n).
void device_axpy_add(const double* d_x, double* d_y, long n,
                     cudaStream_t stream = nullptr);

/// @brief Copy device field x → y (length n).
void device_copy_field(const double* d_x, double* d_y, long n,
                       cudaStream_t stream = nullptr);

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
