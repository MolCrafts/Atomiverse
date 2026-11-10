/// @file cuda/gpw/internal/kpoint_basis.cuh
/// @brief Read-only device Bloch AO values for one GPW assembly.
#pragma once

#include <cuda_runtime.h>

namespace ATV::cpu::gpw::internal {
struct KpointBasisGrid;
}

namespace ATV::cuda::gpw::internal {

class KpointBasisDevice {
  public:
    KpointBasisDevice(cudaStream_t stream,
                      const cpu::gpw::internal::KpointBasisGrid& host);
    ~KpointBasisDevice();

    KpointBasisDevice(const KpointBasisDevice&) = delete;
    KpointBasisDevice& operator=(const KpointBasisDevice&) = delete;

    const double2* data() const { return values_; }
    cudaStream_t stream() const { return stream_; }
    int nk() const { return nk_; }
    int nb() const { return nb_; }
    long npts() const { return npts_; }

  private:
    cudaStream_t stream_;
    int nk_;
    int nb_;
    long npts_;
    double2* values_ = nullptr; ///< [k][grid point][AO], complex interleaved.
};

} // namespace ATV::cuda::gpw::internal
