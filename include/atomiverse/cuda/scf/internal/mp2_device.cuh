/// @file cuda/scf/internal/mp2_device.cuh
/// @brief Private occupied-tiled cuBLAS AO-to-MO transform and CUB MP2 reduction.
#pragma once

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "atomiverse/cpu/scf/mp2.h"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV::cuda::scf::internal {

/// @brief Owned correlation scratch, allocated before the reference iteration.
class MP2Device {
  public:
    MP2Device() = default;
    ~MP2Device();
    MP2Device(const MP2Device&) = delete;
    MP2Device& operator=(const MP2Device&) = delete;

    /// Allocate all scratch and initialize scalar state during setup.
    void build(const SCFWorkspace& ws, int frozen, double os_scale,
               double ss_scale, std::size_t budget, cuda::Frame& frame,
               bool accumulate, cudaStream_t stream);
    /// Four cuBLAS contractions per occupied tile; fused CUB term reduction.
    void evaluate(const SCFWorkspace& ws, cuda::Frame& frame,
                  cudaStream_t stream);
    /// Check only mapped device status; retrieve no physical results.
    void validate(cudaStream_t stream) const;
    /// Final physical-result readback; throws on invalid denominators/results.
    cpu::scf::MP2Result result(cudaStream_t stream, double& reference_energy);

  private:
    void clear_() noexcept;
    cublasHandle_t blas_ = nullptr;
    double* first_ = nullptr;
    double* second_ = nullptr;
    void* reduction_scratch_ = nullptr;
    void* channels_ = nullptr;
    void* packet_ = nullptr;
    void* batch_pointers_ = nullptr;
    unsigned* h_status_ = nullptr;
    unsigned* d_status_ = nullptr;
    std::size_t reduction_bytes_ = 0;
    int n_ = 0;
    int n_occ_ = 0;
    int frozen_ = 0;
    double os_scale_ = 1.0;
    double ss_scale_ = 1.0;
};

} // namespace ATV::cuda::scf::internal
