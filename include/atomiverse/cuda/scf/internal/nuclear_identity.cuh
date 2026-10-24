#pragma once

#include <cub/device/device_reduce.cuh>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "atomiverse/cuda/device/error.cuh"

namespace ATV::cuda::scf::internal {

/// Exact nuclear identity cache for resident RHF/UHF methods. Only the changed
/// scalar leaves the device; reduction storage is reused at fixed atom count.
class NuclearIdentity {
  public:
    // NVCC's generated CUB launch stubs must be able to name this functor.
    struct Difference {
        const int* current;
        const int* previous;
        __host__ __device__ int operator()(int i) const {
            return current[i] != previous[i];
        }
    };

  private:
    using Input =
        thrust::transform_iterator<Difference, thrust::counting_iterator<int>>;

  public:
    bool matches(const int* types, int n, cudaStream_t stream) {
        if (!recorded_ || types_.size() != static_cast<std::size_t>(n))
            return false;
        Input input(thrust::counting_iterator<int>(0),
                    Difference{types, thrust::raw_pointer_cast(types_.data())});
        std::size_t bytes = scratch_.size();
        ATV_CUDA_CHECK(cub::DeviceReduce::Max(
            thrust::raw_pointer_cast(scratch_.data()), bytes, input,
            thrust::raw_pointer_cast(changed_.data()), n, stream));
        int changed = 0;
        ATV_CUDA_CHECK(
            cudaMemcpyAsync(&changed, thrust::raw_pointer_cast(changed_.data()),
                            sizeof(int), cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        return changed == 0;
    }

    void record(const int* types, int n, cudaStream_t stream) {
        const bool resize =
            !recorded_ || types_.size() != static_cast<std::size_t>(n);
        recorded_ = false;
        if (resize) {
            types_.resize(static_cast<std::size_t>(n));
            changed_.resize(1);
            Input input(
                thrust::counting_iterator<int>(0),
                Difference{types, thrust::raw_pointer_cast(types_.data())});
            std::size_t bytes = 0;
            ATV_CUDA_CHECK(cub::DeviceReduce::Max(
                nullptr, bytes, input,
                thrust::raw_pointer_cast(changed_.data()), n, stream));
            scratch_.resize(bytes);
        }
        ATV_CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(types_.data()),
                                       types, n * sizeof(int),
                                       cudaMemcpyDeviceToDevice, stream));
        recorded_ = true;
    }

  private:
    thrust::device_vector<int> types_, changed_;
    thrust::device_vector<char> scratch_;
    bool recorded_ = false;
};

} // namespace ATV::cuda::scf::internal
