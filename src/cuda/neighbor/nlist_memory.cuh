#pragma once

/// @file cuda/neighbor/nlist_memory.cuh
/// @brief RAII device memory wrapper for neighborlist internals.

#include "nlist_error.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ATV::cuda::md::nlist {

/// @brief Lightweight RAII device buffer for internal neighborlist use.
///
/// Unlike cuda::DeviceBuffer, this supports resize without reallocating
/// when the requested size fits in the existing allocation.
template <typename T>
class DeviceVector {
  public:
    DeviceVector() = default;

    explicit DeviceVector(std::size_t count) { resize(count); }

    ~DeviceVector() {
        if (ptr_)
            cudaFree(ptr_);
    }

    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;

    DeviceVector(DeviceVector&& o) noexcept
        : ptr_(o.ptr_), size_(o.size_), capacity_(o.capacity_) {
        o.ptr_ = nullptr;
        o.size_ = 0;
        o.capacity_ = 0;
    }

    DeviceVector& operator=(DeviceVector&& o) noexcept {
        if (this != &o) {
            if (ptr_)
                cudaFree(ptr_);
            ptr_ = o.ptr_;
            size_ = o.size_;
            capacity_ = o.capacity_;
            o.ptr_ = nullptr;
            o.size_ = 0;
            o.capacity_ = 0;
        }
        return *this;
    }

    /// @brief Resize the buffer. Does NOT preserve contents on reallocation.
    void resize(std::size_t count) {
        if (count > capacity_) {
            if (ptr_)
                cudaFree(ptr_);
            ptr_ = nullptr;
            // Geometric growth to amortize allocation cost
            std::size_t new_cap = std::max(count, capacity_ * 2);
            if (new_cap > 0) {
                ATV_CUDA_CHECK(cudaMalloc(&ptr_, new_cap * sizeof(T)));
                capacity_ = new_cap;
            } else {
                capacity_ = 0;
            }
        }
        size_ = count;
    }

    void clear() { size_ = 0; }

    void memset(int value) {
        if (size_ > 0)
            ATV_CUDA_CHECK(cudaMemset(ptr_, value, size_ * sizeof(T)));
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }
    std::size_t capacity() const { return capacity_; }
    std::size_t bytes() const { return size_ * sizeof(T); }
    bool empty() const { return size_ == 0; }

  private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

} // namespace ATV::cuda::md::nlist
