/// @file cuda/device/error.cuh
/// @brief Project-wide CUDA error checking.
///
/// Usable from any .cu file. Throws std::runtime_error on failure.
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {

/// @brief Check a CUDA Runtime API call and throw on failure.
/// @param err   Return code from a CUDA runtime call.
/// @param file  Source file (__FILE__).
/// @param line  Source line (__LINE__).
inline void check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ")
                                 + cudaGetErrorString(err) + " at " + file + ":"
                                 + std::to_string(line));
    }
}

} // namespace cuda
} // namespace ATV

/// @brief Check a CUDA runtime call; throws std::runtime_error on failure.
#define ATV_CUDA_CHECK(err) ::ATV::cuda::check((err), __FILE__, __LINE__)
