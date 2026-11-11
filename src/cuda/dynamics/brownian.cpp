/// @file src/cuda/dynamics/brownian.cpp
/// @brief Host `cuda::md::Brownian::launch` for builds without the device
///        kernel (`ATV_BUILD_CUDA=OFF`).

#include "atomiverse/cuda/dynamics/brownian.cuh"

#include <stdexcept>

namespace ATV {
namespace cuda {
namespace md {

void Brownian::launch(const LaunchContext& ctx) {
    if (!ctx.at(ATV::md::Phase::initial_integrate))
        return;
    throw std::runtime_error(
        "cuda::md::Brownian: kernel brownian_integrate does not exist");
}

} // namespace md
} // namespace cuda
} // namespace ATV
