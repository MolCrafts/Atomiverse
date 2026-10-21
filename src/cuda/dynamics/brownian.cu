/// @file src/cuda/dynamics/brownian.cu
/// @brief Device overdamped Brownian step (`brownian_integrate`).

#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/brownian.cuh"
#include "atomiverse/cuda/dynamics/brownian_step.h"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace ATV {

using cuda::motion_t;
using cuda::real_t;

namespace cuda {
namespace md {

namespace {

__global__ void brownian_integrate_kernel(
    real_t* __restrict__ x, real_t* __restrict__ y, real_t* __restrict__ z,
    real_t* __restrict__ xc, real_t* __restrict__ yc, real_t* __restrict__ zc,
    std::int8_t* __restrict__ image, const real_t* __restrict__ fx,
    const real_t* __restrict__ fy, const real_t* __restrict__ fz,
    const std::int64_t* __restrict__ tag, motion_t mobility, motion_t dt,
    motion_t noise, cuda::MotionBox box, int wrap, std::uint64_t seed, int step,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const std::uint64_t tag_i = tag ? static_cast<std::uint64_t>(tag[i])
                                    : static_cast<std::uint64_t>(i) + 1ull;
    motion_t gx, gy, gz;
    brownian_gaussians(seed, static_cast<std::uint64_t>(step), tag_i, gx, gy,
                       gz);

    const motion_t fxi = static_cast<motion_t>(fx[i]);
    const motion_t fyi = static_cast<motion_t>(fy[i]);
    const motion_t fzi = static_cast<motion_t>(fz[i]);
    const bool finite = brownian_forces_finite(fxi, fyi, fzi);

    motion_t nx = brownian_advance(load_position(x, xc, i), fxi, finite,
                                   mobility, dt, noise, gx);
    motion_t ny = brownian_advance(load_position(y, yc, i), fyi, finite,
                                   mobility, dt, noise, gy);
    motion_t nz = brownian_advance(load_position(z, zc, i), fzi, finite,
                                   mobility, dt, noise, gz);
    if (wrap) {
        wrap_position(x, y, z, xc, yc, zc, image, i, box, nx, ny, nz);
    } else {
        store_position(x, xc, i, nx);
        store_position(y, yc, i, ny);
        store_position(z, zc, i, nz);
    }
}

} // namespace

void Brownian::launch(const LaunchContext& ctx) {
    if (!ctx.at(ATV::md::Phase::initial_integrate))
        return;

    auto& frame = ctx.frame;
    const int n = frame.n_atoms();
    if (n <= 0)
        return;
    const motion_t dt = static_cast<motion_t>(dt_);
    const motion_t D = static_cast<motion_t>(D_);
    const motion_t temperature = static_cast<motion_t>(temperature_);
    const motion_t mobility = brownian_mobility(D, temperature);
    const motion_t noise = brownian_noise_amplitude(D, dt);
    const cuda::MotionBox& box = ctx.runtime.host_motion_box;
    const int wrap =
        brownian_should_wrap(box.ortho, static_cast<double>(box.length(0))) ? 1
                                                                            : 0;
    const int block = cuda::kDefaultBlock;
    const int grid = cuda::grid_size(n);
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    brownian_integrate_kernel<<<grid, block, 0, stream>>>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.x_correction,
        frame.atoms.y_correction, frame.atoms.z_correction, frame.atoms.image,
        frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.atoms.tag,
        mobility, dt, noise, box, wrap, seed_, ctx.step, n);
    ATV_CUDA_CHECK(cudaGetLastError());
}

} // namespace md
} // namespace cuda
} // namespace ATV
