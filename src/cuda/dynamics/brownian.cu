/// @file src/cuda/dynamics/brownian.cu
/// @brief Device overdamped Brownian step (`brownian_integrate`).

#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/brownian.cuh"
#include "atomiverse/cuda/dynamics/brownian_step.h"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace ATV {

namespace cuda {
namespace md {

namespace {

__global__ void brownian_integrate_kernel(
    pos_t* __restrict__ x, pos_t* __restrict__ y, pos_t* __restrict__ z,
    pos_t* __restrict__ xc, pos_t* __restrict__ yc, pos_t* __restrict__ zc,
    const fixed_t* __restrict__ fx, const fixed_t* __restrict__ fy,
    const fixed_t* __restrict__ fz, ForceFault* fault,
    const std::int64_t* __restrict__ tag, mixed_t mobility, mixed_t dt,
    mixed_t noise, cuda::Box box, std::uint64_t seed, BigInt step, int n) {
    if (!forces_consumable(fault, tag, step))
        return;
    box = box.current();
    const mixed_t Lx = box.length(0), Ly = box.length(1), Lz = box.length(2);
    const bool wrap = brownian_should_wrap(box.ortho, double(Lx));
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const std::uint64_t tag_i = tag ? static_cast<std::uint64_t>(tag[i])
                                    : static_cast<std::uint64_t>(i) + 1ull;
    curandStatePhilox4_32_10_t state;
    // Reserve a disjoint counter range from initialization and Langevin.
    // step is an int, so eight outputs/step cannot exhaust this 2^62 window.
    curand_init(seed, tag_i,
                (1ull << 62) + 8ull * static_cast<std::uint64_t>(step), &state);
    const double2 xy = curand_normal2_double(&state);
    const double2 zw = curand_normal2_double(&state);
    const mixed_t gx = mixed_t(xy.x), gy = mixed_t(xy.y), gz = mixed_t(zw.x);

    const mixed_t fxi = decode_force<mixed_t>(fx[i]);
    const mixed_t fyi = decode_force<mixed_t>(fy[i]);
    const mixed_t fzi = decode_force<mixed_t>(fz[i]);
    const bool finite = brownian_forces_finite(fxi, fyi, fzi);

    mixed_t nx = brownian_advance(load_position(x, xc, i), fxi, finite,
                                  mobility, dt, noise, gx);
    mixed_t ny = brownian_advance(load_position(y, yc, i), fyi, finite,
                                  mobility, dt, noise, gy);
    mixed_t nz = brownian_advance(load_position(z, zc, i), fzi, finite,
                                  mobility, dt, noise, gz);
    if (wrap) {
        nx = brownian_wrap(nx, Lx);
        ny = brownian_wrap(ny, Ly);
        nz = brownian_wrap(nz, Lz);
    }
    store_position(x, xc, i, nx);
    store_position(y, yc, i, ny);
    store_position(z, zc, i, nz);
}

} // namespace

void Brownian::launch(const LaunchContext& ctx) {
    if (!ctx.at(ATV::md::Phase::initial_integrate))
        return;

    auto& frame = ctx.frame;
    const int n = frame.n_atoms();
    const mixed_t dt = static_cast<mixed_t>(dt_);
    const mixed_t D = static_cast<mixed_t>(D_);
    const mixed_t temperature = static_cast<mixed_t>(temperature_);
    const mixed_t mobility = brownian_mobility(D, temperature);
    const mixed_t noise = brownian_noise_amplitude(D, dt);
    const cuda::Box& box = ctx.runtime.host_box;
    const int block = cuda::kDefaultBlock;
    const int grid = cuda::grid_size(n);
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    const auto& a = frame.atoms;
    brownian_integrate_kernel<<<grid, block, 0, stream>>>(
        a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.fx, a.fy, a.fz,
        frame.force_fault, a.tag, mobility, dt, noise, box, seed_, ctx.step, n);
}

} // namespace md
} // namespace cuda
} // namespace ATV
