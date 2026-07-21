/// @file cuda/pw/internal/poisson.cu
#include "atomiverse/cuda/pw/internal/poisson.cuh"

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

namespace {

constexpr double kFourPi = 4.0 * 3.141592653589793;
constexpr double kG2Floor = 1e-16; ///< |G|² below this is treated as G=0.

/// @brief One thread per G-vector: V_H(G) = 4π ρ(G)/|G|², V_H(0) = 0.
__global__ void
poisson_hartree_kernel(const cufftDoubleComplex* __restrict__ rho_G,
                       const double* __restrict__ g2,
                       cufftDoubleComplex* __restrict__ V_H_G, int n_g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_g)
        return;
    const double g2i = g2[i];
    if (g2i < kG2Floor) {
        V_H_G[i].x = 0.0;
        V_H_G[i].y = 0.0;
    } else {
        const double s = kFourPi / g2i;
        V_H_G[i].x = s * rho_G[i].x;
        V_H_G[i].y = s * rho_G[i].y;
    }
}

} // namespace

void poisson_hartree(const cufftDoubleComplex* d_rho_G, const double* d_g2,
                     cufftDoubleComplex* d_V_H_G, int n_g,
                     cudaStream_t stream) {
    if (n_g <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid = (n_g + kBlock - 1) / kBlock;
    poisson_hartree_kernel<<<grid, kBlock, 0, stream>>>(d_rho_G, d_g2, d_V_H_G,
                                                        n_g);
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
