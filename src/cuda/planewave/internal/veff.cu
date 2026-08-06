/// @file cuda/planewave/internal/veff.cu
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/planewave/internal/veff.cuh"

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

namespace {

/// One thread per G. Association matches the CPU expression
/// `V_eff_G[i] = V_loc_G[i] + V_H_G[i] + V_xc_G[i]` (left-to-right) per
/// component, so the two backends agree bit-for-bit on identical inputs.
__global__ void
veff_assemble_kernel(const cufftDoubleComplex* __restrict__ vloc,
                     const cufftDoubleComplex* __restrict__ vh,
                     const cufftDoubleComplex* __restrict__ vxc,
                     cufftDoubleComplex* __restrict__ veff, int n_g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_g)
        return;
    veff[i].x = vloc[i].x + vh[i].x + vxc[i].x;
    veff[i].y = vloc[i].y + vh[i].y + vxc[i].y;
}

} // namespace

void veff_assemble_g(const cufftDoubleComplex* d_Vloc_G,
                     const cufftDoubleComplex* d_VH_G,
                     const cufftDoubleComplex* d_Vxc_G,
                     cufftDoubleComplex* d_Veff_G, int n_g,
                     cudaStream_t stream) {
    if (n_g <= 0)
        return;
    constexpr int kBlock = ::ATV::cuda::kDefaultBlock;
    veff_assemble_kernel<<<::ATV::cuda::grid_size(n_g, kBlock), kBlock, 0,
                           stream>>>(d_Vloc_G, d_VH_G, d_Vxc_G, d_Veff_G, n_g);
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
