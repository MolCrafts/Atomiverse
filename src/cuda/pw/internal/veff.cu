/// @file cuda/pw/internal/veff.cu
#include "atomiverse/cuda/pw/internal/veff.cuh"

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

namespace {

__global__ void veff_assemble_kernel(const double* __restrict__ vloc,
                                     const cufftDoubleComplex* __restrict__ vh,
                                     const cufftDoubleComplex* __restrict__ vxc,
                                     cufftDoubleComplex* __restrict__ veff,
                                     int n_g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_g)
        return;
    veff[i].x = vloc[i] + vh[i].x + vxc[i].x;
    veff[i].y = vh[i].y + vxc[i].y;
}

} // namespace

void veff_assemble_g(const double* d_Vloc_real,
                     const cufftDoubleComplex* d_VH_G,
                     const cufftDoubleComplex* d_Vxc_G,
                     cufftDoubleComplex* d_Veff_G, int n_g,
                     cudaStream_t stream) {
    if (n_g <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid = (n_g + kBlock - 1) / kBlock;
    veff_assemble_kernel<<<grid, kBlock, 0, stream>>>(d_Vloc_real, d_VH_G,
                                                      d_Vxc_G, d_Veff_G, n_g);
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
