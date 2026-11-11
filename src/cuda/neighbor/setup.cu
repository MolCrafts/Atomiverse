/// @file cuda/neighbor/setup.cu
/// @brief GPU-side inverse-cell update for the cluster hierarchy.
///
/// All computation runs on the GPU. No host-side math, no H2D transfers.

#include "nlist_impl.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace ATV::cuda::md::nlist {

// Invert in geom_t (FP64) and narrow the result once for force-path keys.
__global__ void update_inverse_cell_k(const geom_t* box, force_t* inverse) {
    geom_t inv[9];
    invert_3x3(box, inv);
    for (int i = 0; i < 9; ++i)
        inverse[i] = static_cast<force_t>(inv[i]);
}

void launch_update_inverse_cell(const geom_t* box, force_t* inverse,
                                cudaStream_t stream) {
    update_inverse_cell_k<<<1, 1, 0, stream>>>(box, inverse);
    ATV_CUDA_CHECK(cudaGetLastError());
}

} // namespace ATV::cuda::md::nlist
