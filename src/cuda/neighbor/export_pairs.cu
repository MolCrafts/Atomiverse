/// @file cuda/neighbor/export_pairs.cu
/// @brief Cluster flat pairs → atom-level PairList with full triclinic PBC.

#include "nlist_data.cuh"
#include "nlist_error.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/types.cuh"

#include <cuda_runtime.h>

namespace ATV::cuda::md::nlist {

// Templated on `bool Ortho` so the per-pair PBC wrap dead-eliminates
// the triclinic branch in the orthorhombic specialization
// (cuda-md-review-cleanup-04-box-by-value-pbc).
//
// Multi-word masks (CS=32): each flat pair has kMaskWords uint64s; bit
// (i_slot * CS + j_slot) marks an interacting atom pair.
template <bool Ortho>
__global__ void export_atom_pairs_kernel(
    const int* __restrict__ flat_ci, const int* __restrict__ flat_cj,
    const uint64_t* __restrict__ flat_mask,
    const int* __restrict__ num_flat_device,
    const int* __restrict__ original_index, const real_t* __restrict__ pos_x,
    const real_t* __restrict__ pos_y, const real_t* __restrict__ pos_z,
    bool periodic, cuda::Box box, int* __restrict__ out_i,
    int* __restrict__ out_j, real_t* __restrict__ out_dx,
    real_t* __restrict__ out_dy, real_t* __restrict__ out_dz,
    real_t* __restrict__ out_dist_sq, int* __restrict__ out_count,
    const int* __restrict__ capacity_ptr) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_flat = *num_flat_device;
    if (idx >= n_flat)
        return;

    int ci = flat_ci[idx];
    int cj = flat_cj[idx];
    int cap = *capacity_ptr;
    const uint64_t* mwords =
        flat_mask + static_cast<std::size_t>(idx) * kMaskWords;

    int my_count = 0;
#pragma unroll
    for (int w = 0; w < kMaskWords; ++w)
        my_count += __popcll(mwords[w]);

    int my_base = 0;
    if (my_count > 0)
        my_base = atomicAdd(out_count, my_count);

    int local = 0;
    for (int w = 0; w < kMaskWords; ++w) {
        uint64_t mask = mwords[w];
        while (mask != 0) {
            int bit = __ffsll(mask) - 1;
            mask &= mask - 1;
            int linear = w * 64 + bit;
            int si = linear / kClusterSize;
            int sj = linear % kClusterSize;

            int ai = original_index[ci * kClusterSize + si];
            int aj = original_index[cj * kClusterSize + sj];
            if (ai < 0 || aj < 0)
                continue;

            work_t dx = pos_x[aj] - pos_x[ai];
            work_t dy = pos_y[aj] - pos_y[ai];
            work_t dz = pos_z[aj] - pos_z[ai];

            if (periodic) {
                if constexpr (Ortho) {
                    dx -= work_t(box.cell[0])
                          * nearbyint(dx * work_t(box.inv_l[0]));
                    dy -= work_t(box.cell[4])
                          * nearbyint(dy * work_t(box.inv_l[1]));
                    dz -= work_t(box.cell[8])
                          * nearbyint(dz * work_t(box.inv_l[2]));
                } else {
                    box.min_image(dx, dy, dz);
                }
            }

            work_t r2 = dx * dx + dy * dy + dz * dz;
            int slot = my_base + local;
            if (slot < cap) {
                out_i[slot] = ai;
                out_j[slot] = aj;
                out_dx[slot] = dx;
                out_dy[slot] = dy;
                out_dz[slot] = dz;
                out_dist_sq[slot] = r2;
            }
            local++;
        }
    }
}

void launch_export_atom_pairs(
    const int* flat_ci, const int* flat_cj, const uint64_t* flat_mask,
    const int* num_flat_device, int n_flat, const int* original_index,
    const real_t* pos_x, const real_t* pos_y, const real_t* pos_z,
    bool periodic, cuda::Box box, int* out_i, int* out_j, real_t* out_dx,
    real_t* out_dy, real_t* out_dz, real_t* out_dist_sq, int* out_count,
    const int* capacity_ptr, cudaStream_t stream) {
    if (n_flat == 0)
        return;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (n_flat + block - 1) / block;
    if (box.ortho) {
        export_atom_pairs_kernel<true><<<grid, block, 0, stream>>>(
            flat_ci, flat_cj, flat_mask, num_flat_device, original_index, pos_x,
            pos_y, pos_z, periodic, box, out_i, out_j, out_dx, out_dy, out_dz,
            out_dist_sq, out_count, capacity_ptr);
    } else {
        export_atom_pairs_kernel<false><<<grid, block, 0, stream>>>(
            flat_ci, flat_cj, flat_mask, num_flat_device, original_index, pos_x,
            pos_y, pos_z, periodic, box, out_i, out_j, out_dx, out_dy, out_dz,
            out_dist_sq, out_count, capacity_ptr);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

} // namespace ATV::cuda::md::nlist
