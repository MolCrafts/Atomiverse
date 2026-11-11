/// @file cuda/neighbor/atom_cell_sort.cu
/// @brief Device-side cell-sort of one Frame's OWNED atoms (spec
///        multi-gpu-atom-cell-sort-01-permute).
///
/// See atom_cell_sort.cuh for the contract and the exact per-atom stride map.
/// The cell-key kernel mirrors atom cell_of convention byte-for-byte so the
/// sort order agrees with the grid the Verlet build later bins against.

#include "atom_cell_sort.cuh"

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ATV::cuda::md::nlist {

namespace {

// Identical to atom cell_of convention (same wrap + clamp + linearization:
// cell = (cx*dy + cy)*dz + cz, cx slowest / cz fastest). Reads the primary
// coordinate; the key math is force_t scratch.
__device__ __forceinline__ int sort_cell_of(pos_t x, pos_t y, pos_t z,
                                            force_t Lx, force_t Ly, force_t Lz,
                                            int dx, int dy, int dz,
                                            bool periodic) {
    force_t fx = x / Lx, fy = y / Ly, fz = z / Lz;
    if (periodic) {
        fx -= floor(fx);
        fy -= floor(fy);
        fz -= floor(fz);
    }
    int cx = static_cast<int>(fx * dx);
    int cy = static_cast<int>(fy * dy);
    int cz = static_cast<int>(fz * dz);
    cx = cx < 0 ? 0 : (cx >= dx ? dx - 1 : cx);
    cy = cy < 0 ? 0 : (cy >= dy ? dy - 1 : cy);
    cz = cz < 0 ? 0 : (cz >= dz ? dz - 1 : cz);
    return (cx * dy + cy) * dz + cz;
}

__global__ void
compute_cell_key_k(const pos_t* __restrict__ x, const pos_t* __restrict__ y,
                   const pos_t* __restrict__ z, int n, force_t Lx, force_t Ly,
                   force_t Lz, int dx, int dy, int dz, bool periodic,
                   int* __restrict__ key, int* __restrict__ perm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    key[i] = sort_cell_of(x[i], y[i], z[i], Lx, Ly, Lz, dx, dy, dz, periodic);
    perm[i] = i;
}

struct RowPermutation {
    const int* order;
    int cols;
    __device__ std::size_t operator()(std::size_t i) const {
        return static_cast<std::size_t>(order[i / cols]) * cols + i % cols;
    }
};

// Stage then copy back: Thrust gather cannot overlap its input and output.
template <typename T>
void permute_rows(T* arr, const int* order, int n, int cols,
                  DeviceVector<T>& stage, cudaStream_t stream) {
    if (arr == nullptr || n == 0 || cols == 0)
        return;
    const std::size_t count = static_cast<std::size_t>(n) * cols;
    if (stage.size() < count)
        stage.resize(count);
    const auto map = thrust::make_transform_iterator(
        thrust::counting_iterator<std::size_t>(0), RowPermutation{order, cols});
    thrust::gather(thrust::cuda::par_nosync.on(stream), map, map + count, arr,
                   stage.data());
    ATV_CUDA_CHECK(cudaMemcpyAsync(arr, stage.data(), count * sizeof(T),
                                   cudaMemcpyDeviceToDevice, stream));
}

} // namespace

void apply_permutation_owned(Frame& frame, const int* order, int n,
                             AtomCellSortBuffers& buf, cudaStream_t stream) {
    if (n <= 0)
        return;
    auto& a = frame.atoms;

    // -- LJ SoA (width 1). Correction columns are null (skipped) unless
    //    kPositionCorrection; every column is staged in its own type. --
    permute_rows(a.x, order, n, 1, buf.stage_pos, stream);
    permute_rows(a.y, order, n, 1, buf.stage_pos, stream);
    permute_rows(a.z, order, n, 1, buf.stage_pos, stream);
    permute_rows(a.x_corr, order, n, 1, buf.stage_pos, stream);
    permute_rows(a.y_corr, order, n, 1, buf.stage_pos, stream);
    permute_rows(a.z_corr, order, n, 1, buf.stage_pos, stream);
    permute_rows(a.vx, order, n, 1, buf.stage_mixed, stream);
    permute_rows(a.vy, order, n, 1, buf.stage_mixed, stream);
    permute_rows(a.vz, order, n, 1, buf.stage_mixed, stream);
    permute_rows(a.fx, order, n, 1, buf.stage_fixed, stream);
    permute_rows(a.fy, order, n, 1, buf.stage_fixed, stream);
    permute_rows(a.fz, order, n, 1, buf.stage_fixed, stream);
    permute_rows(a.mass, order, n, 1, buf.stage_mixed, stream);
    permute_rows(a.charge, order, n, 1, buf.stage_pos, stream);
    permute_rows(a.type, order, n, 1, buf.stage_i, stream);
    permute_rows(a.atomic_number, order, n, 1, buf.stage_i, stream);
    permute_rows(a.tag, order, n, 1, buf.stage_i64, stream);
    permute_rows(a.image, order, n, 3, buf.stage_image, stream); // image[i*3+d]

    // -- Bonded SoA (multi-GPU only; null on single-GPU). Runtime strides
    //    mirror transfer.cu::materialize_per_atom_bonded exactly. --
    permute_rows(a.num_bond, order, n, 1, buf.stage_i, stream);
    permute_rows(a.bond_atom, order, n, a.bond_per_atom, buf.stage_i64, stream);
    permute_rows(a.bond_type, order, n, a.bond_per_atom, buf.stage_i, stream);

    permute_rows(a.num_angle, order, n, 1, buf.stage_i, stream);
    permute_rows(a.angle_atom, order, n, 2 * a.angle_per_atom, buf.stage_i64,
                 stream);
    permute_rows(a.angle_type, order, n, a.angle_per_atom, buf.stage_i, stream);

    permute_rows(a.num_dihedral, order, n, 1, buf.stage_i, stream);
    permute_rows(a.dihedral_atom, order, n, 3 * a.dihedral_per_atom,
                 buf.stage_i64, stream);
    permute_rows(a.dihedral_type, order, n, a.dihedral_per_atom, buf.stage_i,
                 stream);

    permute_rows(a.nspecial, order, n, 3, buf.stage_i, stream);
    permute_rows(a.special, order, n, a.maxspecial, buf.stage_i64, stream);
}

void cell_sort_owned(Frame& frame, geom_t Lx, geom_t Ly, geom_t Lz, int dx,
                     int dy, int dz, bool periodic, AtomCellSortBuffers& buf,
                     cudaStream_t stream) {
    const int n = frame.n_local();
    if (n <= 0)
        return;

    buf.resize(n);
    // Pre-size both the int64 and int staging to the widest per-atom column we
    // touch so apply_permutation_owned never grows a stage mid-loop (a grow
    // would force a device sync). Runtime strides; 3 covers image/nspecial.
    const auto& a = frame.atoms;
    int max_cols = 3;
    max_cols = std::max(max_cols, a.bond_per_atom);
    max_cols = std::max(max_cols, 2 * a.angle_per_atom);
    max_cols = std::max(max_cols, 3 * a.dihedral_per_atom);
    max_cols = std::max(max_cols, a.maxspecial);
    buf.resize_bonded(n, max_cols);

    // 1. cell key + identity index per owned atom (cell narrowed once).
    {
        int block = ATV::cuda::kDefaultBlock;
        int grid = (n + block - 1) / block;
        compute_cell_key_k<<<grid, block, 0, stream>>>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z, n, force_t(Lx),
            force_t(Ly), force_t(Lz), dx, dy, dz, periodic, buf.cell_key.data(),
            buf.perm.data());
        ATV_CUDA_CHECK(cudaGetLastError());
    }

    // 2. cub radix-sort indices by key (pattern from sort_and_cluster.cu).
    //    Keys are cell indices in [0, dx*dy*dz); 32-bit keys, full range.
    {
        size_t needed = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, needed, buf.cell_key.data(), buf.cell_key_out.data(),
            buf.perm.data(), buf.perm_out.data(), n, 0, 32, stream);
        if (needed > buf.sort_temp.capacity())
            buf.sort_temp.resize(needed);
        size_t avail = buf.sort_temp.capacity();
        cub::DeviceRadixSort::SortPairs(
            buf.sort_temp.data(), avail, buf.cell_key.data(),
            buf.cell_key_out.data(), buf.perm.data(), buf.perm_out.data(), n, 0,
            32, stream);
        ATV_CUDA_CHECK(cudaGetLastError());
    }

    // 3. inverse[perm[k]] = k using Thrust scatter on the caller's stream.
    const thrust::counting_iterator<int> indices(0);
    thrust::scatter(thrust::cuda::par_nosync.on(stream), indices, indices + n,
                    buf.perm_out.data(), buf.inverse.data());

    // 4. gather every per-atom array through perm_out (row k <- old perm_out[k]).
    apply_permutation_owned(frame, buf.perm_out.data(), n, buf, stream);
}

} // namespace ATV::cuda::md::nlist
