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

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ATV::cuda::md::nlist {

namespace {

// Identical to atom cell_of convention (same wrap + clamp + linearization:
// cell = (cx*dy + cy)*dz + cz, cx slowest / cz fastest).
__device__ __forceinline__ int sort_cell_of(real_t x, real_t y, real_t z,
                                            real_t Lx, real_t Ly, real_t Lz,
                                            int dx, int dy, int dz,
                                            bool periodic) {
    real_t fx = x / Lx, fy = y / Ly, fz = z / Lz;
    if (periodic) {
        fx -= floorf(fx);
        fy -= floorf(fy);
        fz -= floorf(fz);
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
compute_cell_key_k(const real_t* __restrict__ x, const real_t* __restrict__ y,
                   const real_t* __restrict__ z, int n, real_t Lx, real_t Ly,
                   real_t Lz, int dx, int dy, int dz, bool periodic,
                   int* __restrict__ key, int* __restrict__ perm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    key[i] = sort_cell_of(x[i], y[i], z[i], Lx, Ly, Lz, dx, dy, dz, periodic);
    perm[i] = i;
}

// inverse[order[k]] = k.
__global__ void build_inverse_k(const int* __restrict__ order, int n,
                                int* __restrict__ inverse) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;
    inverse[order[k]] = k;
}

// Gather one width-`cols` row-major column block through `order`:
//   dst[k*cols + c] = src[order[k]*cols + c]
template <typename T>
__global__ void gather_rows_k(const T* __restrict__ src, T* __restrict__ dst,
                              const int* __restrict__ order, int n, int cols) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * cols;
    if (t >= total)
        return;
    int k = t / cols;
    int c = t - k * cols;
    dst[static_cast<std::size_t>(k) * cols + c] =
        src[static_cast<std::size_t>(order[k]) * cols + c];
}

template <typename T>
void gather_rows(const T* src, T* dst, const int* order, int n, int cols,
                 cudaStream_t stream) {
    if (n == 0 || cols == 0 || src == nullptr)
        return;
    int total = n * cols;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (total + block - 1) / block;
    gather_rows_k<T><<<grid, block, 0, stream>>>(src, dst, order, n, cols);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// Permute one width-1 real_t array in place via stage_r (cols==1 fast path).
template <typename T>
void permute_r(T* arr, const int* order, int n, DeviceVector<T>& stage,
               cudaStream_t stream) {
    if (arr == nullptr || n == 0)
        return;
    if (stage.size() < static_cast<std::size_t>(n))
        stage.resize(n);
    gather_rows<T>(arr, stage.data(), order, n, 1, stream);
    ATV_CUDA_CHECK(cudaMemcpyAsync(arr, stage.data(),
                                   static_cast<std::size_t>(n) * sizeof(T),
                                   cudaMemcpyDeviceToDevice, stream));
}

// Permute one width-`cols` int array in place via stage_i.
void permute_i(int* arr, const int* order, int n, int cols,
               DeviceVector<int>& stage, cudaStream_t stream) {
    if (arr == nullptr || n == 0 || cols == 0)
        return;
    const std::size_t need = static_cast<std::size_t>(n) * cols;
    if (stage.size() < need)
        stage.resize(need);
    gather_rows<int>(arr, stage.data(), order, n, cols, stream);
    ATV_CUDA_CHECK(cudaMemcpyAsync(arr, stage.data(), need * sizeof(int),
                                   cudaMemcpyDeviceToDevice, stream));
}

// Permute one width-`cols` int64 array in place via stage_i64.
void permute_i64(std::int64_t* arr, const int* order, int n, int cols,
                 DeviceVector<std::int64_t>& stage, cudaStream_t stream) {
    if (arr == nullptr || n == 0 || cols == 0)
        return;
    const std::size_t need = static_cast<std::size_t>(n) * cols;
    if (stage.size() < need)
        stage.resize(need);
    gather_rows<std::int64_t>(arr, stage.data(), order, n, cols, stream);
    ATV_CUDA_CHECK(cudaMemcpyAsync(arr, stage.data(),
                                   need * sizeof(std::int64_t),
                                   cudaMemcpyDeviceToDevice, stream));
}

// Permute the image array (width 3, std::int8_t) via stage_i8.
void permute_i8(std::int8_t* arr, const int* order, int n, int cols,
                DeviceVector<std::int8_t>& stage, cudaStream_t stream) {
    if (arr == nullptr || n == 0 || cols == 0)
        return;
    const std::size_t need = static_cast<std::size_t>(n) * cols;
    if (stage.size() < need)
        stage.resize(need);
    gather_rows<std::int8_t>(arr, stage.data(), order, n, cols, stream);
    ATV_CUDA_CHECK(cudaMemcpyAsync(arr, stage.data(),
                                   need * sizeof(std::int8_t),
                                   cudaMemcpyDeviceToDevice, stream));
}

} // namespace

void apply_permutation_owned(Frame& frame, const int* order, int n,
                             AtomCellSortBuffers& buf, cudaStream_t stream) {
    if (n <= 0)
        return;
    auto& a = frame.atoms;

    // -- LJ SoA (width 1) --
    permute_r(a.x, order, n, buf.stage_r, stream);
    permute_r(a.y, order, n, buf.stage_r, stream);
    permute_r(a.z, order, n, buf.stage_r, stream);
    permute_r(a.x_correction, order, n, buf.stage_r, stream);
    permute_r(a.y_correction, order, n, buf.stage_r, stream);
    permute_r(a.z_correction, order, n, buf.stage_r, stream);
    permute_r(a.vx, order, n, buf.stage_motion, stream);
    permute_r(a.vy, order, n, buf.stage_motion, stream);
    permute_r(a.vz, order, n, buf.stage_motion, stream);
    permute_r(a.fx, order, n, buf.stage_r, stream);
    permute_r(a.fy, order, n, buf.stage_r, stream);
    permute_r(a.fz, order, n, buf.stage_r, stream);
    permute_r(a.mass, order, n, buf.stage_motion, stream);
    permute_r(a.charge, order, n, buf.stage_r, stream);
    permute_i(a.type, order, n, 1, buf.stage_i, stream);
    permute_i64(a.tag, order, n, 1, buf.stage_i64, stream);
    permute_i8(a.image, order, n, 3, buf.stage_i8, stream); // image[i*3+d]

    // -- Bonded SoA (multi-GPU only; null on single-GPU). Runtime strides
    //    mirror transfer.cu::materialize_per_atom_bonded exactly. --
    permute_i(a.num_bond, order, n, 1, buf.stage_i, stream);
    permute_i64(a.bond_atom, order, n, a.bond_per_atom, buf.stage_i64, stream);
    permute_i(a.bond_type, order, n, a.bond_per_atom, buf.stage_i, stream);

    permute_i(a.num_angle, order, n, 1, buf.stage_i, stream);
    permute_i64(a.angle_atom, order, n, 2 * a.angle_per_atom, buf.stage_i64,
                stream);
    permute_i(a.angle_type, order, n, a.angle_per_atom, buf.stage_i, stream);

    permute_i(a.num_dihedral, order, n, 1, buf.stage_i, stream);
    permute_i64(a.dihedral_atom, order, n, 3 * a.dihedral_per_atom,
                buf.stage_i64, stream);
    permute_i(a.dihedral_type, order, n, a.dihedral_per_atom, buf.stage_i,
              stream);

    permute_i(a.nspecial, order, n, 3, buf.stage_i, stream);
    permute_i64(a.special, order, n, a.maxspecial, buf.stage_i64, stream);
}

void cell_sort_owned(Frame& frame, real_t Lx, real_t Ly, real_t Lz, int dx,
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

    // 1. cell key + identity index per owned atom.
    {
        int block = ATV::cuda::kDefaultBlock;
        int grid = (n + block - 1) / block;
        compute_cell_key_k<<<grid, block, 0, stream>>>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z, n, Lx, Ly, Lz, dx, dy,
            dz, periodic, buf.cell_key.data(), buf.perm.data());
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

    // 3. inverse permutation so the caller can undo the sort.
    {
        int block = ATV::cuda::kDefaultBlock;
        int grid = (n + block - 1) / block;
        build_inverse_k<<<grid, block, 0, stream>>>(buf.perm_out.data(), n,
                                                    buf.inverse.data());
        ATV_CUDA_CHECK(cudaGetLastError());
    }

    // 4. gather every per-atom array through perm_out (row k <- old perm_out[k]).
    apply_permutation_owned(frame, buf.perm_out.data(), n, buf, stream);
}

} // namespace ATV::cuda::md::nlist
