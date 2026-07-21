/// @file cuda/pw/internal/grid_map.cu
#include "atomiverse/cuda/pw/internal/grid_map.cuh"

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

namespace {

/// @brief Fold Miller (h,k,l) onto the half-complex grid index, mirroring the
///        CPU pad loops. Returns the linear index and sets @p conj.
__device__ __forceinline__ int fold_index(int h, int k, int l, int nx, int ny,
                                          int nz, bool& conj) {
    int kx = ((h % nx) + nx) % nx;
    int ky = ((k % ny) + ny) % ny;
    int kz = ((l % nz) + nz) % nz;
    conj = false;
    if (kx > nx / 2) {
        kx = nx - kx;
        ky = (ny - ky) % ny;
        kz = (nz - kz) % nz;
        conj = true;
    }
    const int nxc = nx / 2 + 1;
    return kx + nxc * (ky + ny * kz);
}

__global__ void scatter_kernel(const cufftDoubleComplex* __restrict__ gspace,
                               const int* __restrict__ h,
                               const int* __restrict__ k,
                               const int* __restrict__ l, int n_g,
                               cufftDoubleComplex* __restrict__ halfcomplex,
                               int nx, int ny, int nz, int nc) {
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= n_g)
        return;
    bool conj;
    const int idx = fold_index(h[ig], k[ig], l[ig], nx, ny, nz, conj);
    if (idx < 0 || idx >= nc)
        return;
    cufftDoubleComplex v = gspace[ig];
    halfcomplex[idx].x = v.x;
    halfcomplex[idx].y = conj ? -v.y : v.y;
}

__global__ void
gather_kernel(const cufftDoubleComplex* __restrict__ halfcomplex,
              const int* __restrict__ h, const int* __restrict__ k,
              const int* __restrict__ l, int n_g,
              cufftDoubleComplex* __restrict__ gspace, int nx, int ny, int nz,
              int nc, double scale) {
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= n_g)
        return;
    bool conj;
    const int idx = fold_index(h[ig], k[ig], l[ig], nx, ny, nz, conj);
    if (idx < 0 || idx >= nc) {
        gspace[ig].x = 0.0;
        gspace[ig].y = 0.0;
        return;
    }
    cufftDoubleComplex v = halfcomplex[idx];
    gspace[ig].x = v.x * scale;
    gspace[ig].y = (conj ? -v.y : v.y) * scale;
}

} // namespace

void scatter_g_to_halfcomplex(const cufftDoubleComplex* d_gspace,
                              const int* d_h, const int* d_k, const int* d_l,
                              int n_g, cufftDoubleComplex* d_halfcomplex,
                              int nx, int ny, int nz, cudaStream_t stream) {
    const int nc = (nx / 2 + 1) * ny * nz;
    cudaMemsetAsync(d_halfcomplex, 0,
                    static_cast<size_t>(nc) * sizeof(cufftDoubleComplex),
                    stream);
    if (n_g <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid = (n_g + kBlock - 1) / kBlock;
    scatter_kernel<<<grid, kBlock, 0, stream>>>(d_gspace, d_h, d_k, d_l, n_g,
                                                d_halfcomplex, nx, ny, nz, nc);
}

void gather_halfcomplex_to_g(const cufftDoubleComplex* d_halfcomplex,
                             const int* d_h, const int* d_k, const int* d_l,
                             int n_g, cufftDoubleComplex* d_gspace, int nx,
                             int ny, int nz, double scale,
                             cudaStream_t stream) {
    if (n_g <= 0)
        return;
    const int nc = (nx / 2 + 1) * ny * nz;
    constexpr int kBlock = 256;
    const int grid = (n_g + kBlock - 1) / kBlock;
    gather_kernel<<<grid, kBlock, 0, stream>>>(
        d_halfcomplex, d_h, d_k, d_l, n_g, d_gspace, nx, ny, nz, nc, scale);
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
