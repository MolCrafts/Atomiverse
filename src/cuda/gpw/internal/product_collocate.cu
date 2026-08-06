/// @file cuda/gpw/internal/product_collocate.cu
/// @brief Device product-Gaussian collocation foothold for GPW density.
///
/// Continuous product formula matching the collint shape used by
/// `cpu::gpw::internal::collint_collocate_product`. All scientific loops are
/// device kernels — no host loops over grid points or products in the hot path.
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

/// @brief Integer power x^n for n ≥ 0 (n = 0 → 1).
__device__ __forceinline__ double ipow_nonneg(double x, int n) {
    double v = 1.0;
    for (int i = 0; i < n; ++i)
        v *= x;
    return v;
}

/// @brief One product shape at Cartesian r (Bohr).
__device__ __forceinline__ double product_shape(const DeviceProduct& q,
                                               const double r[3]) {
    if (q.weight == 0.0)
        return 0.0;
    const double dx = r[0] - q.pc[0];
    const double dy = r[1] - q.pc[1];
    const double dz = r[2] - q.pc[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    double v = q.weight * exp(-q.p * r2);
    // (r−ra)^la (r−rb)^lb per axis.
    for (int d = 0; d < 3; ++d) {
        const double da = r[d] - q.ra[d];
        const double db = r[d] - q.rb[d];
        v *= ipow_nonneg(da, q.la[d]) * ipow_nonneg(db, q.lb[d]);
    }
    return v;
}

/// @brief Collocate products: one thread per grid point (z-fast).
///
/// Indexing matches `cpu::gpw::UniformGrid::point`:
///   idx = (i * n1 + j) * n2 + k
///   r[d] = origin[d] + L[d] * index[d] / n[d]
__global__ void collocate_products_kernel(const DeviceProduct* __restrict__ prods,
                                          int nprod, int n0, int n1, int n2,
                                          double L0, double L1, double L2,
                                          double o0, double o1, double o2,
                                          double* __restrict__ rho) {
    const long ntot =
        static_cast<long>(n0) * static_cast<long>(n1) * static_cast<long>(n2);
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x
                     + static_cast<long>(threadIdx.x);
    if (idx >= ntot)
        return;

    const int k = static_cast<int>(idx % n2);
    const long t = idx / n2;
    const int j = static_cast<int>(t % n1);
    const int i = static_cast<int>(t / n1);

    const double r[3] = {o0 + L0 * static_cast<double>(i) / static_cast<double>(n0),
                         o1 + L1 * static_cast<double>(j) / static_cast<double>(n1),
                         o2 + L2 * static_cast<double>(k) / static_cast<double>(n2)};

    double acc = 0.0;
    for (int q = 0; q < nprod; ++q)
        acc += product_shape(prods[q], r);
    rho[idx] += acc;
}

/// @brief Block-reduce Σ ρ, atomic-accumulate into *acc.
__global__ void integrate_rho_kernel(const double* __restrict__ rho, long nr,
                                     double* __restrict__ acc) {
    __shared__ double s[256];
    const long i =
        static_cast<long>(blockIdx.x) * blockDim.x + static_cast<long>(threadIdx.x);
    s[threadIdx.x] = (i < nr) ? rho[i] : 0.0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(acc, s[0]);
}

__global__ void scale_scalar_kernel(double* x, double factor) { *x *= factor; }

} // namespace

void collocate_products(const CollocateGridDesc& grid,
                        const DeviceProduct* d_prods, int nprod, double* d_rho,
                        cudaStream_t stream) {
    if (nprod <= 0)
        return;
    const long ntot = grid.npts();
    if (ntot <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid_dim =
        static_cast<int>((ntot + kBlock - 1) / static_cast<long>(kBlock));
    collocate_products_kernel<<<grid_dim, kBlock, 0, stream>>>(
        d_prods, nprod, grid.n[0], grid.n[1], grid.n[2], grid.L[0], grid.L[1],
        grid.L[2], grid.origin[0], grid.origin[1], grid.origin[2], d_rho);
}

void collocate_product(const CollocateGridDesc& grid,
                       const DeviceProduct* d_prod, double* d_rho,
                       cudaStream_t stream) {
    collocate_products(grid, d_prod, 1, d_rho, stream);
}

void collocate_integrate_rho(const double* d_rho, long nr, double w,
                             double* d_sum, cudaStream_t stream) {
    cudaMemsetAsync(d_sum, 0, sizeof(double), stream);
    if (nr <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid_dim =
        static_cast<int>((nr + kBlock - 1) / static_cast<long>(kBlock));
    integrate_rho_kernel<<<grid_dim, kBlock, 0, stream>>>(d_rho, nr, d_sum);
    scale_scalar_kernel<<<1, 1, 0, stream>>>(d_sum, w);
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
