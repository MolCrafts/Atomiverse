/// @file cuda/gpw/internal/product_integrate.cu
/// @brief Device product-Gaussian integrate foothold for GPW V→H path.
///
/// Continuous product formula matching the collint shape used by
/// `cpu::gpw::internal::collint_integrate_product`. All scientific loops are
/// device kernels — no host loops over grid points or products in the hot path.
#include "atomiverse/cuda/gpw/internal/product_integrate.cuh"

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

/// @brief Bare product shape at Cartesian r (Bohr) — no density weight.
///
/// Matches collint integrate (shape only; k / weight applied by the caller).
__device__ __forceinline__ double product_shape_bare(const DeviceProduct& q,
                                                    const double r[3]) {
    const double dx = r[0] - q.pc[0];
    const double dy = r[1] - q.pc[1];
    const double dz = r[2] - q.pc[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    double v = exp(-q.p * r2);
    for (int d = 0; d < 3; ++d) {
        const double da = r[d] - q.ra[d];
        const double db = r[d] - q.rb[d];
        v *= ipow_nonneg(da, q.la[d]) * ipow_nonneg(db, q.lb[d]);
    }
    return v;
}

/// @brief One block per product; threads grid-stride over points, block-reduce.
///
/// Indexing matches `cpu::gpw::UniformGrid::point`:
///   idx = (i * n1 + j) * n2 + k
///   r[d] = origin[d] + L[d] * index[d] / n[d]
/// Single block per product ⇒ deterministic fp64 sum order within the block
/// reduction tree (not bit-identical to sequential host fold).
__global__ void integrate_products_kernel(const DeviceProduct* __restrict__ prods,
                                          int nprod, int n0, int n1, int n2,
                                          double L0, double L1, double L2,
                                          double o0, double o1, double o2,
                                          const double* __restrict__ V,
                                          double* __restrict__ acc) {
    const int q = static_cast<int>(blockIdx.x);
    if (q >= nprod)
        return;

    const long ntot =
        static_cast<long>(n0) * static_cast<long>(n1) * static_cast<long>(n2);
    const DeviceProduct prod = prods[q];

    double local = 0.0;
    // Grid-stride: coalesced along z-fast when consecutive threads take
    // consecutive idx (stride blockDim.x after a contiguous run).
    for (long idx = static_cast<long>(threadIdx.x); idx < ntot;
         idx += static_cast<long>(blockDim.x)) {
        const int k = static_cast<int>(idx % n2);
        const long t = idx / n2;
        const int j = static_cast<int>(t % n1);
        const int i = static_cast<int>(t / n1);
        const double r[3] = {
            o0 + L0 * static_cast<double>(i) / static_cast<double>(n0),
            o1 + L1 * static_cast<double>(j) / static_cast<double>(n1),
            o2 + L2 * static_cast<double>(k) / static_cast<double>(n2)};
        local += V[idx] * product_shape_bare(prod, r);
    }

    // Block reduce (shared tree; blockDim multiple of 32, power of two).
    __shared__ double s[256];
    s[threadIdx.x] = local;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0;
         stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        acc[q] = s[0];
}

} // namespace

void integrate_products(const CollocateGridDesc& grid,
                        const DeviceProduct* d_prods, int nprod,
                        const double* d_V, double* d_acc, cudaStream_t stream) {
    if (nprod <= 0)
        return;
    const long ntot = grid.npts();
    if (ntot <= 0) {
        cudaMemsetAsync(d_acc, 0, static_cast<size_t>(nprod) * sizeof(double),
                        stream);
        return;
    }
    // 256 threads / product: good occupancy on GH200; one block keeps the
    // per-product reduction deterministic (no multi-block atomics).
    constexpr int kBlock = 256;
    integrate_products_kernel<<<nprod, kBlock, 0, stream>>>(
        d_prods, nprod, grid.n[0], grid.n[1], grid.n[2], grid.L[0], grid.L[1],
        grid.L[2], grid.origin[0], grid.origin[1], grid.origin[2], d_V, d_acc);
}

void integrate_product(const CollocateGridDesc& grid,
                       const DeviceProduct* d_prod, const double* d_V,
                       double* d_acc, cudaStream_t stream) {
    integrate_products(grid, d_prod, 1, d_V, d_acc, stream);
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
