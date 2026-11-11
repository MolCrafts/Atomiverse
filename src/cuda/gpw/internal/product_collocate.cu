/// @file cuda/gpw/internal/product_collocate.cu
/// @brief Device product-Gaussian collocation for GPW density.
///
/// Periodic path: sphere-window collocate matching CPU `collocate_one` /
/// collint wrap — evaluate at **unwrapped** r near the product center, wrap
/// the storage index into the primary cell (cuda-gpw-product-collocate-pbc).
/// Non-periodic: full-grid raw continuous formula (tests / foothold).
#include "product_window.cuh"

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

__device__ __forceinline__ double product_shape_raw(const DeviceProduct& q,
                                                    const double r[3]) {
    if (q.weight == 0.0)
        return 0.0;
    const double dx = r[0] - q.pc[0];
    const double dy = r[1] - q.pc[1];
    const double dz = r[2] - q.pc[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    double v = q.weight * exp(-q.p * r2);
    for (int d = 0; d < 3; ++d) {
        const double da = r[d] - q.ra[d];
        const double db = r[d] - q.rb[d];
        v *= ipow_nonneg(da, q.la[d]) * ipow_nonneg(db, q.lb[d]);
    }
    return v;
}

/// @brief Full-grid continuous (non-periodic / foothold).
__global__ void
collocate_products_fullgrid_kernel(const DeviceProduct* __restrict__ prods,
                                   int nprod, int n0, int n1, int n2, double L0,
                                   double L1, double L2, double o0, double o1,
                                   double o2, double* __restrict__ rho) {
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
    const double r[3] = {
        o0 + L0 * static_cast<double>(i) / static_cast<double>(n0),
        o1 + L1 * static_cast<double>(j) / static_cast<double>(n1),
        o2 + L2 * static_cast<double>(k) / static_cast<double>(n2)};
    double acc = 0.0;
    for (int q = 0; q < nprod; ++q)
        acc += product_shape_raw(prods[q], r);
    rho[idx] += acc;
}

struct DensityScatter {
    double* rho;
    double weight;

    __device__ void operator()(long i, double shape) {
        const double value = weight * shape;
        if (value != 0.0)
            atomicAdd(rho + i, value);
    }
};

__global__ void
collocate_products_sphere_kernel(const DeviceProduct* __restrict__ products,
                                 CollocateGridDesc grid,
                                 double* __restrict__ rho) {
    const DeviceProduct product = products[blockIdx.x];
    if (product.weight == 0.0)
        return;
    __shared__ ProductWindow window;
    __shared__ ProductAxisPoint cache[kProductAxisCacheSize];
    if (threadIdx.x == 0) {
        const double magnitude =
            product.mag > 0.0 ? product.mag : fabs(product.weight);
        window = product_window(grid, product, magnitude);
    }
    __syncthreads();
    if (!(window.radius2 > 0.0))
        return;
    DensityScatter scatter{rho, product.weight};
    visit_periodic_product(grid, product, window, cache, scatter);
}

__global__ void integrate_rho_kernel(const double* __restrict__ rho, long nr,
                                     double* __restrict__ acc) {
    __shared__ double s[256];
    const long i = static_cast<long>(blockIdx.x) * blockDim.x
                   + static_cast<long>(threadIdx.x);
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

    if (grid.periodic) {
        auto window_grid = grid;
        if (!(window_grid.eps_win > 0.0))
            window_grid.eps_win = 1e-14;
        collocate_products_sphere_kernel<<<nprod, kProductWindowBlock, 0,
                                           stream>>>(d_prods, window_grid,
                                                     d_rho);
    } else {
        constexpr int kBlock = 256;
        const int grid_dim =
            static_cast<int>((ntot + kBlock - 1) / static_cast<long>(kBlock));
        collocate_products_fullgrid_kernel<<<grid_dim, kBlock, 0, stream>>>(
            d_prods, nprod, grid.n[0], grid.n[1], grid.n[2], grid.L[0],
            grid.L[1], grid.L[2], grid.origin[0], grid.origin[1],
            grid.origin[2], d_rho);
    }
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
