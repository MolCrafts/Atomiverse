/// @file cuda/gpw/internal/product_collocate.cu
/// @brief Device product-Gaussian collocation for GPW density.
///
/// Periodic path: sphere-window collocate matching CPU `collocate_one` /
/// collint wrap — evaluate at **unwrapped** r near the product center, wrap
/// the storage index into the primary cell (cuda-gpw-product-collocate-pbc).
/// Non-periodic: full-grid raw continuous formula (tests / foothold).
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

__device__ __forceinline__ double ipow_nonneg(double x, int n) {
    double v = 1.0;
    for (int i = 0; i < n; ++i)
        v *= x;
    return v;
}

/// @brief Radius [Bohr] where |mag| r^L exp(-p r²) drops below eps.
/// Twin of CPU `product_radius` / collint `shape_radius`.
__device__ __forceinline__ double product_radius_dev(double mag, double p,
                                                    int Ltot, double eps) {
    if (!(mag > 0.0) || !(p > 0.0) || !(eps > 0.0))
        return 0.0;
    const double lc = log(fmax(mag, 1e-300));
    const double leps = log(fmax(eps, 1e-300));
    double r2 = fmax(0.0, (lc - leps) / p);
    for (int it = 0; it < 4; ++it) {
        const double r = sqrt(fmax(r2, 1e-30));
        r2 = fmax(0.0, (lc + static_cast<double>(Ltot) * log(r) - leps) / p);
    }
    return sqrt(fmax(r2, 0.0));
}

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

__device__ __forceinline__ long wrap_idx(long i, int n) {
    long m = i % n;
    if (m < 0)
        m += n;
    return m;
}

/// @brief Full-grid continuous (non-periodic / foothold).
__global__ void collocate_products_fullgrid_kernel(
    const DeviceProduct* __restrict__ prods, int nprod, int n0, int n1, int n2,
    double L0, double L1, double L2, double o0, double o1, double o2,
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
        acc += product_shape_raw(prods[q], r);
    rho[idx] += acc;
}

/// @brief Sphere-window PBC collocate: one thread per product.
///
/// Unwrapped evaluation + wrap storage (CPU collocate_one).
/// Launch: grid-stride over products, block=128 (choose-launch-configuration —
/// product-parallel; many zero-weight early-outs). atomicAdd on rho
/// (optimize-global-memory-access tradeoff for sparse windows).
/// Radius uses collint group `mag` (fallback \|weight\|) with eps_win — not
/// mass-margin eps (cuda-gpw-multigrid-device-ks).
__global__ void collocate_products_sphere_kernel(
    const DeviceProduct* __restrict__ prods, int nprod, int n0, int n1, int n2,
    double L0, double L1, double L2, double o0, double o1, double o2,
    double eps, double* __restrict__ rho) {
    // Grid-stride over products (handle-boundary-conditions for partial tiles).
    for (int ip = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         ip < nprod;
         ip += static_cast<int>(blockDim.x * gridDim.x)) {
        const DeviceProduct q = prods[ip];
        if (q.weight == 0.0)
            continue;
        const int Ltot = q.la[0] + q.la[1] + q.la[2] + q.lb[0] + q.lb[1]
                         + q.lb[2];
        // Collint twin: window mag is group max|w|; value still uses weight.
        const double win_mag = (q.mag > 0.0) ? q.mag : fabs(q.weight);
        const double rc = product_radius_dev(win_mag, q.p, Ltot, eps);
        if (!(rc > 0.0))
            continue;

        const double dx0 = L0 / static_cast<double>(n0);
        const double dx1 = L1 / static_cast<double>(n1);
        const double dx2 = L2 / static_cast<double>(n2);
        const long ih0 = static_cast<long>(ceil(rc / dx0));
        const long ih1 = static_cast<long>(ceil(rc / dx1));
        const long ih2 = static_cast<long>(ceil(rc / dx2));
        const long ic0 = static_cast<long>(floor(q.pc[0] / dx0));
        const long ic1 = static_cast<long>(floor(q.pc[1] / dx1));
        const long ic2 = static_cast<long>(floor(q.pc[2] / dx2));
        const long lo0 = ic0 - ih0, hi0 = ic0 + ih0;
        const long lo1 = ic1 - ih1, hi1 = ic1 + ih1;
        const long lo2 = ic2 - ih2, hi2 = ic2 + ih2;
        const double rc2 = rc * rc;

        for (long i = lo0; i <= hi0; ++i) {
            const long ii = wrap_idx(i, n0);
            const double r0 =
                o0 + L0 * static_cast<double>(i) / static_cast<double>(n0);
            const double d0 = r0 - q.pc[0];
            for (long j = lo1; j <= hi1; ++j) {
                const long jj = wrap_idx(j, n1);
                const double r1 =
                    o1 + L1 * static_cast<double>(j) / static_cast<double>(n1);
                const double d1 = r1 - q.pc[1];
                const double d01 = d0 * d0 + d1 * d1;
                if (d01 > rc2)
                    continue;
                const long base =
                    (ii * static_cast<long>(n1) + jj) * static_cast<long>(n2);
                for (long k = lo2; k <= hi2; ++k) {
                    const double r2c =
                        o2
                        + L2 * static_cast<double>(k) / static_cast<double>(n2);
                    const double d2 = r2c - q.pc[2];
                    if (d01 + d2 * d2 > rc2)
                        continue;
                    const double r[3] = {r0, r1, r2c};
                    const double val = product_shape_raw(q, r);
                    if (val == 0.0)
                        continue;
                    const long kk = wrap_idx(k, n2);
                    atomicAdd(rho + base + kk, val);
                }
            }
        }
    }
}

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

    if (grid.periodic) {
        // Sphere-window PBC (cuda-gpw-product-collocate-pbc).
        // eps_win: CPU collint kProductWindowEps class (not mass-margin) so
        // multi-level residual matches collint (cuda-gpw-multigrid-device-ks).
        const double eps =
            (grid.eps_win > 0.0) ? grid.eps_win : 2.0e-9;
        constexpr int kBlock = 128;
        // Occupy SMs: ~min( nprod/block, 256 ) blocks (choose-launch-configuration).
        int grid_dim = (nprod + kBlock - 1) / kBlock;
        if (grid_dim > 256)
            grid_dim = 256;
        if (grid_dim < 1)
            grid_dim = 1;
        collocate_products_sphere_kernel<<<grid_dim, kBlock, 0, stream>>>(
            d_prods, nprod, grid.n[0], grid.n[1], grid.n[2], grid.L[0],
            grid.L[1], grid.L[2], grid.origin[0], grid.origin[1],
            grid.origin[2], eps, d_rho);
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
