/// @file cuda/gpw/internal/product_integrate.cu
/// @brief Device product-Gaussian integrate for GPW V→H path.
///
/// Periodic: sphere-window integrate twin of collocate (unwrapped r, wrap
/// index — cuda-gpw-product-collocate-pbc). Non-periodic: full-grid bare shape.
#include "atomiverse/cuda/gpw/internal/product_integrate.cuh"

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

__device__ __forceinline__ double product_shape_bare_raw(const DeviceProduct& q,
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

__device__ __forceinline__ long wrap_idx(long i, int n) {
    long m = i % n;
    if (m < 0)
        m += n;
    return m;
}

/// Full-grid integrate (non-periodic).
__global__ void integrate_products_fullgrid_kernel(
    const DeviceProduct* __restrict__ prods, int nprod, int n0, int n1, int n2,
    double L0, double L1, double L2, double o0, double o1, double o2,
    const double* __restrict__ V, double* __restrict__ acc) {
    const int q = static_cast<int>(blockIdx.x);
    if (q >= nprod)
        return;
    const long ntot =
        static_cast<long>(n0) * static_cast<long>(n1) * static_cast<long>(n2);
    const DeviceProduct prod = prods[q];
    double local = 0.0;
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
        local += V[idx] * product_shape_bare_raw(prod, r);
    }
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

/// Sphere-window PBC integrate: one block per product, threads stride the window.
__global__ void integrate_products_sphere_kernel(
    const DeviceProduct* __restrict__ prods, int nprod, int n0, int n1, int n2,
    double L0, double L1, double L2, double o0, double o1, double o2,
    double eps, const double* __restrict__ V, double* __restrict__ acc) {
    const int ip = static_cast<int>(blockIdx.x);
    if (ip >= nprod)
        return;
    const DeviceProduct q = prods[ip];
    // Integrate is shape-only; radius from collint-class window mag (group
    // max|w| when set) floored at 1 so support covers the significant Gaussian
    // (weight applied only on collocate / scatter k).
    const int Ltot =
        q.la[0] + q.la[1] + q.la[2] + q.lb[0] + q.lb[1] + q.lb[2];
    const double mag = (q.mag > 0.0) ? fmax(q.mag, 1.0) : 1.0;
    const double rc = product_radius_dev(mag, q.p, Ltot, eps);
    if (!(rc > 0.0)) {
        if (threadIdx.x == 0)
            acc[ip] = 0.0;
        return;
    }

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
    const long nwin0 = hi0 - lo0 + 1;
    const long nwin1 = hi1 - lo1 + 1;
    const long nwin2 = hi2 - lo2 + 1;
    const long nwin = nwin0 * nwin1 * nwin2;
    const double rc2 = rc * rc;

    double local = 0.0;
    for (long t = static_cast<long>(threadIdx.x); t < nwin;
         t += static_cast<long>(blockDim.x)) {
        const long koff = t % nwin2;
        const long tmp = t / nwin2;
        const long joff = tmp % nwin1;
        const long ioff = tmp / nwin1;
        const long i = lo0 + ioff;
        const long j = lo1 + joff;
        const long k = lo2 + koff;
        const double r0 =
            o0 + L0 * static_cast<double>(i) / static_cast<double>(n0);
        const double r1 =
            o1 + L1 * static_cast<double>(j) / static_cast<double>(n1);
        const double r2c =
            o2 + L2 * static_cast<double>(k) / static_cast<double>(n2);
        const double d0 = r0 - q.pc[0];
        const double d1 = r1 - q.pc[1];
        const double d2 = r2c - q.pc[2];
        if (d0 * d0 + d1 * d1 + d2 * d2 > rc2)
            continue;
        const double r[3] = {r0, r1, r2c};
        const long ii = wrap_idx(i, n0);
        const long jj = wrap_idx(j, n1);
        const long kk = wrap_idx(k, n2);
        const long gidx =
            (ii * static_cast<long>(n1) + jj) * static_cast<long>(n2) + kk;
        local += V[gidx] * product_shape_bare_raw(q, r);
    }

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
        acc[ip] = s[0];
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
    constexpr int kBlock = 256;
    if (grid.periodic) {
        // Collint eps_win class (cuda-gpw-multigrid-device-ks) — not mass-margin.
        const double eps =
            (grid.eps_win > 0.0) ? grid.eps_win : 2.0e-9;
        integrate_products_sphere_kernel<<<nprod, kBlock, 0, stream>>>(
            d_prods, nprod, grid.n[0], grid.n[1], grid.n[2], grid.L[0],
            grid.L[1], grid.L[2], grid.origin[0], grid.origin[1],
            grid.origin[2], eps, d_V, d_acc);
    } else {
        integrate_products_fullgrid_kernel<<<nprod, kBlock, 0, stream>>>(
            d_prods, nprod, grid.n[0], grid.n[1], grid.n[2], grid.L[0],
            grid.L[1], grid.L[2], grid.origin[0], grid.origin[1],
            grid.origin[2], d_V, d_acc);
    }
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
