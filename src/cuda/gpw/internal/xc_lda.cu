/// @file cuda/gpw/internal/xc_lda.cu
/// @brief Device port of GTH Padé LDA used by cpu::gpw::xc_grid_pade /
///        cpu::xc::pade_lsd. Constants and formulas are transcribed verbatim
///        from src/cpu/xc/pade.cpp so device results match the CPU oracle to
///        ~1e-12. All scientific loops are device kernels — no host loops over
///        grid points.
#include "atomiverse/cuda/gpw/internal/xc_lda.cuh"

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

// ── Padé coefficients (verbatim from pade.cpp / CP2K xc_pade.F) ──────────
// Paramagnetic set.
__device__ constexpr double a0 = 0.4581652932831429e+0;
__device__ constexpr double a1 = 0.2217058676663745e+1;
__device__ constexpr double a2 = 0.7405551735357053e+0;
__device__ constexpr double a3 = 0.1968227878617998e-1;
__device__ constexpr double b1 = 1.0000000000000000e+0;
__device__ constexpr double b2 = 0.4504130959426697e+1;
__device__ constexpr double b3 = 0.1110667363742916e+1;
__device__ constexpr double b4 = 0.2359291751427506e-1;

// Spin-interpolation increments (ferro − para).
__device__ constexpr double da0 = 0.119086804055547e+0;
__device__ constexpr double da1 = 0.6157402568883345e+0;
__device__ constexpr double da2 = 0.1574201515892867e+0;
__device__ constexpr double da3 = 0.3532336663397157e-2;
__device__ constexpr double db1 = 0.0000000000000000e+0;
__device__ constexpr double db2 = 0.2673612973836267e+0;
__device__ constexpr double db3 = 0.2052004607777787e+0;
__device__ constexpr double db4 = 0.4200005045691381e-2;

__device__ constexpr double f13 = 1.0 / 3.0;
__device__ constexpr double f43 = 4.0 / 3.0;
__device__ constexpr double kRhoMin = 1.0e-12;
__device__ constexpr double kPi = 3.14159265358979323846;

/// @brief Spin-interpolation f(x), df/dx (CP2K calc_fx / cpu pade spin_function).
__device__ __forceinline__ void spin_function(double x, double& f, double& df) {
    const double fxfac = 1.0 / (pow(2.0, f43) - 2.0);
    if (x <= -1.0) {
        f = 1.0;
        df = -f43 * fxfac * pow(2.0, f13);
    } else if (x >= 1.0) {
        f = 1.0;
        df = f43 * fxfac * pow(2.0, f13);
    } else {
        const double xp = pow(1.0 + x, f43) + pow(1.0 - x, f43) - 2.0;
        f = xp * fxfac;
        df = (pow(1.0 + x, f13) - pow(1.0 - x, f13)) * fxfac * f43;
    }
}

/// @brief One grid point: Padé LSD matching cpu::xc::pade_lsd.
__device__ __forceinline__ void pade_lsd_point(double rho_a, double rho_b,
                                              double& exc, double& vrho_a,
                                              double& vrho_b) {
    const double rhoab = rho_a + rho_b;
    if (rhoab < kRhoMin) {
        exc = 0.0;
        vrho_a = 0.0;
        vrho_b = 0.0;
        return;
    }

    const double rsfac = cbrt(3.0 / (4.0 * kPi));
    const double rs = rsfac * pow(rhoab, -f13);

    const double x = (rho_a - rho_b) / rhoab;
    double fx1, fx2;
    spin_function(x, fx1, fx2);

    const double fa0 = a0 + fx1 * da0;
    const double fa1 = a1 + fx1 * da1;
    const double fa2 = a2 + fx1 * da2;
    const double fa3 = a3 + fx1 * da3;
    const double fb1 = b1 + fx1 * db1;
    const double fb2 = b2 + fx1 * db2;
    const double fb3 = b3 + fx1 * db3;
    const double fb4 = b4 + fx1 * db4;

    const double p = fa0 + (fa1 + (fa2 + fa3 * rs) * rs) * rs;
    const double q = (fb1 + (fb2 + (fb3 + fb4 * rs) * rs) * rs) * rs;
    const double dpv = fa1 + (2.0 * fa2 + 3.0 * fa3 * rs) * rs;
    const double dq =
        fb1 + (2.0 * fb2 + (3.0 * fb3 + 4.0 * fb4 * rs) * rs) * rs;
    const double xp = da0 + (da1 + (da2 + da3 * rs) * rs) * rs;
    const double xq = (db1 + (db2 + (db3 + db4 * rs) * rs) * rs) * rs;

    const double dr = (dpv * q - p * dq) / (q * q);
    const double dx = 2.0 * (xp * q - p * xq) / (q * q) * fx2 / rhoab;
    const double dc = f13 * rs * dr - p / q;

    exc = -p / q;
    vrho_a = dc - dx * rho_b;
    vrho_b = dc + dx * rho_a;
}

__global__ void xc_lda_kernel(const double* __restrict__ rho_r,
                              double* __restrict__ vxc_r,
                              double* __restrict__ exc_r, int nr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr)
        return;
    const double rho = rho_r[i];
    double exc, va, vb;
    pade_lsd_point(0.5 * rho, 0.5 * rho, exc, va, vb);
    vxc_r[i] = va;
    exc_r[i] = exc;
}

__global__ void xc_lda_lsd_kernel(const double* __restrict__ rho_a,
                                  const double* __restrict__ rho_b,
                                  double* __restrict__ vxc_a,
                                  double* __restrict__ vxc_b,
                                  double* __restrict__ exc_r, int nr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr)
        return;
    double exc, va, vb;
    pade_lsd_point(rho_a[i], rho_b[i], exc, va, vb);
    vxc_a[i] = va;
    vxc_b[i] = vb;
    exc_r[i] = exc;
}

// Block-reduce Σ ρ·exc (unpolarized), atomic-accumulate into *acc.
__global__ void xc_energy_kernel(const double* __restrict__ rho_r,
                                 const double* __restrict__ exc_r, int nr,
                                 double* __restrict__ acc) {
    __shared__ double s[256];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (i < nr) ? rho_r[i] * exc_r[i] : 0.0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(acc, s[0]);
}

// Block-reduce Σ (ρ_a+ρ_b)·exc (spin-resolved).
__global__ void xc_energy_lsd_kernel(const double* __restrict__ rho_a,
                                     const double* __restrict__ rho_b,
                                     const double* __restrict__ exc_r, int nr,
                                     double* __restrict__ acc) {
    __shared__ double s[256];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] =
        (i < nr) ? (rho_a[i] + rho_b[i]) * exc_r[i] : 0.0;
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

void xc_lda_grid(const double* d_rho_r, double* d_vxc_r, double* d_exc_r,
                 int nr, cudaStream_t stream) {
    if (nr <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid = (nr + kBlock - 1) / kBlock;
    xc_lda_kernel<<<grid, kBlock, 0, stream>>>(d_rho_r, d_vxc_r, d_exc_r, nr);
}

void xc_lda_grid_lsd(const double* d_rho_a, const double* d_rho_b,
                     double* d_vxc_a, double* d_vxc_b, double* d_exc_r,
                     int nr, cudaStream_t stream) {
    if (nr <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid = (nr + kBlock - 1) / kBlock;
    xc_lda_lsd_kernel<<<grid, kBlock, 0, stream>>>(d_rho_a, d_rho_b, d_vxc_a,
                                                   d_vxc_b, d_exc_r, nr);
}

void xc_energy(const double* d_rho_r, const double* d_exc_r, int nr, double w,
               double* d_Exc, cudaStream_t stream) {
    cudaMemsetAsync(d_Exc, 0, sizeof(double), stream);
    if (nr <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid = (nr + kBlock - 1) / kBlock;
    xc_energy_kernel<<<grid, kBlock, 0, stream>>>(d_rho_r, d_exc_r, nr, d_Exc);
    scale_scalar_kernel<<<1, 1, 0, stream>>>(d_Exc, w);
}

void xc_energy_lsd(const double* d_rho_a, const double* d_rho_b,
                   const double* d_exc_r, int nr, double w, double* d_Exc,
                   cudaStream_t stream) {
    cudaMemsetAsync(d_Exc, 0, sizeof(double), stream);
    if (nr <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid = (nr + kBlock - 1) / kBlock;
    xc_energy_lsd_kernel<<<grid, kBlock, 0, stream>>>(d_rho_a, d_rho_b, d_exc_r,
                                                      nr, d_Exc);
    scale_scalar_kernel<<<1, 1, 0, stream>>>(d_Exc, w);
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
