/// @file cuda/longrange/pme.cu
/// @brief Smooth Particle-Mesh Ewald reciprocal sum — CUDA (cuFFT).
///        Built only with ATV_BUILD_KSPACE.
///
/// Compile-time spline order and bounded stencil-index wrapping follow the
/// OpenMM PME kernel architecture, platforms/common/src/kernels/pme.cc,
/// SHA 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e. The Essmann recurrence
/// and FFT normalization remain ATV-native; no source is copied. Spline and
/// grid precision follow work_t; energy accumulation remains FP64.
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"

namespace ATV {

using cuda::real_t;

namespace cuda {

// A power-of-two subwarp keeps every atom's interpolation collective aligned
// within a hardware warp, including runtime spline orders above eight.
constexpr int kInterpLanes = 8;

static_assert(std::is_same_v<work_t, float> || std::is_same_v<work_t, double>);
using PmeComplex = std::conditional_t<std::is_same_v<work_t, double>,
                                      cufftDoubleComplex, cufftComplex>;

// Geometry passed by value into kernels (cell⁻¹ row-major + grid dims).
struct PmeGeom {
    double inv_cell[9];
    int K[3];
    int order;
};

namespace kernels {

// Canonical Essmann/Amber fill_bspline on device. theta[k]=M_n(frac+n-1-k),
// dtheta[k]=dM/d(frac). order n ≤ 12.
template <int Order>
__device__ __forceinline__ void pme_bspline(work_t frac, int runtime_order,
                                            work_t* theta, work_t* dtheta) {
    const int n = Order > 0 ? Order : runtime_order;
    theta[n - 1] = work_t(0);
    theta[1] = frac;
    theta[0] = work_t(1) - frac;
    for (int k = 3; k < n; ++k) {
        const work_t div = work_t(1) / (k - 1);
        theta[k - 1] = div * frac * theta[k - 2];
        for (int j = 1; j < k - 1; ++j)
            theta[k - 1 - j] = div
                               * ((frac + j) * theta[k - 2 - j]
                                  + (k - j - frac) * theta[k - 1 - j]);
        theta[0] = div * (work_t(1) - frac) * theta[0];
    }
    dtheta[0] = -theta[0];
    for (int j = 1; j < n; ++j)
        dtheta[j] = theta[j - 1] - theta[j];
    const work_t div = work_t(1) / (n - 1);
    theta[n - 1] = div * frac * theta[n - 2];
    for (int j = 1; j < n - 1; ++j)
        theta[n - 1 - j] = div
                           * ((frac + j) * theta[n - 2 - j]
                              + (n - j - frac) * theta[n - 1 - j]);
    theta[0] = div * (work_t(1) - frac) * theta[0];
}

// Each grid dimension is at least order+1 and every stencil index lies in
// [-(order-1), K]. One adjustment suffices, including the rounded u==K edge.
__device__ __forceinline__ int wrap_idx(int v, int K) {
    if (v < 0)
        v += K;
    if (v >= K)
        v -= K;
    return v;
}

/// @brief Spread owned charges; order adjacent threads share one atom.
template <int Order>
__global__ void
pme_spread(const real_t* __restrict__ x, const real_t* __restrict__ y,
           const real_t* __restrict__ z, const real_t* __restrict__ q,
           int natoms, PmeGeom g, work_t* __restrict__ grid) {
    const int n = Order > 0 ? Order : g.order;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = gid / n, c = gid % n;
    if (i >= natoms)
        return;
    const work_t qi = work_t(q[i]);
    if (qi == 0.0)
        return;
    const double rx = double(x[i]), ry = double(y[i]), rz = double(z[i]);
    constexpr int extent = Order > 0 ? Order : 12;
    work_t th[3][extent], dth[3][extent];
    int base[3];
    for (int d = 0; d < 3; ++d) {
        const double s = g.inv_cell[0 * 3 + d] * rx + g.inv_cell[1 * 3 + d] * ry
                         + g.inv_cell[2 * 3 + d] * rz;
        double u = s * g.K[d];
        u -= floor(u / g.K[d]) * g.K[d];
        const int u0 = int(floor(u));
        pme_bspline<Order>(work_t(u - u0), n, th[d], dth[d]);
        base[d] = u0;
    }
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2], nm1 = n - 1;
    // As in OpenMM spreadCharge, adjacent threads write consecutive z
    // stencil sites. Each issues n*n atomics instead of n*n*n serial atomics.
    // Keep the existing (q*theta_x)*theta_y*theta_z multiplication order.
    const int g2 = wrap_idx(base[2] - nm1 + c, K2);
    // Select without dynamically indexing the order-specialized theta
    // array, so the compiler can retain the spline coefficients in registers.
    work_t wz = 0;
#pragma unroll
    for (int k = 0; k < n; ++k)
        if (k == c)
            wz = th[2][k];
    for (int a = 0; a < n; ++a) {
        const int g0 = wrap_idx(base[0] - nm1 + a, K0);
        const work_t wa = qi * th[0][a];
        for (int b = 0; b < n; ++b) {
            const int g1 = wrap_idx(base[1] - nm1 + b, K1);
            const work_t wab = wa * th[1][b];
            const int idx = g2 + K2 * (g1 + K1 * g0);
            atomicAdd(&grid[idx], wab * wz);
        }
    }
}

/// @brief Influence function: energy + in-place convolution multiply.
__global__ void pme_influence(PmeComplex* __restrict__ grid, PmeGeom g,
                              const double* __restrict__ influence,
                              double escale, double* __restrict__ energy) {
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2];
    const int half_z = K2 / 2 + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_energy = 0.0;
    if (idx < K0 * K1 * half_z) {
        if (idx == 0) {
            grid[idx].x = 0;
            grid[idx].y = 0;
        } else {
            const double H = influence[idx];
            const int m2 = idx % half_z;
            // The kz boundary planes already contain both xy-conjugate
            // indices. Only interior kz modes stand for two full modes.
            const double weight =
                (m2 == 0 || (K2 % 2 == 0 && m2 == K2 / 2)) ? 1.0 : 2.0;
            const double qr = grid[idx].x, qi = grid[idx].y;
            my_energy = weight * escale * H * (qr * qr + qi * qi);
            grid[idx].x = H * qr;
            grid[idx].y = H * qi;
        }
    }

    // Tail lanes and the zero mode contribute zero but must participate in
    // the collective. One atomic per block avoids serializing the whole mesh
    // on the shared energy scalar; keep the partial sums in PME's FP64 type.
    extern __shared__ double warp_scratch[];
    const double block_energy = block_reduce_sum(my_energy, warp_scratch);
    if (threadIdx.x == 0)
        atomicAdd(energy, static_cast<double>(block_energy));
}

/// @brief One lane or an eight-lane group interpolates each atom.
template <int Order, int Lanes>
__global__ void
pme_interp(real_t* __restrict__ fx, real_t* __restrict__ fy,
           real_t* __restrict__ fz, const real_t* __restrict__ x,
           const real_t* __restrict__ y, const real_t* __restrict__ z,
           const real_t* __restrict__ q, int natoms, PmeGeom g,
           const work_t* __restrict__ grid, double fscale, double self_pref,
           double* __restrict__ energy) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = gid / Lanes, lane = gid % Lanes;
    const double qi = (i < natoms) ? double(q[i]) : 0.0;
    // Only one lane owns self energy. All block threads participate before
    // the group-uniform neutral/tail exit, including the last partial block.
    extern __shared__ double warp_scratch[];
    const double block_energy =
        block_reduce_sum(lane == 0 ? self_pref * qi * qi : 0.0, warp_scratch);
    if (threadIdx.x == 0)
        atomicAdd(energy, static_cast<double>(block_energy));
    if (i >= natoms || qi == 0.0)
        return;
    const double rx = double(x[i]), ry = double(y[i]), rz = double(z[i]);
    const int n = Order > 0 ? Order : g.order;
    constexpr int extent = Order > 0 ? Order : 12;
    work_t th[3][extent], dth[3][extent];
    int base[3];
    for (int d = 0; d < 3; ++d) {
        const double s = g.inv_cell[0 * 3 + d] * rx + g.inv_cell[1 * 3 + d] * ry
                         + g.inv_cell[2 * 3 + d] * rz;
        double u = s * g.K[d];
        u -= floor(u / g.K[d]) * g.K[d];
        const int u0 = int(floor(u));
        pme_bspline<Order>(work_t(u - u0), n, th[d], dth[d]);
        base[d] = u0;
    }
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2], nm1 = n - 1;
    work_t gd[3] = {0, 0, 0};
    if constexpr (Lanes == 1) {
        for (int a = 0; a < n; ++a) {
            const int g0 = wrap_idx(base[0] - nm1 + a, K0);
            for (int b = 0; b < n; ++b) {
                const int g1 = wrap_idx(base[1] - nm1 + b, K1);
                for (int c = 0; c < n; ++c) {
                    const int g2 = wrap_idx(base[2] - nm1 + c, K2);
                    const work_t cval = grid[g2 + K2 * (g1 + K1 * g0)];
                    gd[0] += cval * dth[0][a] * th[1][b] * th[2][c];
                    gd[1] += cval * th[0][a] * dth[1][b] * th[2][c];
                    gd[2] += cval * th[0][a] * th[1][b] * dth[2][c];
                }
            }
        }
    } else {
        for (int c = lane; c < n; c += Lanes) {
            const int g2 = wrap_idx(base[2] - nm1 + c, K2);
            // Select z coefficients without dynamically indexing the specialized
            // spline arrays, so order-5/6 coefficients can remain in registers.
            work_t wz = 0, dwz = 0;
#pragma unroll
            for (int k = 0; k < n; ++k)
                if (k == c) {
                    wz = th[2][k];
                    dwz = dth[2][k];
                }
            for (int a = 0; a < n; ++a) {
                const int g0 = wrap_idx(base[0] - nm1 + a, K0);
                for (int b = 0; b < n; ++b) {
                    const int g1 = wrap_idx(base[1] - nm1 + b, K1);
                    const work_t cval = grid[g2 + K2 * (g1 + K1 * g0)];
                    gd[0] += cval * dth[0][a] * th[1][b] * wz;
                    gd[1] += cval * th[0][a] * dth[1][b] * wz;
                    gd[2] += cval * th[0][a] * th[1][b] * dwz;
                }
            }
        }
        // Every active atom owns all eight lanes, even when n < eight. Lanes
        // without z sites contribute zero; orders 9--12 visit a second column.
        const unsigned group_mask = 0xffu << (threadIdx.x & 24);
        for (int offset = Lanes / 2; offset > 0; offset /= 2)
            for (int d = 0; d < 3; ++d)
                gd[d] += __shfl_down_sync(group_mask, gd[d], offset, Lanes);
        if (lane != 0)
            return;
    }
    double fxv = 0, fyv = 0, fzv = 0;
    for (int d = 0; d < 3; ++d) {
        const double gk = double(gd[d]) * g.K[d];
        fxv += gk * g.inv_cell[0 * 3 + d];
        fyv += gk * g.inv_cell[1 * 3 + d];
        fzv += gk * g.inv_cell[2 * 3 + d];
    }
    fx[i] += real_t(-fscale * qi * fxv);
    fy[i] += real_t(-fscale * qi * fyv);
    fz[i] += real_t(-fscale * qi * fzv);
}

} // namespace kernels

namespace {

// The selected complex type, plan kind and execution API must agree in both
// build warmup and capture. Only the configured work_t path is instantiated.
template <typename Real, typename Complex>
cufftResult execute_forward(cufftHandle plan, Real* grid, Complex* spectrum) {
    if constexpr (std::is_same_v<Complex, cufftDoubleComplex>) {
        static_assert(std::is_same_v<Real, double>);
        return cufftExecD2Z(plan, grid, spectrum);
    } else {
        static_assert(std::is_same_v<Complex, cufftComplex>);
        static_assert(std::is_same_v<Real, float>);
        return cufftExecR2C(plan, grid, spectrum);
    }
}

template <typename Real, typename Complex>
cufftResult execute_inverse(cufftHandle plan, Complex* spectrum, Real* grid) {
    if constexpr (std::is_same_v<Complex, cufftDoubleComplex>) {
        static_assert(std::is_same_v<Real, double>);
        return cufftExecZ2D(plan, spectrum, grid);
    } else {
        static_assert(std::is_same_v<Complex, cufftComplex>);
        static_assert(std::is_same_v<Real, float>);
        return cufftExecC2R(plan, spectrum, grid);
    }
}

int good_fft_size(int n) {
    auto ok = [](int m) {
        for (int p : {2, 3, 5, 7})
            while (m % p == 0)
                m /= p;
        return m == 1;
    };
    if (n < 4)
        n = 4;
    while (!ok(n)) {
        if (n == std::numeric_limits<int>::max())
            throw std::overflow_error(
                "KSpacePME: FFT dimension exceeds int indexing");
        ++n;
    }
    return n;
}

std::vector<double> spline_moduli(int K, int n) {
    // M_n at integers via the host mirror of pme_bspline(frac=0).
    std::vector<double> data(n, 0.0), dd(n, 0.0);
    {
        double* theta = data.data();
        double* dtheta = dd.data();
        const double frac = 0.0;
        theta[n - 1] = 0.0;
        theta[1] = frac;
        theta[0] = 1.0 - frac;
        for (int k = 3; k < n; ++k) {
            const double div = 1.0 / (k - 1);
            theta[k - 1] = div * frac * theta[k - 2];
            for (int j = 1; j < k - 1; ++j)
                theta[k - 1 - j] = div
                                   * ((frac + j) * theta[k - 2 - j]
                                      + (k - j - frac) * theta[k - 1 - j]);
            theta[0] = div * (1.0 - frac) * theta[0];
        }
        dtheta[0] = -theta[0];
        for (int j = 1; j < n; ++j)
            dtheta[j] = theta[j - 1] - theta[j];
        const double div = 1.0 / (n - 1);
        theta[n - 1] = div * frac * theta[n - 2];
        for (int j = 1; j < n - 1; ++j)
            theta[n - 1 - j] = div
                               * ((frac + j) * theta[n - 2 - j]
                                  + (n - j - frac) * theta[n - 1 - j]);
        theta[0] = div * (1.0 - frac) * theta[0];
    }
    std::vector<double> mod(K, 0.0);
    for (int i = 0; i < K; ++i) {
        double sr = 0, si = 0;
        for (int k = 0; k < n - 1; ++k) {
            const double arg = 2.0 * ATV::units::pi * i * k / K;
            sr += data[k] * std::cos(arg);
            si += data[k] * std::sin(arg);
        }
        const double den = sr * sr + si * si;
        mod[i] = den > 1e-12 ? 1.0 / den : 0.0;
    }
    for (int i = 0; i < K; ++i)
        if (mod[i] == 0.0)
            mod[i] = 0.5 * (mod[(i - 1 + K) % K] + mod[(i + 1) % K]);
    return mod;
}

} // namespace

KSpacePME::KSpacePME(double alpha, double grid_spacing, int order)
    : alpha_(alpha), spacing_(grid_spacing), order_(order) {
    if (!std::isfinite(alpha) || alpha <= 0.0)
        throw std::invalid_argument("KSpacePME: alpha must be > 0");
    if (!std::isfinite(grid_spacing) || grid_spacing <= 0.0)
        throw std::invalid_argument("KSpacePME: grid spacing must be > 0");
    if (order < 3 || order > 12)
        throw std::invalid_argument("KSpacePME: order must be in [3,12]");
}

KSpacePME::~KSpacePME() {}

void KSpacePME::build(const BuildContext& ctx) {
    const long long stencil_threads =
        static_cast<long long>(ctx.frame.n_local())
        * (order_ > kInterpLanes ? order_ : kInterpLanes);
    if (stencil_threads > std::numeric_limits<int>::max() - (kDefaultBlock - 1))
        throw std::overflow_error(
            "KSpacePME: stencil thread count exceeds int indexing");
    const cuda::MotionBox& box = ctx.runtime.host_motion_box;
    for (int i = 0; i < 9; ++i)
        inv_cell_[i] = double(box.inv_cell[i]);
    for (int i = 0; i < 9; ++i)
        built_cell_[i] = double(ctx.runtime.host_motion_box.cell[i]);
    vol_ = double(box.volume());
    if (!std::isfinite(vol_) || vol_ <= 0)
        throw std::invalid_argument(
            "KSpacePME: finite positive periodic cell volume required");

    std::array<std::vector<double>, 3> moduli;
    for (int d = 0; d < 3; ++d) {
        const double len = double(box.length(d));
        const double requested = std::ceil(len / spacing_);
        if (!std::isfinite(requested) || requested <= 0
            || requested > std::numeric_limits<int>::max())
            throw std::overflow_error(
                "KSpacePME: FFT dimension outside int indexing");
        K_[d] = good_fft_size(int(requested));
        if (K_[d] < order_ + 1)
            K_[d] = good_fft_size(order_ + 1);
        moduli[d] = spline_moduli(K_[d], order_);
    }
    long long real_count = 1;
    const int grid_limit =
        std::numeric_limits<int>::max() - (kDefaultBlock - 1);
    for (int size : K_) {
        if (real_count > grid_limit / size)
            throw std::overflow_error(
                "KSpacePME: FFT grid exceeds int indexing");
        real_count *= size;
    }
    ntot_ = int(real_count);
    const int half_z = K_[2] / 2 + 1;
    nhalf_ = K_[0] * K_[1] * half_z;
    const double inv_pi2_a2 =
        ATV::units::pi * ATV::units::pi / (alpha_ * alpha_);
    auto coefficient = [&](int m0, int m1, int m2) {
        if (m0 == 0 && m1 == 0 && m2 == 0)
            return 0.0;
        const int m[3] = {m0, m1, m2};
        double mx = 0, my = 0, mz = 0;
        for (int d = 0; d < 3; ++d) {
            const int h = m[d] <= K_[d] / 2 ? m[d] : m[d] - K_[d];
            mx += h * inv_cell_[d];
            my += h * inv_cell_[3 + d];
            mz += h * inv_cell_[6 + d];
        }
        const double msq = mx * mx + my * my + mz * mz;
        const double B = moduli[0][m0] * moduli[1][m1] * moduli[2][m2];
        return std::exp(-inv_pi2_a2 * msq) / msq * B;
    };
    std::vector<double> influence(nhalf_);
    for (int m0 = 0; m0 < K_[0]; ++m0)
        for (int m1 = 0; m1 < K_[1]; ++m1)
            for (int m2 = 0; m2 < half_z; ++m2) {
                const int c0 = m0 == 0 ? 0 : K_[0] - m0;
                const int c1 = m1 == 0 ? 0 : K_[1] - m1;
                const int c2 = m2 == 0 ? 0 : K_[2] - m2;
                // The old full-complex inverse used only its real part.
                // For a real charge mesh that equals the inverse of Hsym*Q.
                // Averaging full-index conjugates preserves its energy too,
                // including triclinic cross terms at canonical +N/2 modes.
                influence[m2 + half_z * (m1 + K_[1] * m0)] =
                    0.5 * coefficient(m0, m1, m2)
                    + 0.5 * coefficient(c0, c1, c2);
            }
    ATV_CUDA_CHECK(
        cudaMalloc(&d_influence_, std::size_t(nhalf_) * sizeof(double)));
    ATV_CUDA_CHECK(cudaMemcpy(d_influence_, influence.data(),
                              std::size_t(nhalf_) * sizeof(double),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMalloc(&d_grid_, std::size_t(ntot_) * sizeof(work_t)));
    ATV_CUDA_CHECK(
        cudaMalloc(&d_spectrum_, std::size_t(nhalf_) * sizeof(PmeComplex)));
    cufftHandle forward, inverse;
    constexpr cufftType forward_type =
        std::is_same_v<PmeComplex, cufftDoubleComplex> ? CUFFT_D2Z : CUFFT_R2C;
    constexpr cufftType inverse_type =
        std::is_same_v<PmeComplex, cufftDoubleComplex> ? CUFFT_Z2D : CUFFT_C2R;
    if (cufftPlan3d(&forward, K_[0], K_[1], K_[2], forward_type)
        != CUFFT_SUCCESS)
        throw std::runtime_error("KSpacePME: forward cufftPlan3d failed");
    forward_plan_ = static_cast<int>(forward);
    if (cufftPlan3d(&inverse, K_[0], K_[1], K_[2], inverse_type)
        != CUFFT_SUCCESS)
        throw std::runtime_error("KSpacePME: inverse cufftPlan3d failed");
    inverse_plan_ = static_cast<int>(inverse);
    built_atoms_ = ctx.frame.n_local();
    if (built_atoms_ == 0)
        return;
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    if (cufftSetStream(forward, stream) != CUFFT_SUCCESS
        || cufftSetStream(inverse, stream) != CUFFT_SUCCESS)
        throw std::runtime_error("KSpacePME: cufftSetStream failed");
    // cuFFT supports graph capture without callbacks. Allocate/instantiate
    // once here: force evaluation replays the same resident arrays without
    // issuing each FFT stage and spline kernel separately from the host.
    // https://docs.nvidia.com/cuda/cufft/index.html#cuda-graphs-support
    ATV_CUDA_CHECK(cudaMemsetAsync(
        d_grid_, 0, std::size_t(ntot_) * sizeof(work_t), stream));
    auto* warm_grid = static_cast<work_t*>(d_grid_);
    auto* warm_spectrum = static_cast<PmeComplex*>(d_spectrum_);
    if (execute_forward(forward, warm_grid, warm_spectrum) != CUFFT_SUCCESS
        || execute_inverse(inverse, warm_spectrum, warm_grid) != CUFFT_SUCCESS)
        throw std::runtime_error("KSpacePME: FFT build warmup failed");
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph = nullptr;
    ATV_CUDA_CHECK(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
        enqueue_(LaunchContext{ctx.runtime, ctx.frame, 0, 0});
    } catch (...) {
        const auto capture_status = cudaStreamEndCapture(stream, &graph);
        if (capture_status != cudaSuccess
            && capture_status != cudaErrorStreamCaptureInvalidated)
            std::fprintf(stderr, "KSpacePME: capture cleanup failed: %s\n",
                         cudaGetErrorString(capture_status));
        if (graph) {
            const auto destroy_status = cudaGraphDestroy(graph);
            if (destroy_status != cudaSuccess)
                std::fprintf(stderr, "KSpacePME: graph cleanup failed: %s\n",
                             cudaGetErrorString(destroy_status));
        }
        throw;
    }
    ATV_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t executable = nullptr;
    const auto status = cudaGraphInstantiate(&executable, graph, 0);
    const auto destroy_status = cudaGraphDestroy(graph);
    ATV_CUDA_CHECK(status);
    ATV_CUDA_CHECK(destroy_status);
    graph_exec_ = executable;
}

void KSpacePME::compute(const LaunchContext& ctx) {
    if (ctx.frame.n_local() != built_atoms_)
        throw std::runtime_error(
            "KSpacePME: atom set changed; rebuild the driver");
    for (int i = 0; i < 9; ++i)
        if (double(ctx.runtime.host_motion_box.cell[i]) != built_cell_[i])
            throw std::runtime_error("KSpacePME: fixed-cell reciprocal solver; "
                                     "changing-cell MD is unsupported");
    if (built_atoms_)
        ATV_CUDA_CHECK(
            cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_exec_),
                            static_cast<cudaStream_t>(ctx.runtime.stream)));
}

void KSpacePME::enqueue_(const LaunchContext& ctx) {
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = built_atoms_;
    auto* grid = static_cast<work_t*>(d_grid_);
    auto* spectrum = static_cast<PmeComplex*>(d_spectrum_);
    const cufftHandle forward = static_cast<cufftHandle>(forward_plan_);
    const cufftHandle inverse = static_cast<cufftHandle>(inverse_plan_);

    PmeGeom geom;
    for (int i = 0; i < 9; ++i)
        geom.inv_cell[i] = inv_cell_[i];
    geom.K[0] = K_[0];
    geom.K[1] = K_[1];
    geom.K[2] = K_[2];
    geom.order = order_;

    const double escale = ATV::units::coulomb / (2.0 * ATV::units::pi * vol_);
    const double fscale = ATV::units::coulomb / (ATV::units::pi * vol_);
    const double self_pref =
        -ATV::units::coulomb * alpha_ / std::sqrt(ATV::units::pi);
    const int blk = kDefaultBlock;
    const int energy_scratch_bytes = warps_per_block(blk) * sizeof(double);

    ATV_CUDA_CHECK(
        cudaMemsetAsync(grid, 0, std::size_t(ntot_) * sizeof(work_t), stream));
    auto spread = [&]<int Order>() {
        kernels::pme_spread<Order><<<grid_size(n * order_), blk, 0, stream>>>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.charge, n,
            geom, grid);
        ATV_CUDA_CHECK(cudaGetLastError());
    };
    if (order_ == 5)
        spread.template operator()<5>();
    else if (order_ == 6)
        spread.template operator()<6>();
    else
        spread.template operator()<0>();
    if (execute_forward(forward, grid, spectrum) != CUFFT_SUCCESS)
        throw std::runtime_error("KSpacePME: forward FFT failed");
    kernels::
        pme_influence<<<grid_size(nhalf_), blk, energy_scratch_bytes, stream>>>(
            spectrum, geom, d_influence_, escale, &frame.scalars->energy);
    ATV_CUDA_CHECK(cudaGetLastError());
    if (execute_inverse(inverse, spectrum, grid) != CUFFT_SUCCESS)
        throw std::runtime_error("KSpacePME: inverse FFT failed");
    auto interpolate = [&]<int Order, int Lanes>() {
        kernels::pme_interp<Order, Lanes>
            <<<grid_size(n * Lanes), blk, energy_scratch_bytes, stream>>>(
                frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.atoms.x,
                frame.atoms.y, frame.atoms.z, frame.atoms.charge, n, geom, grid,
                fscale, self_pref, &frame.scalars->energy);
        ATV_CUDA_CHECK(cudaGetLastError());
    };
    auto interpolate_order = [&]<int Lanes>() {
        if (order_ == 5)
            interpolate.template operator()<5, Lanes>();
        else if (order_ == 6)
            interpolate.template operator()<6, Lanes>();
        else
            interpolate.template operator()<0, Lanes>();
    };
    // Subwarps expose more parallelism for small systems. At larger atom
    // counts one lane per atom avoids duplicating spline work and improves
    // occupancy. The crossover is conservative for the measured GH200 path.
    if (n <= 16384)
        interpolate_order.template operator()<kInterpLanes>();
    else
        interpolate_order.template operator()<1>();
    ATV_CUDA_CHECK(cudaGetLastError());
}

void KSpacePME::destroy(const DestroyContext& ctx) {
    ATV_CUDA_CHECK(
        cudaStreamSynchronize(static_cast<cudaStream_t>(ctx.runtime.stream)));
    if (graph_exec_) {
        ATV_CUDA_CHECK(
            cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_)));
        graph_exec_ = nullptr;
    }
    built_atoms_ = 0;
    if (forward_plan_ >= 0) {
        cufftDestroy(static_cast<cufftHandle>(forward_plan_));
        forward_plan_ = -1;
    }
    if (inverse_plan_ >= 0) {
        cufftDestroy(static_cast<cufftHandle>(inverse_plan_));
        inverse_plan_ = -1;
    }
    ATV_CUDA_CHECK(cudaFree(d_influence_));
    d_influence_ = nullptr;
    ATV_CUDA_CHECK(cudaFree(d_spectrum_));
    d_spectrum_ = nullptr;
    ATV_CUDA_CHECK(cudaFree(d_grid_));
    d_grid_ = nullptr;
}

} // namespace cuda
} // namespace ATV
