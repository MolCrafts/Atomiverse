/// @file cuda/longrange/pme.cu
/// @brief Smooth Particle-Mesh Ewald reciprocal sum — CUDA (cuFFT).
///        Built only with ATV_BUILD_KSPACE.
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"

namespace ATV {

using cuda::prec_t;
using cuda::real_t;

namespace cuda {

// Geometry passed by value into kernels (cell⁻¹ row-major + grid dims).
struct PmeGeom {
    double inv_cell[9];
    int K[3];
    int order;
};

namespace kernels {

// Canonical Essmann/Amber fill_bspline on device. theta[k]=M_n(frac+n-1-k),
// dtheta[k]=dM/d(frac). order n ≤ 12.
__device__ inline void pme_bspline(double frac, int n, double* theta,
                                   double* dtheta) {
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

__device__ inline int wrap_idx(int v, int K) { return ((v % K) + K) % K; }

/// @brief Spread owned charges onto the complex grid (atomicAdd to real part).
__global__ void pme_spread(const real_t* __restrict__ x,
                           const real_t* __restrict__ y,
                           const real_t* __restrict__ z,
                           const real_t* __restrict__ q, int natoms, PmeGeom g,
                           cufftDoubleComplex* __restrict__ grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms)
        return;
    const double qi = double(q[i]);
    if (qi == 0.0)
        return;
    const double rx = double(x[i]), ry = double(y[i]), rz = double(z[i]);
    const int n = g.order;
    double th[3][12], dth[3][12];
    int base[3];
    for (int d = 0; d < 3; ++d) {
        const double s = g.inv_cell[0 * 3 + d] * rx + g.inv_cell[1 * 3 + d] * ry
                         + g.inv_cell[2 * 3 + d] * rz;
        double u = s * g.K[d];
        u -= floor(u / g.K[d]) * g.K[d];
        const int u0 = int(floor(u));
        pme_bspline(u - u0, n, th[d], dth[d]);
        base[d] = u0;
    }
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2], nm1 = n - 1;
    for (int a = 0; a < n; ++a) {
        const int g0 = wrap_idx(base[0] - nm1 + a, K0);
        const double wa = qi * th[0][a];
        for (int b = 0; b < n; ++b) {
            const int g1 = wrap_idx(base[1] - nm1 + b, K1);
            const double wab = wa * th[1][b];
            for (int c = 0; c < n; ++c) {
                const int g2 = wrap_idx(base[2] - nm1 + c, K2);
                const int idx = g2 + K2 * (g1 + K1 * g0);
                atomicAdd(&grid[idx].x, wab * th[2][c]);
            }
        }
    }
}

/// @brief Influence function: energy + in-place convolution multiply.
__global__ void pme_influence(cufftDoubleComplex* __restrict__ grid, PmeGeom g,
                              const double* __restrict__ bsp0,
                              const double* __restrict__ bsp1,
                              const double* __restrict__ bsp2,
                              double inv_pi2_a2, double escale,
                              prec_t* __restrict__ energy) {
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K0 * K1 * K2)
        return;
    const int m2 = idx % K2;
    const int m1 = (idx / K2) % K1;
    const int m0 = idx / (K2 * K1);
    if (m0 == 0 && m1 == 0 && m2 == 0) {
        grid[idx].x = 0;
        grid[idx].y = 0;
        return;
    }
    const int h0 = (m0 <= K0 / 2) ? m0 : m0 - K0;
    const int h1 = (m1 <= K1 / 2) ? m1 : m1 - K1;
    const int h2 = (m2 <= K2 / 2) ? m2 : m2 - K2;
    const int hh[3] = {h0, h1, h2};
    double mx = 0, my = 0, mz = 0;
    for (int d = 0; d < 3; ++d) {
        mx += hh[d] * g.inv_cell[0 * 3 + d];
        my += hh[d] * g.inv_cell[1 * 3 + d];
        mz += hh[d] * g.inv_cell[2 * 3 + d];
    }
    const double msq = mx * mx + my * my + mz * mz;
    const double B = bsp0[m0] * bsp1[m1] * bsp2[m2];
    const double H = exp(-inv_pi2_a2 * msq) / msq * B;
    const double qr = grid[idx].x, qi = grid[idx].y;
    atomicAdd(energy, static_cast<prec_t>(escale * H * (qr * qr + qi * qi)));
    grid[idx].x = H * qr;
    grid[idx].y = H * qi;
}

/// @brief Interpolate reciprocal forces onto atoms + per-atom self energy.
__global__ void
pme_interp(real_t* __restrict__ fx, real_t* __restrict__ fy,
           real_t* __restrict__ fz, const real_t* __restrict__ x,
           const real_t* __restrict__ y, const real_t* __restrict__ z,
           const real_t* __restrict__ q, int natoms, PmeGeom g,
           const cufftDoubleComplex* __restrict__ grid, double fscale,
           double self_pref, prec_t* __restrict__ energy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms)
        return;
    const double qi = double(q[i]);
    atomicAdd(energy, static_cast<prec_t>(self_pref * qi * qi));
    if (qi == 0.0)
        return;
    const double rx = double(x[i]), ry = double(y[i]), rz = double(z[i]);
    const int n = g.order;
    double th[3][12], dth[3][12];
    int base[3];
    for (int d = 0; d < 3; ++d) {
        const double s = g.inv_cell[0 * 3 + d] * rx + g.inv_cell[1 * 3 + d] * ry
                         + g.inv_cell[2 * 3 + d] * rz;
        double u = s * g.K[d];
        u -= floor(u / g.K[d]) * g.K[d];
        const int u0 = int(floor(u));
        pme_bspline(u - u0, n, th[d], dth[d]);
        base[d] = u0;
    }
    const int K0 = g.K[0], K1 = g.K[1], K2 = g.K[2], nm1 = n - 1;
    double gd[3] = {0, 0, 0};
    for (int a = 0; a < n; ++a) {
        const int g0 = wrap_idx(base[0] - nm1 + a, K0);
        for (int b = 0; b < n; ++b) {
            const int g1 = wrap_idx(base[1] - nm1 + b, K1);
            for (int c = 0; c < n; ++c) {
                const int g2 = wrap_idx(base[2] - nm1 + c, K2);
                const double cval = grid[g2 + K2 * (g1 + K1 * g0)].x;
                gd[0] += cval * dth[0][a] * th[1][b] * th[2][c];
                gd[1] += cval * th[0][a] * dth[1][b] * th[2][c];
                gd[2] += cval * th[0][a] * th[1][b] * dth[2][c];
            }
        }
    }
    double fxv = 0, fyv = 0, fzv = 0;
    for (int d = 0; d < 3; ++d) {
        const double gk = gd[d] * g.K[d];
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

int good_fft_size(int n) {
    auto ok = [](int m) {
        for (int p : {2, 3, 5, 7})
            while (m % p == 0)
                m /= p;
        return m == 1;
    };
    if (n < 4)
        n = 4;
    while (!ok(n))
        ++n;
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
    if (alpha <= 0.0)
        throw std::invalid_argument("KSpacePME: alpha must be > 0");
    if (order < 3 || order > 12)
        throw std::invalid_argument("KSpacePME: order must be in [3,12]");
}

KSpacePME::~KSpacePME() {}

void KSpacePME::build(const BuildContext& ctx) {
    const cuda::Box& box = ctx.runtime.host_box;
    for (int i = 0; i < 9; ++i)
        inv_cell_[i] = double(box.inv_cell[i]);
    vol_ = double(box.volume());

    for (int d = 0; d < 3; ++d) {
        const double len = double(box.length(d));
        K_[d] = good_fft_size(int(std::ceil(len / spacing_)));
        if (K_[d] < order_ + 1)
            K_[d] = good_fft_size(order_ + 1);
        auto mod = spline_moduli(K_[d], order_);
        cudaMalloc(&d_bsp_[d], K_[d] * sizeof(double));
        cudaMemcpy(d_bsp_[d], mod.data(), K_[d] * sizeof(double),
                   cudaMemcpyHostToDevice);
    }
    ntot_ = K_[0] * K_[1] * K_[2];
    cudaMalloc(&d_grid_, ntot_ * sizeof(cufftDoubleComplex));
    cufftHandle plan;
    if (cufftPlan3d(&plan, K_[0], K_[1], K_[2], CUFFT_Z2Z) != CUFFT_SUCCESS)
        throw std::runtime_error("KSpacePME: cufftPlan3d failed");
    plan_ = static_cast<int>(plan);
    have_plan_ = true;
}

void KSpacePME::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = frame.n_local();
    if (ntot_ == 0 || n == 0)
        return;

    auto* grid = static_cast<cufftDoubleComplex*>(d_grid_);
    cufftHandle plan = static_cast<cufftHandle>(plan_);
    cufftSetStream(plan, stream);

    PmeGeom geom;
    for (int i = 0; i < 9; ++i)
        geom.inv_cell[i] = inv_cell_[i];
    geom.K[0] = K_[0];
    geom.K[1] = K_[1];
    geom.K[2] = K_[2];
    geom.order = order_;

    const double escale = ATV::units::coulomb / (2.0 * ATV::units::pi * vol_);
    const double fscale = ATV::units::coulomb / (ATV::units::pi * vol_);
    const double inv_pi2_a2 =
        ATV::units::pi * ATV::units::pi / (alpha_ * alpha_);
    const double self_pref =
        -ATV::units::coulomb * alpha_ / std::sqrt(ATV::units::pi);
    const int blk = kDefaultBlock;

    cudaMemsetAsync(grid, 0, ntot_ * sizeof(cufftDoubleComplex), stream);
    kernels::pme_spread<<<grid_size(n), blk, 0, stream>>>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.charge, n,
        geom, grid);
    cufftExecZ2Z(plan, grid, grid, CUFFT_FORWARD);
    kernels::pme_influence<<<grid_size(ntot_), blk, 0, stream>>>(
        grid, geom, d_bsp_[0], d_bsp_[1], d_bsp_[2], inv_pi2_a2, escale,
        &frame.scalars->energy);
    cufftExecZ2Z(plan, grid, grid, CUFFT_INVERSE);
    kernels::pme_interp<<<grid_size(n), blk, 0, stream>>>(
        frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.atoms.x,
        frame.atoms.y, frame.atoms.z, frame.atoms.charge, n, geom, grid, fscale,
        self_pref, &frame.scalars->energy);
}

void KSpacePME::destroy(const DestroyContext&) {
    if (have_plan_) {
        cufftDestroy(static_cast<cufftHandle>(plan_));
        have_plan_ = false;
    }
    for (int d = 0; d < 3; ++d) {
        cudaFree(d_bsp_[d]);
        d_bsp_[d] = nullptr;
    }
    cudaFree(d_grid_);
    d_grid_ = nullptr;
}

} // namespace cuda
} // namespace ATV
