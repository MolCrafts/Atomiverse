/// @file cuda/longrange/ewald.cu
/// @brief Reciprocal-space Ewald summation implementation (CUDA).
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"

namespace ATV {

using cuda::real_t;

namespace cuda {
namespace kernels {

/// @brief Structure factor S(k)=Σ_j q_j exp(i k·r_j). One thread per k.
__global__ void ewald_structure_factor(
    const real_t* __restrict__ x, const real_t* __restrict__ y,
    const real_t* __restrict__ z, const real_t* __restrict__ q, int n,
    const double* __restrict__ kx, const double* __restrict__ ky,
    const double* __restrict__ kz, int ng, double* __restrict__ sr,
    double* __restrict__ si) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= ng)
        return;
    const double kxk = kx[k], kyk = ky[k], kzk = kz[k];
    double rr = 0, ii = 0;
    for (int j = 0; j < n; ++j) {
        const double phase =
            kxk * double(x[j]) + kyk * double(y[j]) + kzk * double(z[j]);
        const double qj = double(q[j]);
        rr += qj * cos(phase);
        ii += qj * sin(phase);
    }
    sr[k] = rr;
    si[k] = ii;
}

/// @brief Reciprocal energy Σ_k K·kw·|S(k)|². One thread per k, atomicAdd.
__global__ void ewald_energy(const double* __restrict__ kw,
                             const double* __restrict__ sr,
                             const double* __restrict__ si, int ng, double K,
                             double* __restrict__ energy) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= ng)
        return;
    const double e = K * kw[k] * (sr[k] * sr[k] + si[k] * si[k]);
    atomicAdd(energy, static_cast<double>(e));
}

/// @brief Reciprocal forces on owned atoms + per-atom self energy.
__global__ void
ewald_forces(real_t* __restrict__ fx, real_t* __restrict__ fy,
             real_t* __restrict__ fz, const real_t* __restrict__ x,
             const real_t* __restrict__ y, const real_t* __restrict__ z,
             const real_t* __restrict__ q, int n, const double* __restrict__ kx,
             const double* __restrict__ ky, const double* __restrict__ kz,
             const double* __restrict__ kw, const double* __restrict__ sr,
             const double* __restrict__ si, int ng, double K, double self_pref,
             double* __restrict__ energy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const double xi = double(x[i]), yi = double(y[i]), zi = double(z[i]);
    const double qi = double(q[i]);
    double fxi = 0, fyi = 0, fzi = 0;
    for (int k = 0; k < ng; ++k) {
        const double phase = kx[k] * xi + ky[k] * yi + kz[k] * zi;
        const double c = cos(phase), s = sin(phase);
        const double fac = 2.0 * K * kw[k] * qi * (s * sr[k] - c * si[k]);
        fxi += fac * kx[k];
        fyi += fac * ky[k];
        fzi += fac * kz[k];
    }
    fx[i] += real_t(fxi);
    fy[i] += real_t(fyi);
    fz[i] += real_t(fzi);
    // Self energy (partitions per atom): -ATV::units::coulomb·(α/√π)·q_i².
    atomicAdd(energy, static_cast<double>(self_pref * qi * qi));
}

} // namespace kernels

KSpaceEwald::KSpaceEwald(double alpha, double accuracy)
    : alpha_(alpha), accuracy_(accuracy) {
    if (alpha <= 0.0)
        throw std::invalid_argument("KSpaceEwald: alpha must be > 0");
}

KSpaceEwald::~KSpaceEwald() {}

void KSpaceEwald::build(const BuildContext& ctx) {
    // Reciprocal basis b_i = 2π·(column i of inv_cell); g = Σ m_i b_i.
    const cuda::MotionBox& box = ctx.runtime.host_motion_box;
    const double tp = 2.0 * ATV::units::pi;
    double b[3][3];
    for (int i = 0; i < 3; ++i)
        for (int c = 0; c < 3; ++c)
            b[i][c] = tp * double(box.inv_cell[c * 3 + i]);
    auto blen = [&](int i) {
        return std::sqrt(b[i][0] * b[i][0] + b[i][1] * b[i][1]
                         + b[i][2] * b[i][2]);
    };
    const double gcut = 2.0 * alpha_ * std::sqrt(-std::log(accuracy_));
    const double gcut2 = gcut * gcut;
    int kmax[3];
    for (int i = 0; i < 3; ++i)
        kmax[i] = std::max(1, static_cast<int>(std::ceil(gcut / blen(i))));
    const double inv4a2 = 1.0 / (4.0 * alpha_ * alpha_);

    std::vector<double> kx, ky, kz, kw;
    for (int mx = -kmax[0]; mx <= kmax[0]; ++mx)
        for (int my = -kmax[1]; my <= kmax[1]; ++my)
            for (int mz = -kmax[2]; mz <= kmax[2]; ++mz) {
                if (mx == 0 && my == 0 && mz == 0)
                    continue;
                const double gx = mx * b[0][0] + my * b[1][0] + mz * b[2][0];
                const double gy = mx * b[0][1] + my * b[1][1] + mz * b[2][1];
                const double gz = mx * b[0][2] + my * b[1][2] + mz * b[2][2];
                const double g2 = gx * gx + gy * gy + gz * gz;
                if (g2 > gcut2)
                    continue;
                kx.push_back(gx);
                ky.push_back(gy);
                kz.push_back(gz);
                kw.push_back(std::exp(-g2 * inv4a2) / g2);
            }
    ng_ = static_cast<int>(kx.size());
    const size_t bytes = ng_ * sizeof(double);
    ATV_CUDA_CHECK(cudaMalloc(&d_kx_, bytes));
    ATV_CUDA_CHECK(cudaMalloc(&d_ky_, bytes));
    ATV_CUDA_CHECK(cudaMalloc(&d_kz_, bytes));
    ATV_CUDA_CHECK(cudaMalloc(&d_kw_, bytes));
    ATV_CUDA_CHECK(cudaMalloc(&d_sr_, bytes));
    ATV_CUDA_CHECK(cudaMalloc(&d_si_, bytes));
    ATV_CUDA_CHECK(cudaMemcpy(d_kx_, kx.data(), bytes, cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_ky_, ky.data(), bytes, cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_kz_, kz.data(), bytes, cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_kw_, kw.data(), bytes, cudaMemcpyHostToDevice));
}

void KSpaceEwald::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = frame.n_local();
    if (ng_ == 0 || n == 0)
        return;

    const double vol =
        static_cast<double>(ctx.runtime.host_motion_box.volume());
    const double K = ATV::units::coulomb * 2.0 * ATV::units::pi / vol;
    const double self_pref =
        -ATV::units::coulomb * alpha_ / std::sqrt(ATV::units::pi);

    const int blk = kDefaultBlock;
    kernels::ewald_structure_factor<<<grid_size(ng_), blk, 0, stream>>>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.charge, n,
        d_kx_, d_ky_, d_kz_, ng_, d_sr_, d_si_);
    ATV_CUDA_CHECK(cudaGetLastError());
    kernels::ewald_energy<<<grid_size(ng_), blk, 0, stream>>>(
        d_kw_, d_sr_, d_si_, ng_, K, &frame.scalars->energy);
    ATV_CUDA_CHECK(cudaGetLastError());
    kernels::ewald_forces<<<grid_size(n), blk, 0, stream>>>(
        frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.atoms.x,
        frame.atoms.y, frame.atoms.z, frame.atoms.charge, n, d_kx_, d_ky_,
        d_kz_, d_kw_, d_sr_, d_si_, ng_, K, self_pref, &frame.scalars->energy);
    ATV_CUDA_CHECK(cudaGetLastError());
    // Note: the net-charge correction (nonzero only for non-neutral systems)
    // is omitted on the CUDA path in this revision; see cpu::KSpaceEwald.
}

void KSpaceEwald::destroy(const DestroyContext&) {
    ATV_CUDA_CHECK(cudaFree(d_kx_));
    ATV_CUDA_CHECK(cudaFree(d_ky_));
    ATV_CUDA_CHECK(cudaFree(d_kz_));
    ATV_CUDA_CHECK(cudaFree(d_kw_));
    ATV_CUDA_CHECK(cudaFree(d_sr_));
    ATV_CUDA_CHECK(cudaFree(d_si_));
    d_kx_ = d_ky_ = d_kz_ = d_kw_ = d_sr_ = d_si_ = nullptr;
}

} // namespace cuda
} // namespace ATV
