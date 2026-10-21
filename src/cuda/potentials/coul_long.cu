/// @file cuda/potentials/coul_long.cu
/// @brief CUDA CoulLong — real-space screened Coulomb (OpenMM cluster force).
///
/// Walks NeighborList cluster CSR (`cl_*`) with the same tile layout as
/// PairLJCut / OpenMM. Must share α with KSpacePME / KSpaceEwald.
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"
#include "atomiverse/cuda/potentials/pair_kernel.h"

namespace ATV {

using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace kernels {

// 2/sqrt(pi) (LAMMPS EWALD_F / OpenMM erfc derivative).
__device__ constexpr work_t kEwaldF = work_t(1.1283791670955126);

__global__ void
gather_cluster_pos_q(const int* __restrict__ orig_index,
                     const real_t* __restrict__ x, const real_t* __restrict__ y,
                     const real_t* __restrict__ z, const real_t* __restrict__ q,
                     int n_slots, real_t* __restrict__ spx,
                     real_t* __restrict__ spy, real_t* __restrict__ spz,
                     real_t* __restrict__ spq) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_slots)
        return;
    int a = orig_index[k];
    if (a >= 0) {
        spx[k] = x[a];
        spy[k] = y[a];
        spz[k] = z[a];
        spq[k] = q[a];
    }
}

template <bool Ortho, bool Virial>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void coul_long_tile(
    real_t* __restrict__ fx, real_t* __restrict__ fy, real_t* __restrict__ fz,
    double* __restrict__ energy, double* __restrict__ stress,
    const real_t* __restrict__ spx, const real_t* __restrict__ spy,
    const real_t* __restrict__ spz, const real_t* __restrict__ spq,
    const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const std::uint64_t* __restrict__ cl_mask,
    const int* __restrict__ orig_index, int num_clusters, cuda::Box box,
    bool periodic, work_t alpha, work_t cutoff_sq, work_t qqr2e) {
    constexpr int CS = ATV::cuda::kOpenmmClusterSize;
    static_assert(CS == 32, "tile force assumes OpenMM CS=32");
    constexpr int kMaxWarps = ATV::cuda::kDefaultBlock / 32;
    __shared__ real_t shx[kMaxWarps][CS];
    __shared__ real_t shy[kMaxWarps][CS];
    __shared__ real_t shz[kMaxWarps][CS];
    __shared__ real_t shq[kMaxWarps][CS];
    __shared__ int sh_aj[kMaxWarps][CS];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = tid / CS;
    int slot = tid % CS;
    int warp_in_block = threadIdx.x / 32;

    work_t my_energy = work_t(0);
    work_t my_vxx = work_t(0), my_vyy = work_t(0), my_vzz = work_t(0);
    work_t my_vxy = work_t(0), my_vxz = work_t(0), my_vyz = work_t(0);

    int ai = (ci < num_clusters) ? orig_index[ci * CS + slot] : -1;
    real_t xi = 0, yi = 0, zi = 0, qi = 0;
    real_t fxi = 0, fyi = 0, fzi = 0;
    if (ai >= 0) {
        xi = spx[ci * CS + slot];
        yi = spy[ci * CS + slot];
        zi = spz[ci * CS + slot];
        qi = spq[ci * CS + slot];
    }

    if (ci < num_clusters) {
        int rstart = row_ptr[ci], rend = row_ptr[ci + 1];
        for (int p = rstart; p < rend; ++p) {
            int cj = col_idx[p];
            shx[warp_in_block][slot] = spx[cj * CS + slot];
            shy[warp_in_block][slot] = spy[cj * CS + slot];
            shz[warp_in_block][slot] = spz[cj * CS + slot];
            shq[warp_in_block][slot] = spq[cj * CS + slot];
            sh_aj[warp_in_block][slot] = orig_index[cj * CS + slot];
            __syncwarp();

            if (ai >= 0) {
                unsigned mbits =
                    cl_mask ? ATV::cuda::openmm_mask_jbits(cl_mask, p, slot)
                            : 0xffffffffu;
                while (mbits) {
                    int sj = __ffs(static_cast<int>(mbits)) - 1;
                    mbits &= mbits - 1;
                    int aj = sh_aj[warp_in_block][sj];
                    if (aj < 0 || aj == ai)
                        continue;
                    real_t dx = shx[warp_in_block][sj] - xi;
                    real_t dy = shy[warp_in_block][sj] - yi;
                    real_t dz = shz[warp_in_block][sj] - zi;
                    if (periodic) {
                        if constexpr (Ortho) {
                            dx -= box.cell[0] * nearbyint(dx * box.inv_l[0]);
                            dy -= box.cell[4] * nearbyint(dy * box.inv_l[1]);
                            dz -= box.cell[8] * nearbyint(dz * box.inv_l[2]);
                        } else {
                            box.min_image(dx, dy, dz);
                        }
                    }
                    real_t r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= cutoff_sq || r2 == real_t(0))
                        continue;
                    real_t qj = shq[warp_in_block][sj];
                    real_t r = sqrt(r2);
                    real_t inv_r2 = real_t(1) / r2;
                    real_t grij = alpha * r;
                    real_t expm2 = exp(-grij * grij);
                    real_t erfcv = erfc(grij);
                    real_t prefactor = qqr2e * qi * qj / r;
                    real_t forcecoul =
                        prefactor * (erfcv + real_t(kEwaldF) * grij * expm2);
                    real_t f_over_r = forcecoul * inv_r2;
                    fxi += -f_over_r * dx;
                    fyi += -f_over_r * dy;
                    fzi += -f_over_r * dz;
                    real_t e_pair = prefactor * erfcv;
                    my_energy += work_t(0.5) * work_t(e_pair);
                    if constexpr (Virial) {
                        my_vxx += work_t(0.5) * work_t(dx * (f_over_r * dx));
                        my_vyy += work_t(0.5) * work_t(dy * (f_over_r * dy));
                        my_vzz += work_t(0.5) * work_t(dz * (f_over_r * dz));
                        my_vxy += work_t(0.5) * work_t(dx * (f_over_r * dy));
                        my_vxz += work_t(0.5) * work_t(dx * (f_over_r * dz));
                        my_vyz += work_t(0.5) * work_t(dy * (f_over_r * dz));
                    }
                }
            }
            __syncwarp();
        }
        if (ai >= 0) {
            fx[ai] += fxi;
            fy[ai] += fyi;
            fz[ai] += fzi;
        }
    }

    extern __shared__ work_t warp_scratch[];
    if constexpr (Virial) {
        work_t my_vals[7] = {my_energy, my_vxx, my_vyy, my_vzz,
                             my_vxy,    my_vxz, my_vyz};
        work_t out[7];
        block_reduce_sum_n<7>(my_vals, warp_scratch, out);
        if (threadIdx.x == 0) {
            atomicAdd(energy, static_cast<double>(out[0]));
            if (stress != nullptr) {
#pragma unroll
                for (int c = 0; c < 6; ++c)
                    atomicAdd(&stress[c], static_cast<double>(out[c + 1]));
            }
        }
    } else {
        work_t my_vals[1] = {my_energy};
        work_t out[1];
        block_reduce_sum_n<1>(my_vals, warp_scratch, out);
        if (threadIdx.x == 0)
            atomicAdd(energy, static_cast<double>(out[0]));
    }
}

} // namespace kernels

CoulLong::CoulLong(double alpha, double cutoff) : Pair(cutoff), alpha_(alpha) {
    if (alpha <= 0.0)
        throw std::invalid_argument("CoulLong: alpha must be > 0");
    if (cutoff <= 0.0)
        throw std::invalid_argument("CoulLong: cutoff must be > 0");
}

void CoulLong::build(const BuildContext& ctx) {
    const int slots =
        ((ctx.frame.capacity() + kOpenmmClusterSize - 1) / kOpenmmClusterSize)
        * kOpenmmClusterSize;
    if (slots > spos_cap_) {
        ATV_CUDA_CHECK(cudaFree(spx_));
        ATV_CUDA_CHECK(cudaFree(spy_));
        ATV_CUDA_CHECK(cudaFree(spz_));
        ATV_CUDA_CHECK(cudaFree(spq_));
        spx_ = spy_ = spz_ = spq_ = nullptr;
        ATV_CUDA_CHECK(cudaMalloc(&spx_, slots * sizeof(real_t)));
        ATV_CUDA_CHECK(cudaMalloc(&spy_, slots * sizeof(real_t)));
        ATV_CUDA_CHECK(cudaMalloc(&spz_, slots * sizeof(real_t)));
        ATV_CUDA_CHECK(cudaMalloc(&spq_, slots * sizeof(real_t)));
        spos_cap_ = slots;
    }
}

void CoulLong::destroy(const DestroyContext&) {
    if (spx_) {
        ATV_CUDA_CHECK(cudaFree(spx_));
        ATV_CUDA_CHECK(cudaFree(spy_));
        ATV_CUDA_CHECK(cudaFree(spz_));
        ATV_CUDA_CHECK(cudaFree(spq_));
        spx_ = spy_ = spz_ = spq_ = nullptr;
        spos_cap_ = 0;
    }
}

void CoulLong::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const PairList* pairs = ctx.runtime.md.pair_list;
    if (!pairs)
        return;

    const work_t cutoff_sq =
        static_cast<work_t>(cutoff_) * static_cast<work_t>(cutoff_);
    cuda::Box box = ctx.runtime.host_box;
    double* stress_ptr = compute_stress_ ? &frame.scalars->stress[0] : nullptr;

    // OpenMM cluster force — sole CUDA MD nonbonded path (cuda-openmm-md).
    if (pairs->cl_row_ptr == nullptr || pairs->cl_col_idx == nullptr
        || pairs->cl_orig_index == nullptr || pairs->num_clusters <= 0)
        return;

    constexpr int CS = kOpenmmClusterSize;
    const int nc = pairs->num_clusters;
    const int n_slots = nc * CS;
    if (n_slots > spos_cap_) {
        throw std::runtime_error(
            "cuda::CoulLong: cluster slots exceed Frame capacity at build");
    }
    {
        int block = kDefaultBlock;
        int grid = grid_size(n_slots);
        kernels::gather_cluster_pos_q<<<grid, block, 0, stream>>>(
            pairs->cl_orig_index, frame.atoms.x, frame.atoms.y, frame.atoms.z,
            frame.atoms.charge, n_slots, spx_, spy_, spz_, spq_);
        ATV_CUDA_CHECK(cudaGetLastError());
    }

    const bool virial = (stress_ptr != nullptr);
    int block = kDefaultBlock;
    int grid = grid_size(n_slots);
    int shmem = warps_per_block(block) * sizeof(work_t);

    auto launch = [&](auto ortho_t, auto vir_t) {
        constexpr bool O = decltype(ortho_t)::value;
        constexpr bool V = decltype(vir_t)::value;
        kernels::coul_long_tile<O, V><<<grid, block, shmem, stream>>>(
            frame.atoms.fx, frame.atoms.fy, frame.atoms.fz,
            &frame.scalars->energy, stress_ptr, spx_, spy_, spz_, spq_,
            pairs->cl_row_ptr, pairs->cl_col_idx, pairs->cl_mask,
            pairs->cl_orig_index, nc, box, /*periodic=*/true,
            static_cast<work_t>(alpha_), cutoff_sq,
            static_cast<work_t>(ATV::units::coulomb));
        ATV_CUDA_CHECK(cudaGetLastError());
    };
    using T = std::true_type;
    using F = std::false_type;
    if (box.ortho) {
        if (virial)
            launch(T{}, T{});
        else
            launch(T{}, F{});
    } else {
        if (virial)
            launch(F{}, T{});
        else
            launch(F{}, F{});
    }
}

} // namespace cuda
} // namespace ATV
