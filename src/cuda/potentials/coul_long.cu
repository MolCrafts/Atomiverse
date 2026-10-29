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
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"
#include "atomiverse/cuda/potentials/pair_kernel.h"

namespace ATV {

namespace cuda {
namespace kernels {

// 2/sqrt(pi) (LAMMPS EWALD_F / OpenMM erfc derivative).
__device__ constexpr force_t kEwaldF = force_t(1.1283791670955126);

__global__ void
gather_cluster_pos_q(const int* __restrict__ orig_index,
                     const pos_t* __restrict__ x, const pos_t* __restrict__ y,
                     const pos_t* __restrict__ z, const force_t* __restrict__ q,
                     int n_slots, force_t* __restrict__ spx,
                     force_t* __restrict__ spy, force_t* __restrict__ spz,
                     force_t* __restrict__ spq) {
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

template <bool Ortho, bool Energy, bool Virial>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void coul_long_tile(
    fixed_t* __restrict__ fx, fixed_t* __restrict__ fy,
    fixed_t* __restrict__ fz, ForceFault* fault, acc_t* __restrict__ energy,
    acc_t* __restrict__ stress, const force_t* __restrict__ spx,
    const force_t* __restrict__ spy, const force_t* __restrict__ spz,
    const force_t* __restrict__ spq, const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, const std::uint64_t* __restrict__ cl_mask,
    const int* __restrict__ orig_index, int num_clusters, cuda::Box box_arg,
    bool periodic, force_t alpha, force_t cutoff_sq, force_t qqr2e) {
    const auto box = box_arg.current().as<force_t>();
    constexpr int CS = ATV::cuda::kOpenmmClusterSize;
    static_assert(CS == 32, "tile force assumes OpenMM CS=32");
    constexpr int kMaxWarps = ATV::cuda::kDefaultBlock / 32;
    __shared__ force_t shx[kMaxWarps][CS];
    __shared__ force_t shy[kMaxWarps][CS];
    __shared__ force_t shz[kMaxWarps][CS];
    __shared__ force_t shq[kMaxWarps][CS];
    __shared__ int sh_aj[kMaxWarps][CS];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = tid / CS;
    int slot = tid % CS;
    int warp_in_block = threadIdx.x / 32;

    acc_t my_energy = 0;
    acc_t my_vxx = 0, my_vyy = 0, my_vzz = 0;
    acc_t my_vxy = 0, my_vxz = 0, my_vyz = 0;

    int ai = (ci < num_clusters) ? orig_index[ci * CS + slot] : -1;
    force_t xi = 0, yi = 0, zi = 0, qi = 0;
    force_t fxi = 0, fyi = 0, fzi = 0;
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
                    force_t dx = shx[warp_in_block][sj] - xi;
                    force_t dy = shy[warp_in_block][sj] - yi;
                    force_t dz = shz[warp_in_block][sj] - zi;
                    if (periodic) {
                        if constexpr (Ortho) {
                            dx -= box.cell[0] * nearbyint(dx * box.inv_l[0]);
                            dy -= box.cell[4] * nearbyint(dy * box.inv_l[1]);
                            dz -= box.cell[8] * nearbyint(dz * box.inv_l[2]);
                        } else {
                            box.min_image(dx, dy, dz);
                        }
                    }
                    force_t r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= cutoff_sq || r2 == force_t(0))
                        continue;
                    force_t qj = shq[warp_in_block][sj];
                    force_t r = sqrt(r2);
                    force_t inv_r2 = force_t(1) / r2;
                    force_t grij = alpha * r;
                    force_t expm2 = exp(-grij * grij);
                    force_t erfcv = erfc(grij);
                    force_t prefactor = qqr2e * qi * qj / r;
                    force_t forcecoul =
                        prefactor * (erfcv + force_t(kEwaldF) * grij * expm2);
                    force_t f_over_r = forcecoul * inv_r2;
                    fxi += -f_over_r * dx;
                    fyi += -f_over_r * dy;
                    fzi += -f_over_r * dz;
                    if constexpr (Energy)
                        my_energy += acc_t(0.5) * acc_t(prefactor * erfcv);
                    if constexpr (Virial) {
                        my_vxx += acc_t(0.5) * acc_t(dx * (f_over_r * dx));
                        my_vyy += acc_t(0.5) * acc_t(dy * (f_over_r * dy));
                        my_vzz += acc_t(0.5) * acc_t(dz * (f_over_r * dz));
                        my_vxy += acc_t(0.5) * acc_t(dx * (f_over_r * dy));
                        my_vxz += acc_t(0.5) * acc_t(dx * (f_over_r * dz));
                        my_vyz += acc_t(0.5) * acc_t(dy * (f_over_r * dz));
                    }
                }
            }
            __syncwarp();
        }
        // Full list: each owned row is written by exactly one lane.
        if (ai >= 0) {
            add_force_owned(fx, ai, fxi, fault);
            add_force_owned(fy, ai, fyi, fault);
            add_force_owned(fz, ai, fzi, fault);
        }
    }

    if constexpr (Virial) {
        ::cuda::std::array<acc_t, 7> my_vals = {
            my_energy, my_vxx, my_vyy, my_vzz, my_vxy, my_vxz, my_vyz};

        using Reduce =
            cub::BlockReduce<::cuda::std::array<acc_t, 7>, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const auto out =
            Reduce(reduction).Reduce(my_vals, SumComponents<acc_t, 7>{});
        if (threadIdx.x == 0) {
            if constexpr (Energy)
                atomicAdd(energy, out[0]);
            if (stress != nullptr) {
#pragma unroll
                for (int c = 0; c < 6; ++c)
                    atomicAdd(&stress[c], out[c + 1]);
            }
        }
    } else if constexpr (Energy) {
        // A step that asked for neither skips the reduction entirely.
        ::cuda::std::array<acc_t, 1> my_vals = {my_energy};

        using Reduce =
            cub::BlockReduce<::cuda::std::array<acc_t, 1>, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const auto out =
            Reduce(reduction).Reduce(my_vals, SumComponents<acc_t, 1>{});
        if (threadIdx.x == 0)
            atomicAdd(energy, out[0]);
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
        cudaFree(spx_);
        cudaFree(spy_);
        cudaFree(spz_);
        cudaFree(spq_);
        spx_ = spy_ = spz_ = spq_ = nullptr;
        ATV_CUDA_CHECK(cudaMalloc(&spx_, slots * sizeof(force_t)));
        ATV_CUDA_CHECK(cudaMalloc(&spy_, slots * sizeof(force_t)));
        ATV_CUDA_CHECK(cudaMalloc(&spz_, slots * sizeof(force_t)));
        ATV_CUDA_CHECK(cudaMalloc(&spq_, slots * sizeof(force_t)));
        spos_cap_ = slots;
    }
}

void CoulLong::destroy(const DestroyContext&) {
    if (spx_) {
        cudaFree(spx_);
        cudaFree(spy_);
        cudaFree(spz_);
        cudaFree(spq_);
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

    const force_t cutoff_sq =
        static_cast<force_t>(cutoff_) * static_cast<force_t>(cutoff_);
    cuda::Box box = ctx.runtime.host_box;
    // Per-step LAMMPS vflag (the step's consumers decide).
    acc_t* stress_ptr =
        ctx.stress_requested ? &frame.scalars->stress[0] : nullptr;

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
    }

    const bool energy = ctx.energy_requested;
    const bool virial = (stress_ptr != nullptr);
    int block = kDefaultBlock;
    int grid = grid_size(n_slots);
    constexpr int shmem = 0;

    auto launch = [&](auto ortho_t, auto energy_t, auto vir_t) {
        constexpr bool O = decltype(ortho_t)::value;
        constexpr bool E = decltype(energy_t)::value;
        constexpr bool V = decltype(vir_t)::value;
        kernels::coul_long_tile<O, E, V><<<grid, block, shmem, stream>>>(
            frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.force_fault,
            &frame.scalars->energy, stress_ptr, spx_, spy_, spz_, spq_,
            pairs->cl_row_ptr, pairs->cl_col_idx, pairs->cl_mask,
            pairs->cl_orig_index, nc, box, /*periodic=*/true,
            static_cast<force_t>(alpha_), cutoff_sq,
            static_cast<force_t>(ATV::units::coulomb));
    };
    auto by_flags = [&](auto ortho_t) {
        using T = std::true_type;
        using F = std::false_type;
        if (energy && virial)
            launch(ortho_t, T{}, T{});
        else if (energy)
            launch(ortho_t, T{}, F{});
        else if (virial)
            launch(ortho_t, F{}, T{});
        else
            launch(ortho_t, F{}, F{});
    };
    if (box.ortho)
        by_flags(std::true_type{});
    else
        by_flags(std::false_type{});
}

} // namespace cuda
} // namespace ATV
