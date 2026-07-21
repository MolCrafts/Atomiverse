/// @file cuda/potentials/coul_long.cu
/// @brief CUDA CoulLong — real-space screened Coulomb (kernel + class).
///
/// Cloned from the atom-centric, atomic-free `lj_force_cluster` kernel
/// (src/cuda/potentials/pair_lj_cut.cu): one thread owns one atom, walks its
/// cluster's FULL neighbor list, computes each pair from current positions,
/// accumulates force in registers, writes once. Energy/virial halved to undo
/// the double-visit of a full neighbor list.

#include <cuda_runtime.h>
#include <stdexcept>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace kernels {

// 2/sqrt(pi) (LAMMPS EWALD_F).
__device__ constexpr work_t kEwaldF = work_t(1.1283791670955126);

template <bool Ortho>
__global__
__launch_bounds__(ATV::cuda::kDefaultBlock, 4) void coul_long_cluster(
    real_t* __restrict__ fx, real_t* __restrict__ fy, real_t* __restrict__ fz,
    prec_t* __restrict__ energy, prec_t* __restrict__ stress,
    const real_t* __restrict__ x, const real_t* __restrict__ y,
    const real_t* __restrict__ z, const real_t* __restrict__ q,
    const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const int* __restrict__ orig_index, int num_clusters, cuda::Box box,
    bool periodic, work_t alpha, work_t cutoff_sq, work_t qqr2e) {
    constexpr int CS = 8; // kClusterSize
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = tid / CS;
    int slot = tid % CS;

    work_t my_energy = work_t(0);
    work_t my_vxx = work_t(0), my_vyy = work_t(0), my_vzz = work_t(0);
    work_t my_vxy = work_t(0), my_vxz = work_t(0), my_vyz = work_t(0);

    if (ci < num_clusters) {
        int ai = orig_index[ci * CS + slot];
        if (ai >= 0) {
            const work_t xi = work_t(x[ai]), yi = work_t(y[ai]),
                         zi = work_t(z[ai]);
            const work_t qi = work_t(q[ai]);
            work_t fxi = work_t(0), fyi = work_t(0), fzi = work_t(0);
            int rstart = row_ptr[ci], rend = row_ptr[ci + 1];
            for (int p = rstart; p < rend; ++p) {
                int cj = col_idx[p];
#pragma unroll
                for (int sj = 0; sj < CS; ++sj) {
                    int aj = orig_index[cj * CS + sj];
                    if (aj < 0 || aj == ai)
                        continue;
                    work_t dx = work_t(x[aj]) - xi;
                    work_t dy = work_t(y[aj]) - yi;
                    work_t dz = work_t(z[aj]) - zi;
                    if (periodic) {
                        if constexpr (Ortho) {
                            dx -= box.cell[0] * nearbyint(dx * box.inv_l[0]);
                            dy -= box.cell[4] * nearbyint(dy * box.inv_l[1]);
                            dz -= box.cell[8] * nearbyint(dz * box.inv_l[2]);
                        } else {
                            box.min_image(dx, dy, dz);
                        }
                    }
                    work_t r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= cutoff_sq || r2 == work_t(0))
                        continue;
                    const work_t qj = work_t(q[aj]);
                    const work_t r = sqrt(r2);
                    const work_t inv_r2 = work_t(1) / r2;
                    const work_t grij = alpha * r;
                    const work_t expm2 = exp(-grij * grij);
                    const work_t erfcv = erfc(grij);
                    const work_t prefactor = qqr2e * qi * qj / r;
                    const work_t forcecoul =
                        prefactor * (erfcv + kEwaldF * grij * expm2);
                    const work_t f_over_r = forcecoul * inv_r2;
                    // dx = x[aj]-x[ai], force on ai is -f_over_r·dx.
                    fxi += -f_over_r * dx;
                    fyi += -f_over_r * dy;
                    fzi += -f_over_r * dz;
                    const work_t e_pair = prefactor * erfcv;
                    my_energy += work_t(0.5) * e_pair;
                    my_vxx += work_t(0.5) * (dx * (f_over_r * dx));
                    my_vyy += work_t(0.5) * (dy * (f_over_r * dy));
                    my_vzz += work_t(0.5) * (dz * (f_over_r * dz));
                    my_vxy += work_t(0.5) * (dx * (f_over_r * dy));
                    my_vxz += work_t(0.5) * (dx * (f_over_r * dz));
                    my_vyz += work_t(0.5) * (dy * (f_over_r * dz));
                }
            }
            fx[ai] += real_t(fxi);
            fy[ai] += real_t(fyi);
            fz[ai] += real_t(fzi);
        }
    }

    extern __shared__ work_t warp_scratch[];
    work_t my_vals[7] = {my_energy, my_vxx, my_vyy, my_vzz,
                         my_vxy,    my_vxz, my_vyz};
    work_t out[7];
    block_reduce_sum_n<7>(my_vals, warp_scratch, out);
    if (threadIdx.x == 0) {
        atomicAdd(energy, static_cast<prec_t>(out[0]));
        if (stress != nullptr) {
#pragma unroll
            for (int c = 0; c < 6; ++c)
                atomicAdd(&stress[c], static_cast<prec_t>(out[c + 1]));
        }
    }
}

} // namespace kernels

CoulLong::CoulLong(double alpha, double cutoff) : Pair(cutoff), alpha_(alpha) {
    if (alpha <= 0.0)
        throw std::invalid_argument("CoulLong: alpha must be > 0");
    if (cutoff <= 0.0)
        throw std::invalid_argument("CoulLong: cutoff must be > 0");
}

void CoulLong::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    // TODO(roykid, 2026-07-10): the cluster CSR read here is only POPULATED
    // by NeighborList's flat_pairs_requested rebuild block, and nothing
    // in-tree sets that flag since the RDF observer migrated to the molrs
    // accumulator (arch-refactor-07). The pointers below are published
    // unconditionally (they pass this guard) but point at never-filled
    // placeholders unless a flat-pair requester is registered — the Coulomb
    // path needs its own cluster-CSR request (GPU-node verification owed).
    const PairList* pairs = ctx.runtime.md.pair_list;
    if (!pairs || pairs->cl_row_ptr == nullptr || pairs->num_clusters <= 0)
        return;
    constexpr int CS = 8;
    int block = kDefaultBlock;
    int grid = grid_size(pairs->num_clusters * CS);
    int shmem = warps_per_block(block) * sizeof(work_t);
    const work_t cutoff_sq =
        static_cast<work_t>(cutoff_) * static_cast<work_t>(cutoff_);
    cuda::Box box = ctx.runtime.host_box;
    if (box.ortho) {
        kernels::coul_long_cluster<true><<<grid, block, shmem, stream>>>(
            frame.atoms.fx, frame.atoms.fy, frame.atoms.fz,
            &frame.scalars->energy, &frame.scalars->stress[0], frame.atoms.x,
            frame.atoms.y, frame.atoms.z, frame.atoms.charge, pairs->cl_row_ptr,
            pairs->cl_col_idx, pairs->cl_orig_index, pairs->num_clusters, box,
            /*periodic=*/true, static_cast<work_t>(alpha_), cutoff_sq,
            static_cast<work_t>(ATV::kCoulomb));
    } else {
        kernels::coul_long_cluster<false><<<grid, block, shmem, stream>>>(
            frame.atoms.fx, frame.atoms.fy, frame.atoms.fz,
            &frame.scalars->energy, &frame.scalars->stress[0], frame.atoms.x,
            frame.atoms.y, frame.atoms.z, frame.atoms.charge, pairs->cl_row_ptr,
            pairs->cl_col_idx, pairs->cl_orig_index, pairs->num_clusters, box,
            /*periodic=*/true, static_cast<work_t>(alpha_), cutoff_sq,
            static_cast<work_t>(ATV::kCoulomb));
    }
}

} // namespace cuda
} // namespace ATV
