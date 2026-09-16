/// @file cuda/potentials/lj_kernel.cuh
/// @brief Pointer-level LJ kernel -- shared pointer-level kernel for cuda::PairLJCut.
///
/// This kernel is the single source of truth for LJ force+energy computation.
/// It accepts raw pointers so it can be called from any backend.
#pragma once

#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace kernels {

/// @brief Minimum pair distance to avoid LJ singularity [Angstrom].
inline constexpr cuda::work_t kMinPairDist = cuda::work_t(1e-10);

/// @brief LJ 12-6 force + energy + pair virial kernel. One thread per pair.
///
/// Forces accumulated via atomicAdd (Newton's 3rd law); energy + 6
/// stress components reduced per-block via `cuda::block_reduce_sum_n<7>`
/// (warp-shuffle, single per-scalar barrier). Per
/// cuda-md-review-cleanup-06-launch-config-streams, the kernel works
/// in r² space — `pdist_sq` stores `dx²+dy²+dz²` (no sqrt at export),
/// `cutoff_sq` and `sigma_sq` are precomputed by the caller, and
/// the per-thread `inv_r²` is a single FP64 reciprocal (no sqrt,
/// no inv_r·inv_r chain).
///
/// @param fx,fy,fz        Force arrays [n_atoms, device]. Accumulated via atomicAdd.
/// @param energy          Potential-energy scalar [1, device]. atomicAdd.
/// @param stress          Pair virial [6, device]. atomicAdd, may be nullptr.
/// @param pi,pj           Pair atom indices [n_pairs, device].
/// @param pdx,pdy,pdz     Pair displacements [n_pairs, device].
/// @param pdist_sq        Pair distance² [n_pairs, device].
/// @param epsilon         LJ well depth [eV].
/// @param sigma_sq        LJ collision diameter squared [Å²].
/// @param cutoff_sq       Interaction cutoff squared [Å²].
/// @param min_pair_dist_sq Singularity-guard floor² [Å²] (kMinPairDist²).
/// @param v_shift         Potential shift at cutoff [eV].
/// @param n_pairs_ptr     Pair count [1, device].
/// @param excl_offsets,excl_partners  Special-bonds CSR (nullptr → no exclusions).
/// @param s_excl          Special-bonds scale factor.
__global__ void lj_force(
    cuda::real_t* __restrict__ fx, cuda::real_t* __restrict__ fy,
    cuda::real_t* __restrict__ fz, cuda::prec_t* __restrict__ energy,
    cuda::prec_t* __restrict__ stress, const int* __restrict__ pi,
    const int* __restrict__ pj, const cuda::real_t* __restrict__ pdx,
    const cuda::real_t* __restrict__ pdy, const cuda::real_t* __restrict__ pdz,
    const cuda::real_t* __restrict__ pdist_sq, cuda::work_t epsilon,
    cuda::work_t sigma_sq, cuda::work_t cutoff_sq,
    cuda::work_t min_pair_dist_sq, cuda::work_t v_shift,
    const int* __restrict__ n_pairs_ptr, const int* __restrict__ excl_offsets,
    const int* __restrict__ excl_partners, cuda::work_t s_excl);

/// @brief Compute LJ potential shift at cutoff.
/// @param epsilon Well depth [eV].
/// @param sigma Collision diameter [Angstrom].
/// @param cutoff Interaction cutoff [Angstrom].
/// @return Shift value V(cutoff).
inline double lj_shift(double epsilon, double sigma, double cutoff) {
    double sr = sigma / cutoff;
    double sr2 = sr * sr;
    double sr6 = sr2 * sr2 * sr2;
    double sr12 = sr6 * sr6;
    return 4.0 * epsilon * (sr12 - sr6);
}

} // namespace kernels
} // namespace cuda
} // namespace ATV
