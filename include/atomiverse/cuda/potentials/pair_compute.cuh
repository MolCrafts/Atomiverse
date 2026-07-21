/// @file cuda/potentials/pair_compute.cuh
/// @brief Generic per-atom Verlet-list force launcher (force-infra-03).
///
/// The shared CUDA pair traversal + reduction, parameterized on a
/// `ATV::potentials::PairKernel` (LJ / Coulomb-real / combined). One thread per
/// atom iterates its FULL per-atom Verlet list, computes the per-pair force via
/// the kernel, accumulates the owning atom's force in registers (atomic-free —
/// thread i uniquely owns fx[i]), and block-reduces energy + the 6 virial
/// components. The full list double-counts, so energy/virial are halved —
/// identical ownership/structure to the prior hand-written `lj_force_verlet`.
///
/// The kernel is instantiated at `real_t` (the prior float-math path), so the
/// numerics match bit-for-bit. `Kernel::needs_charge` gates the per-pair charge
/// read so charge-free styles (LJ) pay nothing for it.
#pragma once

#include <cstddef>

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/potentials/pair_kernel.h"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Binary search for `target` in the sorted exclusion partner slice.
__device__ inline bool pc_excl_contains(const int* __restrict__ partners,
                                        int beg, int end, int target) {
    int lo = beg, hi = end;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        const int v = partners[mid];
        if (v == target)
            return true;
        if (v < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return false;
}

// Virial path needs ~7 extra accumulators; capping at 6 resident blocks/SM
// starved it of registers (ncu: ~80M local-memory spill requests at 1M atoms).
// Give the virial instantiation more registers (min 4 blocks/SM) while keeping
// the lean no-virial path dense (6 blocks/SM).
/// Pack SoA positions into an AoS float4 [x,y,z,_]. The neighbor-position
/// gather in pair_compute_verlet reads one packed float4 per neighbor (a single
/// ≤16-byte transaction) instead of three scattered 4-byte loads from separate
/// x/y/z arrays. ncu measured the SoA gather at 23.8 sectors/request and 5.3
/// useful bytes/sector (L2-bandwidth-bound) for large multi-GPU subdomains;
/// the packed gather roughly triples bytes/sector. The pack itself is a fully
/// coalesced one-pass copy.
static __global__ void pack_pos4(const real_t* __restrict__ x,
                                 const real_t* __restrict__ y,
                                 const real_t* __restrict__ z,
                                 float4* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = make_float4(x[i], y[i], z[i], 0.0f);
}

/// Launch the SoA→AoS position pack (coalesced one-pass copy).
inline void launch_pack_pos4(const real_t* x, const real_t* y, const real_t* z,
                             float4* out, int n, cudaStream_t stream) {
    if (n <= 0)
        return;
    const int block = kDefaultBlock;
    const int grid = (n + block - 1) / block;
    pack_pos4<<<grid, block, 0, stream>>>(x, y, z, out, n);
}

template <bool Ortho, bool Virial, bool Periodic, bool HasExcl, class Kernel>
__global__ __launch_bounds__(
    ATV::cuda::kDefaultBlock,
    Virial ? 4
           : 6) void pair_compute_verlet(real_t* __restrict__ fx,
                                         real_t* __restrict__ fy,
                                         real_t* __restrict__ fz,
                                         prec_t* __restrict__ energy,
                                         prec_t* __restrict__ stress,
                                         const float4* __restrict__ pos4,
                                         const real_t* __restrict__ q,
                                         const int* __restrict__ vl_nbr,
                                         const int* __restrict__ vl_cnt,
                                         int vl_stride, int n_rows,
                                         const int* __restrict__ rows,
                                         cuda::Box box, Kernel kernel,
                                         work_t cutoff_sq,
                                         work_t min_pair_dist_sq,
                                         const int* __restrict__ excl_offsets,
                                         const int* __restrict__ excl_partners,
                                         work_t s_excl) {
    // Row indirection (multi-GPU): thread t computes force row i = rows[t]
    // (an owned-atom subset — interior or boundary), or i = t when rows is
    // null (the full/prefix launch). Neighbor positions still come from the
    // whole owned+ghost pos4 either way.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = (t < n_rows) ? (rows ? rows[t] : t) : -1;

    work_t my_energy = work_t(0);
    work_t my_vxx = work_t(0), my_vyy = work_t(0), my_vzz = work_t(0);
    work_t my_vxy = work_t(0), my_vxz = work_t(0), my_vyz = work_t(0);

    const real_t cut_r = real_t(cutoff_sq);
    const real_t minr_r = real_t(min_pair_dist_sq);
    const real_t sexcl_r = real_t(s_excl);

    if (i >= 0) {
        const float4 pi = pos4[i];
        const real_t xi = pi.x, yi = pi.y, zi = pi.z;
        real_t fxi = real_t(0), fyi = real_t(0), fzi = real_t(0);
        real_t qi = real_t(0);
        if constexpr (Kernel::needs_charge)
            qi = q[i];
        int beg_excl = 0, end_excl = 0;
        if constexpr (HasExcl) {
            beg_excl = excl_offsets[i];
            end_excl = excl_offsets[i + 1];
        }

        const int cnt = vl_cnt[i];
        for (int k = 0; k < cnt; ++k) {
            const int aj = vl_nbr[static_cast<std::size_t>(k) * vl_stride + i];
            const float4 pj = pos4[aj];
            real_t dx = pj.x - xi;
            real_t dy = pj.y - yi;
            real_t dz = pj.z - zi;
            if constexpr (Periodic) {
                if constexpr (Ortho) {
                    dx -= box.cell[0] * nearbyintf(dx * box.inv_l[0]);
                    dy -= box.cell[4] * nearbyintf(dy * box.inv_l[1]);
                    dz -= box.cell[8] * nearbyintf(dz * box.inv_l[2]);
                } else {
                    box.min_image(dx, dy, dz);
                }
            }
            const real_t r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cut_r || r2 < minr_r)
                continue;
            real_t scale = real_t(1);
            if constexpr (HasExcl) {
                if (end_excl > beg_excl
                    && pc_excl_contains(excl_partners, beg_excl, end_excl,
                                        aj)) {
                    if (sexcl_r == real_t(0))
                        continue;
                    scale = sexcl_r;
                }
            }
            real_t qj = real_t(0);
            if constexpr (Kernel::needs_charge)
                qj = q[aj];

            // raw(): the launcher already pre-skipped r2 >= cutoff_sq, so the
            // leaf kernel need not re-mask (perf-parity with the hand kernel).
            const ATV::potentials::PairContribution<real_t> c =
                kernel.raw(r2, qi, qj);
            const real_t f_over_r = scale * c.fpair;
            fxi += -f_over_r * dx;
            fyi += -f_over_r * dy;
            fzi += -f_over_r * dz;
            my_energy += work_t(0.5) * work_t(scale * c.energy);
            if constexpr (Virial) {
                my_vxx += work_t(0.5) * work_t(dx * (f_over_r * dx));
                my_vyy += work_t(0.5) * work_t(dy * (f_over_r * dy));
                my_vzz += work_t(0.5) * work_t(dz * (f_over_r * dz));
                my_vxy += work_t(0.5) * work_t(dx * (f_over_r * dy));
                my_vxz += work_t(0.5) * work_t(dx * (f_over_r * dz));
                my_vyz += work_t(0.5) * work_t(dy * (f_over_r * dz));
            }
        }
        fx[i] += fxi;
        fy[i] += fyi;
        fz[i] += fzi;
    }

    extern __shared__ work_t warp_scratch[];
    if constexpr (Virial) {
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
    } else {
        work_t my_vals[1] = {my_energy};
        work_t out[1];
        block_reduce_sum_n<1>(my_vals, warp_scratch, out);
        if (threadIdx.x == 0)
            atomicAdd(energy, static_cast<prec_t>(out[0]));
    }
}

/// @brief Host launcher: Ortho/Virial dispatch + geometry, mirroring the prior
///        PairLJCut::compute. `stress != nullptr` selects the virial template.
///
/// `n_rows`/`rows` select which force rows the kernel computes: `rows ==
/// nullptr` ⇒ the prefix [0, n_rows) (n_rows == n_atoms reproduces the
/// historical full launch); non-null `rows` ⇒ the explicit owned-atom subset
/// `rows[0..n_rows)` (multi-GPU interior/boundary split).
template <class Kernel>
void launch_pair_compute(const Kernel& kernel, real_t* fx, real_t* fy,
                         real_t* fz, prec_t* energy, prec_t* stress,
                         const float4* pos4, const real_t* q, const int* vl_nbr,
                         const int* vl_cnt, int vl_stride, int n_rows,
                         const int* rows, cuda::Box box, bool periodic,
                         work_t cutoff_sq, work_t min_pair_dist_sq,
                         const int* excl_offsets, const int* excl_partners,
                         work_t s_excl, cudaStream_t stream) {
    if (n_rows <= 0)
        return;
    const int block = kDefaultBlock;
    const int grid = grid_size(n_rows);
    const int shmem = warps_per_block(block) * sizeof(work_t);
    const bool virial = (stress != nullptr);
    const bool has_excl = (excl_offsets != nullptr);
    // Compile-time specialize on box-orthogonality, virial, periodicity, and
    // presence of an exclusion CSR. Templating Periodic/HasExcl lets the
    // compiler drop the per-pair min-image / partner-lookup branches and the
    // associated dead code + registers (the common path is Periodic, no-excl).
    auto go = [&](auto O_t, auto V_t, auto P_t, auto E_t) {
        constexpr bool O = decltype(O_t)::value;
        constexpr bool V = decltype(V_t)::value;
        constexpr bool P = decltype(P_t)::value;
        constexpr bool E = decltype(E_t)::value;
        pair_compute_verlet<O, V, P, E, Kernel><<<grid, block, shmem, stream>>>(
            fx, fy, fz, energy, stress, pos4, q, vl_nbr, vl_cnt, vl_stride,
            n_rows, rows, box, kernel, cutoff_sq, min_pair_dist_sq,
            excl_offsets, excl_partners, s_excl);
    };
    using T = std::true_type;
    using F = std::false_type;
    auto pick_excl = [&](auto O_t, auto V_t, auto P_t) {
        if (has_excl)
            go(O_t, V_t, P_t, T{});
        else
            go(O_t, V_t, P_t, F{});
    };
    auto pick_periodic = [&](auto O_t, auto V_t) {
        if (periodic)
            pick_excl(O_t, V_t, T{});
        else
            pick_excl(O_t, V_t, F{});
    };
    auto pick_virial = [&](auto O_t) {
        if (virial)
            pick_periodic(O_t, T{});
        else
            pick_periodic(O_t, F{});
    };
    if (box.ortho)
        pick_virial(T{});
    else
        pick_virial(F{});
}

} // namespace md
} // namespace cuda
} // namespace ATV
