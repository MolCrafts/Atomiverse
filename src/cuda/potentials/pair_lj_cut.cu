/// @file cuda/potentials/pair_lj_cut.cu
/// @brief CUDA PairLJCut — Lennard-Jones 12-6 shifted (kernel + class).
///        Includes LAMMPS-style `special_bonds` 1-2/1-3/1-4 exclusions.

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/lj_kernel.cuh"
#include "atomiverse/cuda/potentials/pair_compute.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/potentials/pair_kernel.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace kernels {

/// @brief Device-side binary search of a sorted int range.
///        Returns true if `target` is in `[beg, end)`.
__device__ inline bool excl_contains(const int* __restrict__ partners, int beg,
                                     int end, int target) {
    int lo = beg, hi = end;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int v = partners[mid];
        if (v == target)
            return true;
        if (v < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return false;
}

// Gather current atom positions into the cluster-sorted slot layout so the
// force kernel reads neighbour positions COALESCED (spx[cj*8+sj] is contiguous
// per cluster) instead of scattering x[orig_index[..]] ~80×/atom. One scattered
// read per slot here amortises the scatter that previously happened per pair —
// the dominant cost of lj_force_cluster (latency-bound uncoalesced gathers).
__global__ void gather_cluster_pos(const int* __restrict__ orig_index,
                                   const real_t* __restrict__ x,
                                   const real_t* __restrict__ y,
                                   const real_t* __restrict__ z, int n_slots,
                                   real_t* __restrict__ spx,
                                   real_t* __restrict__ spy,
                                   real_t* __restrict__ spz) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_slots)
        return;
    int a = orig_index[k];
    if (a >= 0) {
        spx[k] = x[a];
        spy[k] = y[a];
        spz[k] = z[a];
    }
}

/// @brief Atom-centric, **atomic-free** LJ force. One thread owns one atom
///        (cluster slot). It walks its cluster's FULL neighbor-cluster list
///        (col_idx over [row_ptr[ci], row_ptr[ci+1])), computes each pair
///        distance from CURRENT atom positions (passed pre-gathered in the
///        cluster-sorted layout spx/spy/spz, min-image), accumulates the owned
///        atom's force in registers, and writes it ONCE — no atomicAdd, no flat
///        pair list, no per-step export.
///
/// Replaces the old one-thread-per-pair kernel that did 6 atomicAdd/pair
/// (~160-way contention per force slot in a dense melt → the 76%-of-runtime
/// bottleneck). With a FULL neighbor list each interaction is computed twice
/// (once by each endpoint's owning thread), so the force on each atom is
/// complete without writing its neighbors; energy and virial are halved to
/// undo the double count. kClusterSize is the neighbor list's fixed cluster
/// width; pad slots carry a negative orig_index sentinel.
template <bool Ortho, bool Virial>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void lj_force_cluster(
    real_t* __restrict__ fx, real_t* __restrict__ fy, real_t* __restrict__ fz,
    prec_t* __restrict__ energy, prec_t* __restrict__ stress,
    const real_t* __restrict__ spx, const real_t* __restrict__ spy,
    const real_t* __restrict__ spz, const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, const std::uint64_t* __restrict__ cl_mask,
    const int* __restrict__ orig_index, int num_clusters, cuda::Box box,
    bool periodic, work_t epsilon, work_t sigma_sq, work_t cutoff_sq,
    work_t min_pair_dist_sq, work_t v_shift,
    const int* __restrict__ excl_offsets, const int* __restrict__ excl_partners,
    work_t s_excl) {
    constexpr int CS = 8; // kClusterSize (neighbor-list cluster width)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = tid / CS;
    int slot = tid % CS;

    work_t my_energy = work_t(0);
    work_t my_vxx = work_t(0), my_vyy = work_t(0), my_vzz = work_t(0);
    work_t my_vxy = work_t(0), my_vxz = work_t(0), my_vyz = work_t(0);

    // Mixed precision (matches LAMMPS KOKKOS SINGLE_DOUBLE): geometry + force
    // in real_t (float — FP32 is 2x FP64 throughput on GH200 and the divide is
    // far cheaper), energy/virial accumulated in work_t (double). Force is
    // stored as real_t anyway, so float accumulation loses nothing.
    const real_t eps_r = real_t(epsilon);
    const real_t sigsq_r = real_t(sigma_sq);
    const real_t cut_r = real_t(cutoff_sq);
    const real_t minr_r = real_t(min_pair_dist_sq);
    const real_t vshift_r = real_t(v_shift);
    const real_t sexcl_r = real_t(s_excl);

    if (ci < num_clusters) {
        int ai = orig_index[ci * CS + slot];
        if (ai >= 0) {
            real_t xi = spx[ci * CS + slot], yi = spy[ci * CS + slot],
                   zi = spz[ci * CS + slot];
            real_t fxi = real_t(0), fyi = real_t(0), fzi = real_t(0);
            const bool has_excl = (excl_offsets != nullptr);
            int beg_excl = has_excl ? excl_offsets[ai] : 0;
            int end_excl = has_excl ? excl_offsets[ai + 1] : 0;

            int rstart = row_ptr[ci], rend = row_ptr[ci + 1];
            for (int p = rstart; p < rend; ++p) {
                int cj = col_idx[p];
                // This atom owns row `slot` of cluster ci: byte slot*8..slot*8+7
                // of the pair mask flags which j-slots of cj are within the
                // search radius. Iterating only set bits replaces the blind
                // 8-slot scan + per-slot distance prefilter — the dominant cost
                // — with ≈ the real neighbor count. (All-ones when no real mask
                // was computed ⇒ identical to the old full scan.)
                unsigned mbyte =
                    static_cast<unsigned>((cl_mask[p] >> (slot * 8)) & 0xFFull);
                while (mbyte) {
                    int sj = __ffs(static_cast<int>(mbyte)) - 1;
                    mbyte &= mbyte - 1;
                    int aj = orig_index[cj * CS + sj];
                    if (aj < 0 || aj == ai)
                        continue;
                    real_t dx = spx[cj * CS + sj] - xi;
                    real_t dy = spy[cj * CS + sj] - yi;
                    real_t dz = spz[cj * CS + sj] - zi;
                    if (periodic) {
                        if constexpr (Ortho) {
                            dx -= box.cell[0] * nearbyintf(dx * box.inv_l[0]);
                            dy -= box.cell[4] * nearbyintf(dy * box.inv_l[1]);
                            dz -= box.cell[8] * nearbyintf(dz * box.inv_l[2]);
                        } else {
                            box.min_image(dx, dy, dz);
                        }
                    }
                    real_t r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= cut_r || r2 < minr_r)
                        continue;
                    real_t scale = real_t(1);
                    if (has_excl && end_excl > beg_excl
                        && excl_contains(excl_partners, beg_excl, end_excl,
                                         aj)) {
                        if (sexcl_r == real_t(0))
                            continue;
                        scale = sexcl_r;
                    }
                    real_t inv_r2 = real_t(1) / r2;
                    real_t sr2 = sigsq_r * inv_r2;
                    real_t sr6 = sr2 * sr2 * sr2;
                    real_t sr12 = sr6 * sr6;
                    real_t f_over_r = scale * real_t(24) * eps_r
                                      * (real_t(2) * sr12 - sr6) * inv_r2;
                    // dx = x[aj] - x[ai], so the force ON ai is -f_over_r·dx.
                    fxi += -f_over_r * dx;
                    fyi += -f_over_r * dy;
                    fzi += -f_over_r * dz;
                    // Each pair is visited twice (ai's and aj's threads): halve
                    // the energy/virial to recover the unique-pair sum. Promote
                    // the per-pair term to work_t for the double accumulation.
                    real_t e_pair =
                        scale * (real_t(4) * eps_r * (sr12 - sr6) - vshift_r);
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
            // Single non-atomic accumulate: this thread uniquely owns atom ai
            // (orig_index is a permutation), and force-phase kernels serialize
            // on the stream, so += safely adds to bonded contributions.
            fx[ai] += fxi;
            fy[ai] += fyi;
            fz[ai] += fzi;
        }
    }

    // Packed warp-shuffle reduction: 7 scalars (energy + 6 stress
    // components) reduced via cg::reduce. Replaces the original
    // sequential 7-pass shared-mem reductions with 7 × `__syncthreads`
    // (one per scalar inside `block_reduce_sum`) instead of 7 × 8
    // shared-mem barrier-steps. Drops shared-mem footprint from
    // `blockDim.x × sizeof(work_t)` to (blockDim.x / 32) × sizeof(work_t)
    // (cuda-md-review-cleanup-05-reductions-and-atomics).
    extern __shared__ work_t warp_scratch[]; // per-warp staging cells

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
        // Virial gated off: reduce energy alone; the my_v* accumulators stay
        // 0 and are dead-code-eliminated, freeing their registers.
        work_t my_vals[1] = {my_energy};
        work_t out[1];
        block_reduce_sum_n<1>(my_vals, warp_scratch, out);
        if (threadIdx.x == 0)
            atomicAdd(energy, static_cast<prec_t>(out[0]));
    }
}

} // namespace kernels
} // namespace cuda

namespace cuda {

PairLJCut::PairLJCut(double epsilon, double sigma, double cutoff)
    : Pair(cutoff), epsilon_(epsilon), sigma_(sigma) {
    v_shift_ = kernels::lj_shift(epsilon, sigma, cutoff);
}

PairLJCut& PairLJCut::set_special_bonds(double s12, double s13, double s14) & {
    // Supported combinations: all equal (incl. (0,0,0)) or all 1.0.
    if (s12 == 1.0 && s13 == 1.0 && s14 == 1.0) {
        s12_ = s13_ = s14_ = 1.0;
        s_excl_ = 1.0;
        has_exclusions_ = false;
        return *this;
    }
    if (s12 != s13 || s12 != s14) {
        throw std::invalid_argument(
            "cuda::PairLJCut::set_special_bonds: mixed scaling combinations "
            "(s12 != s13 or s12 != s14) are not supported in this revision. "
            "Pass three equal values (typically (0,0,0) for Dreiding-UA).");
    }
    s12_ = s13_ = s14_ = s12;
    s_excl_ = s12;
    has_exclusions_ = true; // exclusions enumerated at build()
    return *this;
}

PairLJCut PairLJCut::set_special_bonds(double s12, double s13, double s14) && {
    // Rvalue overload: delegate to the lvalue body, then move-return so
    // the chained `cuda::PairLJCut{…}.set_special_bonds(…)` expression
    // yields a movable rvalue (not a dangling lvalue reference to a
    // temporary). cf. `cpu::PairLJCut::set_special_bonds(…) &&`.
    this->set_special_bonds(s12, s13, s14);
    return std::move(*this);
}

void PairLJCut::enable_full_exclusions() {
    // cuda::Pair hook: full 1-2/1-3/1-4 exclusion is set_special_bonds(0,0,0).
    set_special_bonds(0.0, 0.0, 0.0);
}

PairLJCut& PairLJCut::set_compute_stress(bool on) & {
    compute_stress_ = on;
    return *this;
}

PairLJCut PairLJCut::set_compute_stress(bool on) && {
    compute_stress_ = on;
    return std::move(*this);
}

void PairLJCut::build(const BuildContext& ctx) {
    // Re-callable: the multi-GPU bonded path rebuilds the exclusion CSR every
    // reneighbor (the flat topology's local indices change as ghosts move), so
    // free any prior CSR first to avoid leaking on repeated build().
    if (d_excl_offsets_) {
        cudaFree(d_excl_offsets_);
        d_excl_offsets_ = nullptr;
    }
    if (d_excl_partners_) {
        cudaFree(d_excl_partners_);
        d_excl_partners_ = nullptr;
    }
    n_excl_total_ = 0;
    if (!has_exclusions_) {
        // No special bonds: publish "nothing to bake".
        ctx.runtime.md.excl_offsets = nullptr;
        ctx.runtime.md.excl_partners = nullptr;
        ctx.runtime.md.s_excl = 1.0;
        return;
    }
    // Fast path: if the per-atom special SoA is on-device, NeighborList bakes
    // 1-2/1-3/1-4 exclusions directly from it (find_special at build) — skip the
    // O(N) host std::set exclusion-CSR rebuild entirely (its per-reneighbor cost
    // was the dominant multi-GPU bottleneck). Just publish the scale; the force
    // kernel reads exclusions_baked and drops its per-pair scan.
    if (ctx.frame.atoms.special != nullptr) {
        ctx.runtime.md.excl_offsets = nullptr;
        ctx.runtime.md.excl_partners = nullptr;
        ctx.runtime.md.s_excl = s_excl_;
        return;
    }
    auto& f = ctx.frame;
    int n = f.n_atoms();
    int n_b = f.n_bonds();
    int n_a = f.n_angles();
    int n_d = f.n_dihedrals();

    // D2H read of topology (build-time cost; not in hot path).
    std::vector<int> bi(n_b), bj(n_b);
    std::vector<int> ai(n_a), ak(n_a);
    std::vector<int> di(n_d), dl(n_d);
    if (n_b > 0) {
        cudaMemcpy(bi.data(), f.bonds.atomi, n_b * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(bj.data(), f.bonds.atomj, n_b * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }
    if (n_a > 0) {
        cudaMemcpy(ai.data(), f.angles.atomi, n_a * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(ak.data(), f.angles.atomk, n_a * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }
    if (n_d > 0) {
        cudaMemcpy(di.data(), f.dihedrals.atomi, n_d * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(dl.data(), f.dihedrals.atoml, n_d * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    // Per-atom sorted-unique partner sets.
    std::vector<std::set<int>> excl(n);
    auto add_pair = [&](int a, int b) {
        if (a >= 0 && a < n && b >= 0 && b < n && a != b) {
            excl[a].insert(b);
            excl[b].insert(a);
        }
    };
    for (int k = 0; k < n_b; ++k)
        add_pair(bi[k], bj[k]);
    for (int k = 0; k < n_a; ++k)
        add_pair(ai[k], ak[k]);
    for (int k = 0; k < n_d; ++k)
        add_pair(di[k], dl[k]);

    // Pack into CSR.
    std::vector<int> offsets(n + 1, 0);
    for (int i = 0; i < n; ++i)
        offsets[i + 1] = offsets[i] + static_cast<int>(excl[i].size());
    n_excl_total_ = offsets[n];
    std::vector<int> partners(n_excl_total_);
    for (int i = 0; i < n; ++i) {
        int cursor = offsets[i];
        for (int p : excl[i])
            partners[cursor++] = p;
        // std::set already sorted ascending; nothing to do.
    }

    // Upload to device.
    cudaMalloc(&d_excl_offsets_, (n + 1) * sizeof(int));
    cudaMemcpy(d_excl_offsets_, offsets.data(), (n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    if (n_excl_total_ > 0) {
        cudaMalloc(&d_excl_partners_, n_excl_total_ * sizeof(int));
        cudaMemcpy(d_excl_partners_, partners.data(),
                   n_excl_total_ * sizeof(int), cudaMemcpyHostToDevice);
    }
    // Publish the CSR so NeighborList::launch can bake exclusions into the
    // per-atom Verlet list (skipping the per-pair scan in the force kernel).
    ctx.runtime.md.excl_offsets = d_excl_offsets_;
    ctx.runtime.md.excl_partners = d_excl_partners_;
    ctx.runtime.md.s_excl = s_excl_;
}

void PairLJCut::destroy(const DestroyContext& /*ctx*/) {
    if (d_excl_offsets_) {
        cudaFree(d_excl_offsets_);
        d_excl_offsets_ = nullptr;
    }
    if (d_excl_partners_) {
        cudaFree(d_excl_partners_);
        d_excl_partners_ = nullptr;
    }
    n_excl_total_ = 0;
    if (spx_) {
        cudaFree(spx_);
        cudaFree(spy_);
        cudaFree(spz_);
        spx_ = spy_ = spz_ = nullptr;
        spos_cap_ = 0;
    }
    if (pos4_) {
        cudaFree(pos4_);
        pos4_ = nullptr;
        pos4_cap_ = 0;
    }
}

void PairLJCut::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const PairList* pairs = ctx.runtime.md.pair_list;
    if (!pairs)
        return;
    // Per-atom Verlet-list path: one thread per atom iterates its tight
    // neighbor list (built once per rebuild by NeighborList) reading neighbor
    // positions directly from the frame's current x/y/z. No cluster
    // indirection, no position gather, no atomicAdd on forces (thread i
    // uniquely owns fx[i]). Replaces the cluster-CSR walk that left ~7/32 warp
    // lanes active (ncu).
    if (pairs->vl_nbr == nullptr || pairs->vl_cnt == nullptr
        || pairs->vl_stride <= 0)
        return;
    int n_atoms = frame.n_atoms(); // active owned(+ghost) count this launch
    if (n_atoms <= 0)
        return;

    // Pack the frame's current positions into AoS float4 so the force kernel's
    // neighbour-position gather is one ≤16-byte transaction per neighbour
    // (coalesced pack, then an efficient gather) — see pos4_ doc / pack_pos4.
    if (n_atoms > pos4_cap_) {
        if (pos4_)
            cudaFree(pos4_);
        cudaMalloc(&pos4_, static_cast<std::size_t>(n_atoms) * sizeof(float4));
        pos4_cap_ = n_atoms;
    }
    md::launch_pack_pos4(frame.atoms.x, frame.atoms.y, frame.atoms.z, pos4_,
                         n_atoms, stream);

    // Pass nullptr for exclusion CSR when none configured — kernel takes the
    // fast path and avoids the partner lookup overhead entirely. ALSO pass
    // nullptr when NeighborList already baked the exclusions into the per-atom
    // Verlet list (excluded partners simply aren't in it): the force kernel then
    // does zero per-pair exclusion work — the LAMMPS layout.
    const bool baked = ctx.runtime.md.exclusions_baked;
    const int* excl_off =
        (has_exclusions_ && !baked) ? d_excl_offsets_ : nullptr;
    const int* excl_par =
        (has_exclusions_ && !baked) ? d_excl_partners_ : nullptr;
    const work_t cutoff_sq =
        static_cast<work_t>(cutoff_) * static_cast<work_t>(cutoff_);
    const work_t sigma_sq =
        static_cast<work_t>(sigma_) * static_cast<work_t>(sigma_);
    const work_t min_pair_dist_sq =
        static_cast<work_t>(kernels::kMinPairDist)
        * static_cast<work_t>(kernels::kMinPairDist);
    cuda::Box box = ctx.runtime.host_box;
    // When stress is gated off, pass a null stress pointer (the launcher selects
    // the Virial=false template, skipping the 6 inner-loop virial products and
    // the stress reduction; frees the accumulator registers).
    prec_t* stress_ptr = compute_stress_ ? &frame.scalars->stress[0] : nullptr;
    // LJ force math via the backend-agnostic PairKernel seam (force-infra-03).
    // Instantiated at real_t inside the launcher → identical float-math path as
    // the prior hand-written lj_force_verlet. LJ is charge-free (q = nullptr).
    const ATV::potentials::LJKernel lj{
        static_cast<double>(epsilon_), static_cast<double>(sigma_sq),
        static_cast<double>(cutoff_sq), static_cast<double>(v_shift_)};
    // Force-row selection: the multi-GPU engine restricts force rows to owned
    // atoms (ghost pair rows are wrong to reverse-communicate and waste work)
    // and may pass an explicit interior/boundary subset. Default (-1): every
    // atom row, the single-GPU behavior.
    const int n_rows = (ctx.runtime.md.n_force_rows >= 0)
                           ? ctx.runtime.md.n_force_rows
                           : n_atoms;
    md::launch_pair_compute(
        lj, frame.atoms.fx, frame.atoms.fy, frame.atoms.fz,
        &frame.scalars->energy, stress_ptr, pos4_, /*q=*/nullptr, pairs->vl_nbr,
        pairs->vl_cnt, pairs->vl_stride, n_rows, ctx.runtime.md.force_rows, box,
        /*periodic=*/true, cutoff_sq, min_pair_dist_sq, excl_off, excl_par,
        static_cast<work_t>(s_excl_), stream);
}

} // namespace cuda
} // namespace ATV
