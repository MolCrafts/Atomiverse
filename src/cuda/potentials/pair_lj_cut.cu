/// @file cuda/potentials/pair_lj_cut.cu
/// @brief CUDA PairLJCut — Lennard-Jones 12-6 shifted (kernel + class).
///        Includes LAMMPS-style `special_bonds` 1-2/1-3/1-4 exclusions.

#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/lj_kernel.cuh"
#include "atomiverse/cuda/potentials/pair_kernel.h"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <set>
#include <stdexcept>
#include <string>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace ATV {

namespace cuda {
namespace kernels {

// Gather current atom positions into the cluster-sorted slot layout so the
// force kernel reads neighbour positions COALESCED (spx[cj*8+sj] is contiguous
// per cluster) instead of scattering x[orig_index[..]] ~80×/atom. One scattered
// read per slot here amortises the scatter that previously happened per pair —
// the dominant cost of the pre-tile atom-walk force (latency-bound gathers).
__global__ void gather_cluster_pos(const int* __restrict__ orig_index,
                                   const pos_t* __restrict__ x,
                                   const pos_t* __restrict__ y,
                                   const pos_t* __restrict__ z, int n_slots,
                                   force_t* __restrict__ spx,
                                   force_t* __restrict__ spy,
                                   force_t* __restrict__ spz) {
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
/// OpenMM-style tile force: one warp per cluster (CS=32). Neighbor cluster
/// positions are staged in per-warp shared memory; interaction mask selects
/// active j-slots. FULL neighbor + 0.5 energy (both endpoints accumulate).
template <bool Ortho, bool Virial>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void lj_force_tile(
    fixed_t* __restrict__ fx, fixed_t* __restrict__ fy,
    fixed_t* __restrict__ fz, ForceFault* fault, acc_t* __restrict__ energy,
    acc_t* __restrict__ stress, const force_t* __restrict__ spx,
    const force_t* __restrict__ spy, const force_t* __restrict__ spz,
    const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const std::uint64_t* __restrict__ cl_mask,
    const int* __restrict__ orig_index, int num_clusters, int n_owned,
    cuda::Box box_arg, bool periodic, force_t epsilon, force_t sigma_sq,
    force_t cutoff_sq, force_t min_pair_dist_sq, force_t v_shift,
    const int* __restrict__ excl_offsets, const int* __restrict__ excl_partners,
    force_t s_excl) {
    const auto box = box_arg.current().as<force_t>();
    constexpr int CS = ATV::cuda::kOpenmmClusterSize;
    static_assert(CS == 32, "tile force assumes OpenMM CS=32");

    // Per-warp tile staging: [warp][slot]
    constexpr int kMaxWarps = ATV::cuda::kDefaultBlock / 32;
    __shared__ force_t shx[kMaxWarps][CS];
    __shared__ force_t shy[kMaxWarps][CS];
    __shared__ force_t shz[kMaxWarps][CS];
    __shared__ int sh_aj[kMaxWarps][CS];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = tid / CS;
    int slot = tid % CS;
    int warp_in_block = threadIdx.x / 32;

    acc_t my_energy = 0;
    acc_t my_vxx = 0, my_vyy = 0, my_vzz = 0;
    acc_t my_vxy = 0, my_vxz = 0, my_vyz = 0;

    const force_t eps_r = force_t(epsilon);
    const force_t sigsq_r = force_t(sigma_sq);
    const force_t cut_r = force_t(cutoff_sq);
    const force_t minr_r = force_t(min_pair_dist_sq);
    const force_t vshift_r = force_t(v_shift);
    const force_t sexcl_r = force_t(s_excl);

    // All lanes of a warp must participate in __syncwarp; pad slots keep
    // ai < 0 and skip force math but still load tiles.
    int ai = (ci < num_clusters) ? orig_index[ci * CS + slot] : -1;
    if (ai >= n_owned)
        ai = -1;
    force_t xi = 0, yi = 0, zi = 0;
    force_t fxi = 0, fyi = 0, fzi = 0;
    bool has_excl = false;
    int beg_excl = 0, end_excl = 0;
    if (ai >= 0) {
        xi = spx[ci * CS + slot];
        yi = spy[ci * CS + slot];
        zi = spz[ci * CS + slot];
        has_excl = (excl_offsets != nullptr);
        beg_excl = has_excl ? excl_offsets[ai] : 0;
        end_excl = has_excl ? excl_offsets[ai + 1] : 0;
    }

    if (ci < num_clusters) {
        int rstart = row_ptr[ci], rend = row_ptr[ci + 1];
        for (int p = rstart; p < rend; ++p) {
            int cj = col_idx[p];
            shx[warp_in_block][slot] = spx[cj * CS + slot];
            shy[warp_in_block][slot] = spy[cj * CS + slot];
            shz[warp_in_block][slot] = spz[cj * CS + slot];
            sh_aj[warp_in_block][slot] = orig_index[cj * CS + slot];
            __syncwarp();

            if (ai >= 0) {
                unsigned mbits = ATV::cuda::openmm_mask_jbits(cl_mask, p, slot);
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
                    if (r2 >= cut_r || r2 < minr_r)
                        continue;
                    force_t scale = force_t(1);
                    if (has_excl && end_excl > beg_excl
                        && thrust::binary_search(
                            thrust::seq, excl_partners + beg_excl,
                            excl_partners + end_excl, aj)) {
                        if (sexcl_r == force_t(0))
                            continue;
                        scale = sexcl_r;
                    }
                    force_t inv_r2 = force_t(1) / r2;
                    force_t sr2 = sigsq_r * inv_r2;
                    force_t sr6 = sr2 * sr2 * sr2;
                    force_t sr12 = sr6 * sr6;
                    force_t f_over_r = scale * force_t(24) * eps_r
                                       * (force_t(2) * sr12 - sr6) * inv_r2;
                    fxi += -f_over_r * dx;
                    fyi += -f_over_r * dy;
                    fzi += -f_over_r * dz;
                    force_t e_pair =
                        scale * (force_t(4) * eps_r * (sr12 - sr6) - vshift_r);
                    my_energy += acc_t(0.5) * acc_t(e_pair);
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

    // CUB reduces energy and each virial component in a single collective.
    if constexpr (Virial) {
        ::cuda::std::array<acc_t, 7> my_vals = {
            my_energy, my_vxx, my_vyy, my_vzz, my_vxy, my_vxz, my_vyz};

        using Reduce =
            cub::BlockReduce<::cuda::std::array<acc_t, 7>, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const auto out =
            Reduce(reduction).Reduce(my_vals, SumComponents<acc_t, 7>{});
        if (threadIdx.x == 0) {
            atomicAdd(energy, out[0]);
            if (stress != nullptr) {
#pragma unroll
                for (int c = 0; c < 6; ++c)
                    atomicAdd(&stress[c], out[c + 1]);
            }
        }
    } else {
        // Virial gated off: reduce energy alone; the my_v* accumulators stay
        // 0 and are dead-code-eliminated, freeing their registers.
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
    const int slots =
        ((ctx.frame.capacity() + kOpenmmClusterSize - 1) / kOpenmmClusterSize)
        * kOpenmmClusterSize;
    if (slots > spos_cap_) {
        cudaFree(spx_);
        cudaFree(spy_);
        cudaFree(spz_);
        spx_ = spy_ = spz_ = nullptr;
        ATV_CUDA_CHECK(cudaMalloc(&spx_, slots * sizeof(force_t)));
        ATV_CUDA_CHECK(cudaMalloc(&spy_, slots * sizeof(force_t)));
        ATV_CUDA_CHECK(cudaMalloc(&spz_, slots * sizeof(force_t)));
        spos_cap_ = slots;
    }
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
    ctx.runtime.md.exclusions_baked = false;
    if (!has_exclusions_) {
        // No special bonds: publish "nothing to bake".
        ctx.runtime.md.excl_offsets = nullptr;
        ctx.runtime.md.excl_partners = nullptr;
        ctx.runtime.md.s_excl = 1.0;
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
}

void PairLJCut::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const PairList* pairs = ctx.runtime.md.pair_list;
    if (!pairs)
        return;

    const bool baked = ctx.runtime.md.exclusions_baked;
    const int* excl_off =
        (has_exclusions_ && !baked) ? d_excl_offsets_ : nullptr;
    const int* excl_par =
        (has_exclusions_ && !baked) ? d_excl_partners_ : nullptr;
    const force_t cutoff_sq =
        static_cast<force_t>(cutoff_) * static_cast<force_t>(cutoff_);
    const force_t sigma_sq =
        static_cast<force_t>(sigma_) * static_cast<force_t>(sigma_);
    const force_t min_pair_dist_sq =
        static_cast<force_t>(kernels::kMinPairDist)
        * static_cast<force_t>(kernels::kMinPairDist);
    cuda::Box box = ctx.runtime.host_box;
    acc_t* stress_ptr = compute_stress_ ? &frame.scalars->stress[0] : nullptr;
    const bool virial = (stress_ptr != nullptr);

    // OpenMM cluster force — sole CUDA MD nonbonded path (cuda-openmm-md).
    if (pairs->cl_row_ptr == nullptr || pairs->cl_col_idx == nullptr
        || pairs->cl_orig_index == nullptr || pairs->num_clusters <= 0)
        return;

    constexpr int CS = kOpenmmClusterSize;
    const int nc = pairs->num_clusters;
    const int n_slots = nc * CS;
    // Live positions from frame (not rebuild-time cluster_x). Prefer published
    // cl_x only as a gather target cache when present is wrong — always gather
    // current frame.atoms into scratch so force is skin-correct.
    if (n_slots > spos_cap_) {
        throw std::runtime_error(
            "cuda::PairLJCut: cluster slots exceed Frame capacity at build");
    }
    {
        int block = kDefaultBlock;
        int grid = grid_size(n_slots);
        kernels::gather_cluster_pos<<<grid, block, 0, stream>>>(
            pairs->cl_orig_index, frame.atoms.x, frame.atoms.y, frame.atoms.z,
            n_slots, spx_, spy_, spz_);
    }

    const uint64_t* mask = pairs->cl_mask;
    int block = kDefaultBlock;
    int grid = grid_size(n_slots);
    constexpr int shmem = 0;

    auto launch = [&](auto ortho_t, auto vir_t) {
        constexpr bool O = decltype(ortho_t)::value;
        constexpr bool V = decltype(vir_t)::value;
        kernels::lj_force_tile<O, V><<<grid, block, shmem, stream>>>(
            frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.force_fault,
            &frame.scalars->energy, stress_ptr, spx_, spy_, spz_,
            pairs->cl_row_ptr, pairs->cl_col_idx, mask, pairs->cl_orig_index,
            nc, frame.n_local(), box, /*periodic=*/true,
            static_cast<force_t>(epsilon_), sigma_sq, cutoff_sq,
            min_pair_dist_sq, static_cast<force_t>(v_shift_), excl_off,
            excl_par, static_cast<force_t>(s_excl_));
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
