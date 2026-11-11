/// @file cuda/potentials/internal/cluster_tile.cuh
/// @brief The CUDA cluster-tile pair engine: one templated packed-J tile
///        kernel over a `cuda::tile` term (`cuda/potentials/pair_kernel.cuh`),
///        shared by every short-range pair style.
///
/// Algorithm reference: OpenMM nonbonded.cu and findInteractingBlocks.cu,
/// commit 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e (packed J atoms, warp data
/// exchange, single periodic copy). The implementation is ATV-native, with
/// no copied OpenMM source.
///
/// Per neighbor rebuild (a hook inside the list's rebuild body): the baked
/// CSR masks are packed into upper-triangle tiles of one 32-atom I cluster
/// and 32 sorted J slots (`pack_tiles`), and the term's static per-atom
/// parameters are gathered into slot order. Per step the tile kernel reads
/// the list's live slot positions and cluster images (published by its skin
/// gate), so a style does no per-step gather of its own. A warp owns one
/// tile; each pair is evaluated once and its force goes to both atoms
/// (Newton's third law through warp shuffles and deferred fixed-point
/// atomics). Energy and virial go to per-chunk partials that a separate
/// reduction folds into the frame, after any sparse correction pass the
/// style writes into the extra partial slots.
///
/// nvcc only; include from the style's translation unit.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cuda/std/array>
#include <cuda_runtime.h>
#include <stdexcept>

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_kernel.cuh"
#include "packed_tiles.cuh"

namespace ATV::cuda::internal {
namespace {

// 128 threads x 4 resident blocks: 256/2, 128/3 and 128/6 measured no
// better on GH200 (256/2 is 7% slower in double at 49,680 atoms).
constexpr int kTileBlock = 128;
constexpr int kTileMinBlocks = 4;
using ForceBox = BoxGeometry<force_t>;

__device__ __forceinline__ void wrap_to(force_t& x, force_t& y, force_t& z,
                                        const ClusterImage& image,
                                        const ForceBox& box) {
    x -= box.cell[0] * nearbyint((x - image.center[0]) * box.inv_l[0]);
    y -= box.cell[4] * nearbyint((y - image.center[1]) * box.inv_l[1]);
    z -= box.cell[8] * nearbyint((z - image.center[2]) * box.inv_l[2]);
}

template <class Term>
__global__ void gather_tile_params(Term term, const int* __restrict__ index,
                                   int slots,
                                   typename Term::Param* __restrict__ out) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= slots)
        return;
    const int i = index[s];
    out[s] = i >= 0 ? term.param(i) : typename Term::Param{};
}

// Like OpenMM's nonbonded energy buffer, each producer writes a disjoint
// partial; a separate reduction avoids serializing every force block on one
// system energy/stress address. The caller supplies its block size to CUB.
template <bool Virial, int BlockSize>
__device__ void tally(acc_t e, const acc_t* v, acc_t* partial,
                      int partial_stride, int slot) {
    constexpr int components = Virial ? 7 : 1;
    using Values = ::cuda::std::array<acc_t, components>;
    using Reduce = cub::BlockReduce<Values, BlockSize>;
    __shared__ typename Reduce::TempStorage reduction;
    Values values{};
    values[0] = e;
    if constexpr (Virial)
        for (int d = 0; d < 6; ++d)
            values[d + 1] = v[d];
    const auto out =
        Reduce(reduction).Reduce(values, SumComponents<acc_t, components>{});
    if (threadIdx.x == 0) {
        partial[slot] = out[0];
        if constexpr (Virial)
            for (int d = 0; d < 6; ++d)
                partial[static_cast<std::size_t>(d + 1) * partial_stride
                        + slot] = out[d + 1];
    }
}

// Tile partials occupy [0, live chunks) of a reserve of `tile_slots`; the
// style's extra partials follow the reserve. Logical index i enumerates the
// live tile chunks, then the extras, so each block sums the same partials in
// the same order as a contiguous layout would.
template <bool Energy, bool Virial>
__global__ void reduce_partials(const acc_t* partial, const int* tile_count,
                                int tile_slots, int extra_slots, int stride,
                                EnergyTerm energy, acc_t* stress) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunks = (*tile_count + warps_per_block(kTileBlock) - 1)
                       / warps_per_block(kTileBlock);
    constexpr int components = Virial ? 7 : 1;
    using Values = ::cuda::std::array<acc_t, components>;
    using Reduce = cub::BlockReduce<Values, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    Values values{};
    if (i < chunks + extra_slots) {
        const int slot = i < chunks ? i : tile_slots + (i - chunks);
        for (int d = 0; d < components; ++d)
            values[d] = partial[static_cast<std::size_t>(d) * stride + slot];
    }
    const auto reduced =
        Reduce(reduction).Reduce(values, SumComponents<acc_t, components>{});
    if (threadIdx.x == 0) {
        if constexpr (Energy)
            energy.add(reduced[0]);
        if constexpr (Virial)
            for (int d = 0; d < 6; ++d)
                atomicAdd(stress + d, reduced[d + 1]);
    }
}

// A warp owns one I cluster and 32 packed J atoms; the packed masks hold
// only upper-triangle interactions. Pair math runs in force_t, energy and
// virial accumulate in acc_t per interaction, and each lane adds its I and J
// forces to the fixed-point columns once per tile. Rows at or past
// `n_owned` (multi-GPU halo ghosts) take no force and half the energy and
// virial of a pair with an owned partner: the ghost's owner counts the other
// half. One block per reserved chunk of `warps` tiles; a block past the live
// (device) tile count exits before any barrier (a striding loop cost 17%).
template <class Term, bool Ortho, bool Energy, bool Virial>
__global__ __launch_bounds__(kTileBlock, kTileMinBlocks) void cluster_tile(
    Term term, const pos_t* __restrict__ cx, const pos_t* __restrict__ cy,
    const pos_t* __restrict__ cz,
    const typename Term::Param* __restrict__ params,
    const int* __restrict__ index, const int* __restrict__ packed_ci,
    const int* __restrict__ packed_j, const std::uint32_t* __restrict__ masks,
    const int* __restrict__ tile_count, Box box_arg,
    const ClusterImage* __restrict__ images, force_t cut2, int n_owned,
    fixed_t* fx, fixed_t* fy, fixed_t* fz, ForceFault* fault, acc_t* partial,
    int partial_stride) {
    using Param = typename Term::Param;
    constexpr unsigned all = tile::kFullWarp;
    const ForceBox box = box_arg.current().as<force_t>();
    const int num_pairs = *tile_count;
    const int warps = blockDim.x / 32;
    const int chunk = blockIdx.x;
    if (chunk * warps >= num_pairs)
        return;
    const int lane = threadIdx.x % 32;
    const int p = chunk * warps + threadIdx.x / 32;
    acc_t e = 0, v[6] = {};
    DeferredForceAdds<6> adds;
    if (p < num_pairs) {
        const int ci = packed_ci[p];
        const int si = ci * 32 + lane;
        const std::size_t slot = static_cast<std::size_t>(p) * 32 + lane;
        const int sj = packed_j[slot];
        const int ai = index[si], bj = sj >= 0 ? index[sj] : -1;
        force_t ax = cx[si], ay = cy[si], az = cz[si];
        force_t bx = 0, by = 0, bz = 0;
        const Param pa = params[si];
        Param pb{};
        if (sj >= 0) {
            bx = cx[sj];
            by = cy[sj];
            bz = cz[sj];
            pb = params[sj];
        }
        const unsigned mask = masks[slot];
        force_t fi[3] = {}, fj[3] = {};
        if (__any_sync(all, mask != 0)) {
            bool single_copy = false;
            if constexpr (Ortho) {
                const ClusterImage image = images[ci];
                single_copy = image.single;
                if (single_copy) {
                    wrap_to(ax, ay, az, image, box);
                    if (bj >= 0)
                        wrap_to(bx, by, bz, image, box);
                }
            }
            for (int step = 0; step < 32; ++step) {
                const int jlane = (lane + step) & 31;
                const int aj = __shfl_sync(all, bj, jlane);
                force_t dx = __shfl_sync(all, bx, jlane) - ax;
                force_t dy = __shfl_sync(all, by, jlane) - ay;
                force_t dz = __shfl_sync(all, bz, jlane) - az;
                const Param pj = Term::shfl(pb, jlane);
                if constexpr (Ortho) {
                    if (!single_copy) {
                        dx -= box.cell[0] * nearbyint(dx * box.inv_l[0]);
                        dy -= box.cell[4] * nearbyint(dy * box.inv_l[1]);
                        dz -= box.cell[8] * nearbyint(dz * box.inv_l[2]);
                    }
                } else {
                    box.min_image(dx, dy, dz);
                }
                const force_t r2 = dx * dx + dy * dy + dz * dz;
                force_t contribution[3] = {};
                if (ai >= 0 && aj >= 0 && ai != aj && (mask & (1u << jlane))
                    && r2 > 0 && r2 < cut2) {
                    const tile::Contribution c = term.pair(r2, pa, pj);
                    const force_t fr = c.fpair;
                    contribution[0] = fr * dx;
                    contribution[1] = fr * dy;
                    contribution[2] = fr * dz;
                    for (int d = 0; d < 3; ++d)
                        fi[d] -= contribution[d];
                    if constexpr (Energy || Virial) {
                        const acc_t w =
                            acc_t(0.5)
                            * acc_t(int(ai < n_owned) + int(aj < n_owned));
                        if constexpr (Energy)
                            e += w * acc_t(c.energy);
                        if constexpr (Virial) {
                            v[0] += w * acc_t(fr * dx * dx);
                            v[1] += w * acc_t(fr * dy * dy);
                            v[2] += w * acc_t(fr * dz * dz);
                            v[3] += w * acc_t(fr * dx * dy);
                            v[4] += w * acc_t(fr * dx * dz);
                            v[5] += w * acc_t(fr * dy * dz);
                        }
                    }
                }
                // Send each contribution back to its original j lane. Tail
                // lanes and excluded pairs still reach every shuffle.
                const int source = (lane - step) & 31;
                for (int d = 0; d < 3; ++d)
                    fj[d] += __shfl_sync(all, contribution[d], source);
            }
        }
        // Issue the six atomics now; their wraparound check runs after the
        // energy/virial reduction below.
        fixed_t* const columns[3] = {fx, fy, fz};
        const int row_i = ai < n_owned ? ai : -1;
        const int row_j = bj < n_owned ? bj : -1;
#pragma unroll
        for (int d = 0; d < 3; ++d) {
            adds.add(d, columns[d], row_i, fi[d], fault);
            adds.add(3 + d, columns[d], row_j, fj[d], fault);
        }
    }
    if constexpr (Energy || Virial)
        tally<Virial, kTileBlock>(e, v, partial, partial_stride, chunk);
    adds.check(fault);
}

/// @brief Host side of the engine for one term: the packed tiles, the
///        slot-ordered parameters and the energy/virial partials of one
///        pair style, kept current by a hook in the neighbor rebuild.
template <class Term>
class ClusterTileEngine {
  public:
    ClusterTileEngine() = default;
    ClusterTileEngine(const ClusterTileEngine&) = delete;
    ClusterTileEngine& operator=(const ClusterTileEngine&) = delete;
    ~ClusterTileEngine() { destroy(); }

    /// @brief Size everything for the frame capacity and the neighbor
    ///        list's CSR reserve, and hook packing and the parameter gather
    ///        into every rebuild of that list. Re-callable.
    /// @param ctx Build context (the list must be built and published).
    /// @param term Term with its device parameter columns.
    /// @param extra_partials Partial slots reserved after the tiles for the
    ///        style's own correction pass.
    void build(const BuildContext& ctx, const Term& term, int extra_partials) {
        destroy();
        const PairList* list = ctx.runtime.md.pair_list;
        if (!list)
            throw std::invalid_argument(
                "cuda pair tile engine: build NeighborList before the pair "
                "style");
        if (!list->hooks)
            throw std::invalid_argument("cuda pair tile engine: the neighbor "
                                        "list publishes no rebuild hooks");
        list_ = list;
        term_ = term;
        fault_ = ctx.frame.force_fault;
        slots_ = ((ctx.frame.capacity() + 31) / 32) * 32;
        const int max_clusters = slots_ / 32;
        if (slots_)
            ATV_CUDA_CHECK(cudaMalloc(
                &params_, std::size_t(slots_) * sizeof(typename Term::Param)));
        ATV_CUDA_CHECK(cudaMalloc(&packed_.valid, sizeof(int)));
        ATV_CUDA_CHECK(cudaMemset(packed_.valid, 0, sizeof(int)));
        // A row contributes at most 32 columns per CSR entry, hence its
        // packed tile count cannot exceed its entry count: the CSR reserve
        // bounds the packed arrays and the tile partials.
        packed_.capacity = list->cluster_pair_capacity;
        ATV_CUDA_CHECK(
            cudaMalloc(&packed_.counts, (max_clusters + 1) * sizeof(int)));
        ATV_CUDA_CHECK(
            cudaMalloc(&packed_.offsets, (max_clusters + 1) * sizeof(int)));
        // No tile before the first rebuild packs some.
        ATV_CUDA_CHECK(
            cudaMemset(packed_.offsets, 0, (max_clusters + 1) * sizeof(int)));
        packed_.scan_bytes = packed_tiles_scan_bytes(max_clusters);
        if (packed_.scan_bytes)
            ATV_CUDA_CHECK(cudaMalloc(&packed_.scan_temp, packed_.scan_bytes));
        if (packed_.capacity) {
            const std::size_t cap = static_cast<std::size_t>(packed_.capacity);
            ATV_CUDA_CHECK(cudaMalloc(&packed_.ci, cap * sizeof(int)));
            ATV_CUDA_CHECK(cudaMalloc(&packed_.j, cap * 32 * sizeof(int)));
            ATV_CUDA_CHECK(
                cudaMalloc(&packed_.masks, cap * 32 * sizeof(std::uint32_t)));
        }
        tile_slots_ = static_cast<int>((static_cast<long long>(packed_.capacity)
                                        + warps_per_block(kTileBlock) - 1)
                                       / warps_per_block(kTileBlock));
        // One block per reserved chunk; blocks past the live (device) count
        // exit. A resident striding grid measured 15% slower on GH200.
        tile_grid_ = std::max(1, tile_slots_);
        extra_ = extra_partials;
        partial_capacity_ = tile_slots_ + extra_;
        if (partial_capacity_ > 0)
            ATV_CUDA_CHECK(cudaMalloc(
                &partial_, static_cast<std::size_t>(partial_capacity_) * 7
                               * sizeof(acc_t)));
        hooks_ = list->hooks;
        hooks_->add(this, [this](void* stream) {
            const auto s = static_cast<cudaStream_t>(stream);
            pack_tiles(*list_, packed_, fault_, s);
            gather(s);
        });
    }

    /// @brief Re-gather the slot parameters (an input edit changed them).
    void gather(cudaStream_t stream) const {
        const int slots = list_ ? list_->num_clusters * 32 : 0;
        if (slots <= 0 || !params_)
            return;
        gather_tile_params<Term>
            <<<grid_size(slots), kDefaultBlock, 0, stream>>>(
                term_, list_->cl_orig_index, slots, params_);
        ATV_CUDA_CHECK(cudaGetLastError());
    }

    /// @brief Enqueue the tile force of the step.
    /// @param ctx Launch context (the list of build() is the published one).
    /// @param cut2 Outer squared cutoff of the term [Å²].
    template <bool Energy, bool Virial>
    void tiles(const LaunchContext& ctx, force_t cut2) const {
        const PairList* p = ctx.runtime.md.pair_list;
        if (p != list_ || p->hooks != hooks_)
            throw std::runtime_error("cuda pair tile engine: neighbor list "
                                     "replaced after pair build");
        if (p->num_clusters * 32 > slots_)
            throw std::runtime_error(
                "cuda pair tile engine: Frame grew after build");
        if (!tile_slots_ || p->num_clusters == 0)
            return;
        const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
        auto& f = ctx.frame;
        const int n_owned = f.n_local();
        auto run = [&](auto ortho) {
            constexpr bool O = decltype(ortho)::value;
            cluster_tile<Term, O, Energy, Virial>
                <<<tile_grid_, kTileBlock, 0, stream>>>(
                    term_, p->cl_x, p->cl_y, p->cl_z, params_, p->cl_orig_index,
                    packed_.ci, packed_.j, packed_.masks, tile_count_(),
                    ctx.runtime.host_box, p->cl_image, cut2, n_owned,
                    f.atoms.fx, f.atoms.fy, f.atoms.fz, f.force_fault, partial_,
                    partial_capacity_);
        };
        if (ctx.runtime.host_box.ortho)
            run(std::true_type{});
        else
            run(std::false_type{});
        ATV_CUDA_CHECK(cudaGetLastError());
    }

    /// @brief Fold the tile partials and the extra slots into the frame.
    template <bool Energy, bool Virial>
    void reduce(const LaunchContext& ctx, EnergyTerm energy,
                acc_t* stress) const {
        if constexpr (Energy || Virial) {
            if (!partial_capacity_ || !list_ || list_->num_clusters == 0)
                return;
            reduce_partials<Energy, Virial>
                <<<grid_size(partial_capacity_), kDefaultBlock, 0,
                   static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
                    partial_, tile_count_(), tile_slots_, extra_,
                    partial_capacity_, energy, stress);
            ATV_CUDA_CHECK(cudaGetLastError());
        }
    }

    acc_t* partial() const { return partial_; }
    int partial_stride() const { return partial_capacity_; }
    /// First extra partial slot (after the tile reserve).
    int extra_offset() const { return tile_slots_; }

    /// @brief Packed tiles of the latest rebuild (synchronizes).
    int packed_tiles() const {
        if (!packed_.offsets || !list_)
            return 0;
        int count = 0;
        ATV_CUDA_CHECK(cudaMemcpy(&count, tile_count_(), sizeof(int),
                                  cudaMemcpyDeviceToHost));
        return count;
    }
    /// @brief CSR entries with an upper-triangle bit (synchronizes).
    int valid_tiles() const {
        if (!packed_.valid)
            return 0;
        int count = 0;
        ATV_CUDA_CHECK(cudaMemcpy(&count, packed_.valid, sizeof(int),
                                  cudaMemcpyDeviceToHost));
        return count;
    }

    /// @brief Detach the hook and free everything.
    void destroy() {
        if (hooks_)
            hooks_->remove(this);
        hooks_ = nullptr;
        list_ = nullptr;
        cudaFree(params_);
        cudaFree(partial_);
        cudaFree(packed_.counts);
        cudaFree(packed_.offsets);
        cudaFree(packed_.ci);
        cudaFree(packed_.j);
        cudaFree(packed_.masks);
        cudaFree(packed_.scan_temp);
        cudaFree(packed_.valid);
        params_ = nullptr;
        partial_ = nullptr;
        packed_ = {};
        slots_ = tile_slots_ = tile_grid_ = extra_ = partial_capacity_ = 0;
    }

  private:
    const int* tile_count_() const {
        return packed_.offsets + list_->num_clusters;
    }

    const PairList* list_ = nullptr;
    RebuildHooks* hooks_ = nullptr;
    Term term_{};
    ForceFault* fault_ = nullptr;
    typename Term::Param* params_ = nullptr;
    PackedTileWorkspace packed_{};
    acc_t* partial_ = nullptr; ///< SoA: energy, then six virial planes.
    int slots_ = 0;
    int tile_slots_ = 0;
    int tile_grid_ = 0;
    int extra_ = 0;
    int partial_capacity_ = 0;
};

} // namespace
} // namespace ATV::cuda::internal
