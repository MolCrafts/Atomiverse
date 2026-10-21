/// @file cuda/potentials/pair_lj_cut_coul_long.cu
/// @brief Fused packed-J tile force and sparse Ewald exception correction.
/// Algorithm reference: OpenMM nonbonded.cu, findInteractingBlocks.cu and
/// nonbondedExceptions.cc,
/// commit 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e. Uses the published ideas
/// of parameter fusion, warp data exchange and a separate exception pass;
/// implementation is ATV-native, with no copied OpenMM source.
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include "internal/packed_tiles.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ATV::cuda {
namespace {
constexpr int kTileBlock = 128;
constexpr work_t kWorkInfinity = std::numeric_limits<work_t>::infinity();
constexpr work_t kWorkEpsilon = std::numeric_limits<work_t>::epsilon();

// OpenMM packs only J atoms that interact with an I cluster, crossing the
// original J-cluster boundaries. Here the exact baked masks are the input;
// no geometry or exclusion decision is repeated by the packer.
__device__ unsigned upper_mask(const PairList& source, int p, int ci,
                               int lane) {
    const int cj = source.cl_col_idx[p];
    unsigned bits = openmm_mask_jbits(source.cl_mask, p, lane);
    if (cj < ci)
        return 0;
    if (cj == ci)
        bits &= lane == 31 ? 0u : (0xffffffffu << (lane + 1));
    return bits;
}

__device__ unsigned column_union(unsigned bits) {
    for (int shift = 16; shift > 0; shift /= 2)
        bits |= __shfl_xor_sync(0xffffffffu, bits, shift);
    return bits;
}

__global__ void count_packed_tiles(PairList source, int* counts) {
    const int ci = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (ci > source.num_clusters)
        return;
    int columns = 0;
    if (ci < source.num_clusters)
        for (int p = source.cl_row_ptr[ci]; p < source.cl_row_ptr[ci + 1]; ++p)
            columns += __popc(column_union(upper_mask(source, p, ci, lane)));
    if (lane == 0)
        counts[ci] = (columns + 31) / 32;
}

__global__ void fill_packed_tiles(PairList source,
                                  internal::PackedTileWorkspace out) {
    const int ci = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (ci >= source.num_clusters)
        return;
    int tile = out.offsets[ci], column = 0;
    unsigned packed_bits = 0;
    for (int p = source.cl_row_ptr[ci]; p < source.cl_row_ptr[ci + 1]; ++p) {
        const unsigned row = upper_mask(source, p, ci, lane);
        unsigned columns = column_union(row);
        while (columns) {
            const int j = __ffs(columns) - 1;
            columns &= columns - 1;
            if (row & (1u << j))
                packed_bits |= 1u << column;
            if (lane == column)
                out.j[static_cast<std::size_t>(tile) * 32 + lane] =
                    source.cl_col_idx[p] * 32 + j;
            if (++column == 32) {
                out.masks[static_cast<std::size_t>(tile) * 32 + lane] =
                    packed_bits;
                if (lane == 0)
                    out.ci[tile] = ci;
                ++tile;
                column = 0;
                packed_bits = 0;
            }
        }
    }
    if (column) {
        const std::size_t slot = static_cast<std::size_t>(tile) * 32 + lane;
        if (lane >= column)
            out.j[slot] = -1;
        out.masks[slot] = packed_bits;
        if (lane == 0)
            out.ci[tile] = ci;
    }
}

struct Atom {
    work_t x, y, z, q, sigma_half, epsilon_root;
};
struct Exception {
    int i, j;
    work_t lj_scale, coul_scale;
};
struct ClusterImage {
    work_t center[3];
    int single;
};

// OpenMM's singlePeriodicCopy criterion, evaluated from live I coordinates
// rather than rebuild-time bounds. Thus a reused Hilbert cluster needs no
// assumption about its motion within the neighbor skin. Wide clusters safely
// keep the per-pair minimum image calculation.
__device__ __forceinline__ ClusterImage cluster_image(const Atom& a, int ai,
                                                      const Box& box,
                                                      work_t cut2) {
    const work_t position[3] = {a.x, a.y, a.z};
    ClusterImage image{};
    const work_t cutoff = sqrt(cut2);
#pragma unroll
    for (int d = 0; d < 3; ++d) {
        work_t lo = ai >= 0 ? position[d] : kWorkInfinity;
        work_t hi = ai >= 0 ? position[d] : -kWorkInfinity;
#pragma unroll
        for (int shift = 16; shift > 0; shift /= 2) {
            lo = fmin(lo, __shfl_xor_sync(0xffffffffu, lo, shift));
            hi = fmax(hi, __shfl_xor_sync(0xffffffffu, hi, shift));
        }
        const work_t length = work_t(box.cell[4 * d]);
        const work_t halfwidth = (hi - lo) * work_t(0.5);
        const work_t margin = work_t(32) * kWorkEpsilon
                              * (fabs(lo) + fabs(hi) + fabs(length) + cutoff);
        // Every cutoff neighbor then lies strictly inside the single box
        // centered here. The guard is warp-uniform, including padded lanes.
        if (!(halfwidth + cutoff + margin < length * work_t(0.5)))
            return image;
        image.center[d] = lo + halfwidth;
    }
    image.single = 1;
    return image;
}

__device__ __forceinline__ void
apply_cluster_image(Atom& a, const ClusterImage& image, const Box& box) {
    a.x -= box.cell[0] * nearbyint((a.x - image.center[0]) * box.inv_l[0]);
    a.y -= box.cell[4] * nearbyint((a.y - image.center[1]) * box.inv_l[1]);
    a.z -= box.cell[8] * nearbyint((a.z - image.center[2]) * box.inv_l[2]);
}

__global__ void gather(const int* index, int slots, const real_t* x,
                       const real_t* y, const real_t* z, const real_t* q,
                       const work_t* sigma, const work_t* epsilon, Atom* sorted,
                       ClusterImage* images, Box box, work_t cut2) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= slots)
        return;
    const int i = index[s];
    const Atom atom =
        i < 0 ? Atom{} : Atom{x[i], y[i], z[i], q[i], sigma[i], epsilon[i]};
    sorted[s] = atom;
    // Slots are padded to whole warps, so all lanes of every live cluster
    // participate. Compute once per cluster each step, not once per tile.
    if (box.ortho) {
        const ClusterImage image = cluster_image(atom, i, box, cut2);
        if (threadIdx.x % 32 == 0)
            images[s / 32] = image;
    }
}

// Like OpenMM's nonbonded energy buffer, each producer writes a disjoint
// partial. A separate reduction avoids serializing every force block on one
// system energy/stress address. All launched blocks overwrite their slot.
template <bool Virial>
__device__ void tally(work_t e, const work_t* v, prec_t* partial,
                      int partial_stride, int partial_offset) {
    __shared__ work_t scratch[kDefaultBlock / 32];
    work_t vals[7] = {e, v[0], v[1], v[2], v[3], v[4], v[5]}, out[7];
    if constexpr (Virial) {
        block_reduce_sum_n<7>(vals, scratch, out);
    } else {
        out[0] = block_reduce_sum(e, scratch);
    }
    if (threadIdx.x == 0) {
        const std::size_t slot = partial_offset + blockIdx.x;
        partial[slot] = prec_t(out[0]);
        if constexpr (Virial)
            for (int d = 0; d < 6; ++d)
                partial[static_cast<std::size_t>(d + 1) * partial_stride
                        + slot] = prec_t(out[d + 1]);
    }
}

template <bool Virial>
__global__ void reduce_partials(const prec_t* partial, int count, int stride,
                                double* energy, double* stress) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ prec_t scratch[kDefaultBlock / 32];
    if constexpr (Virial) {
        prec_t values[7] = {}, reduced[7];
        if (i < count)
            for (int d = 0; d < 7; ++d)
                values[d] = partial[static_cast<std::size_t>(d) * stride + i];
        block_reduce_sum_n<7>(values, scratch, reduced);
        if (threadIdx.x == 0) {
            atomicAdd(energy, reduced[0]);
            for (int d = 0; d < 6; ++d)
                atomicAdd(stress + d, reduced[d + 1]);
        }
    } else {
        const prec_t value = i < count ? partial[i] : prec_t(0);
        const prec_t reduced = block_reduce_sum(value, scratch);
        if (threadIdx.x == 0)
            atomicAdd(energy, reduced);
    }
}

template <bool Ortho, bool Virial>
__global__ __launch_bounds__(kTileBlock, 4) void fused_tile(
    const Atom* atoms, const int* index, const int* packed_ci,
    const int* packed_j, const std::uint32_t* masks, int num_pairs, Box box,
    const ClusterImage* images, work_t alpha, work_t cut2, real_t* fx,
    real_t* fy, real_t* fz, prec_t* partial, int partial_stride) {
    // A warp owns one I cluster and 32 packed J atoms. Baked masks already
    // contain only upper-triangle interactions. Floating
    // atomics retain the force range needed for initially clashing systems.
    const int p = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    work_t e = 0, v[6] = {};
    if (p < num_pairs) {
        const int si = packed_ci[p] * 32 + lane;
        const std::size_t slot = static_cast<std::size_t>(p) * 32 + lane;
        const int sj = packed_j[slot];
        const int ai = index[si], bj = sj >= 0 ? index[sj] : -1;
        Atom a = atoms[si], b = sj >= 0 ? atoms[sj] : Atom{};
        const unsigned mask = masks[slot];
        work_t fi[3] = {}, fj[3] = {};
        if (__any_sync(0xffffffffu, mask != 0)) {
            bool single_copy = false;
            if constexpr (Ortho) {
                const ClusterImage image = images[packed_ci[p]];
                single_copy = image.single;
                if (single_copy) {
                    apply_cluster_image(a, image, box);
                    if (bj >= 0)
                        apply_cluster_image(b, image, box);
                }
            }
            for (int step = 0; step < 32; ++step) {
                const int jlane = (lane + step) & 31;
                work_t dx = __shfl_sync(0xffffffffu, b.x, jlane) - a.x;
                work_t dy = __shfl_sync(0xffffffffu, b.y, jlane) - a.y;
                work_t dz = __shfl_sync(0xffffffffu, b.z, jlane) - a.z;
                const work_t qj = __shfl_sync(0xffffffffu, b.q, jlane);
                const work_t sig =
                    a.sigma_half
                    + __shfl_sync(0xffffffffu, b.sigma_half, jlane);
                const work_t eps =
                    a.epsilon_root
                    * __shfl_sync(0xffffffffu, b.epsilon_root, jlane);
                if constexpr (Ortho) {
                    if (!single_copy) {
                        dx -= box.cell[0] * nearbyint(dx * box.inv_l[0]);
                        dy -= box.cell[4] * nearbyint(dy * box.inv_l[1]);
                        dz -= box.cell[8] * nearbyint(dz * box.inv_l[2]);
                    }
                } else
                    box.min_image(dx, dy, dz);
                const work_t r2 = dx * dx + dy * dy + dz * dz;
                work_t contribution[3] = {};
                // Packed masks exclude self pairs and every padded J lane.
                if (ai >= 0 && (mask & (1u << jlane)) && r2 > 0 && r2 < cut2) {
                    const work_t inv = rsqrt(r2), inv2 = inv * inv,
                                 ar = alpha * (r2 * inv);
                    const work_t sr2 = sig * sig * inv2, sr6 = sr2 * sr2 * sr2;
                    const work_t ex = exp(-ar * ar), er = erfcx(ar) * ex;
                    const work_t qq = work_t(units::coulomb) * a.q * qj * inv;
                    const work_t fr =
                        (eps * (12 * sr6 - 6) * sr6
                         + qq * (er + work_t(1.1283791670955126) * (ar * ex)))
                        * inv2;
                    contribution[0] = fr * dx;
                    contribution[1] = fr * dy;
                    contribution[2] = fr * dz;
                    for (int d = 0; d < 3; ++d)
                        fi[d] -= contribution[d];
                    e += eps * sr6 * (sr6 - 1) + qq * er;
                    if constexpr (Virial) {
                        v[0] += fr * dx * dx;
                        v[1] += fr * dy * dy;
                        v[2] += fr * dz * dz;
                        v[3] += fr * dx * dy;
                        v[4] += fr * dx * dz;
                        v[5] += fr * dy * dz;
                    }
                }
                // Send each contribution back to its original j lane. Tail
                // lanes and excluded pairs still reach every shuffle.
                const int source = (lane - step) & 31;
                for (int d = 0; d < 3; ++d)
                    fj[d] += __shfl_sync(0xffffffffu, contribution[d], source);
            }
        }
        if (ai >= 0) {
            atomicAdd(fx + ai, real_t(fi[0]));
            atomicAdd(fy + ai, real_t(fi[1]));
            atomicAdd(fz + ai, real_t(fi[2]));
        }
        if (bj >= 0) {
            atomicAdd(fx + bj, real_t(fj[0]));
            atomicAdd(fy + bj, real_t(fj[1]));
            atomicAdd(fz + bj, real_t(fj[2]));
        }
    }
    tally<Virial>(e, v, partial, partial_stride, 0);
}

template <bool Virial>
__global__ void exceptions(const Exception* pairs, int count, const real_t* x,
                           const real_t* y, const real_t* z, const real_t* q,
                           const work_t* sigma, const work_t* epsilon, Box box,
                           work_t alpha, real_t* fx, real_t* fy, real_t* fz,
                           prec_t* partial, int partial_stride,
                           int partial_offset) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    work_t e = 0, v[6] = {};
    if (p < count) {
        const Exception t = pairs[p];
        const int i = t.i, j = t.j;
        work_t dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
        box.min_image(dx, dy, dz);
        const work_t r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 0) {
            const work_t inv2 = 1 / r2, inv = 1 / sqrt(r2), ar = alpha / inv;
            const work_t sig = sigma[i] + sigma[j], sr2 = sig * sig * inv2,
                         sr6 = sr2 * sr2 * sr2;
            const work_t eps = t.lj_scale * epsilon[i] * epsilon[j];
            const work_t qq = work_t(units::coulomb) * q[i] * q[j] * inv;
            const work_t screened = t.coul_scale - erf(ar);
            const work_t fr =
                (eps * (12 * sr6 - 6) * sr6
                 + qq
                       * (screened
                          + work_t(1.1283791670955126) * ar * exp(-ar * ar)))
                * inv2;
            const work_t f[3] = {fr * dx, fr * dy, fr * dz};
            atomicAdd(fx + i, real_t(-f[0]));
            atomicAdd(fy + i, real_t(-f[1]));
            atomicAdd(fz + i, real_t(-f[2]));
            atomicAdd(fx + j, real_t(f[0]));
            atomicAdd(fy + j, real_t(f[1]));
            atomicAdd(fz + j, real_t(f[2]));
            e = eps * sr6 * (sr6 - 1) + qq * screened;
            if constexpr (Virial) {
                v[0] = fr * dx * dx;
                v[1] = fr * dy * dy;
                v[2] = fr * dz * dz;
                v[3] = fr * dx * dy;
                v[4] = fr * dx * dz;
                v[5] = fr * dy * dz;
            }
        }
    }
    tally<Virial>(e, v, partial, partial_stride, partial_offset);
}

template <class T>
void upload(T*& device, const std::vector<T>& host) {
    if (host.empty())
        return;
    ATV_CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    ATV_CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
}
} // namespace

std::size_t internal::packed_tiles_scan_bytes(int num_clusters) {
    std::size_t bytes = 0;
    ATV_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, static_cast<const int*>(nullptr),
        static_cast<int*>(nullptr), num_clusters + 1));
    return bytes;
}

int internal::pack_tiles(const PairList& source,
                         const PackedTileWorkspace& work, cudaStream_t stream) {
    count_packed_tiles<<<(source.num_clusters + 4) / 4, kTileBlock, 0,
                         stream>>>(source, work.counts);
    ATV_CUDA_CHECK(cudaGetLastError());
    std::size_t bytes = work.scan_bytes;
    ATV_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        work.scan_temp, bytes, work.counts, work.offsets,
        source.num_clusters + 1, stream));
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(work.host_count, work.offsets + source.num_clusters,
                        sizeof(int), cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int count = *work.host_count;
    if (count > work.capacity)
        throw std::runtime_error(
            "cuda::PairLJCutCoulLong: packed tiles exceed build-time reserve");
    if (count) {
        fill_packed_tiles<<<(source.num_clusters + 3) / 4, kTileBlock, 0,
                            stream>>>(source, work);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
    return count;
}

struct PairLJCutCoulLong::State {
    Atom* sorted = nullptr;
    ClusterImage* images = nullptr;
    work_t* sigma = nullptr;
    work_t* epsilon = nullptr;
    int* offsets = nullptr;
    int* partners = nullptr;
    Exception* pairs = nullptr;
    int count = 0, slots = 0;
    prec_t* partial = nullptr; // SoA: energy followed by six virial planes.
    int partial_capacity = 0;
    int exception_blocks = 0;
    internal::PackedTileWorkspace packed;
    int packed_count = 0;
    std::uint64_t packed_generation = 0;
    ~State() {
        ATV_CUDA_CHECK(cudaFree(sorted));
        ATV_CUDA_CHECK(cudaFree(images));
        ATV_CUDA_CHECK(cudaFree(sigma));
        ATV_CUDA_CHECK(cudaFree(epsilon));
        ATV_CUDA_CHECK(cudaFree(offsets));
        ATV_CUDA_CHECK(cudaFree(partners));
        ATV_CUDA_CHECK(cudaFree(pairs));
        ATV_CUDA_CHECK(cudaFree(partial));
        ATV_CUDA_CHECK(cudaFree(packed.counts));
        ATV_CUDA_CHECK(cudaFree(packed.offsets));
        ATV_CUDA_CHECK(cudaFree(packed.ci));
        ATV_CUDA_CHECK(cudaFree(packed.j));
        ATV_CUDA_CHECK(cudaFree(packed.masks));
        ATV_CUDA_CHECK(cudaFree(packed.scan_temp));
        if (packed.host_count)
            ATV_CUDA_CHECK(cudaFreeHost(packed.host_count));
    }
};
PairLJCutCoulLong::PairLJCutCoulLong(std::vector<double> epsilon,
                                     std::vector<double> sigma, double alpha,
                                     double cutoff)
    : Pair(cutoff), epsilon_(std::move(epsilon)), sigma_(std::move(sigma)),
      alpha_(alpha) {
    if (epsilon_.empty() || epsilon_.size() != sigma_.size()
        || !std::isfinite(alpha) || alpha <= 0 || !std::isfinite(cutoff)
        || cutoff <= 0)
        throw std::invalid_argument(
            "cuda::PairLJCutCoulLong: equal nonempty LJ tables and positive "
            "alpha/cutoff required");
    for (std::size_t t = 0; t < epsilon_.size(); ++t)
        if (!std::isfinite(epsilon_[t]) || epsilon_[t] < 0
            || !std::isfinite(sigma_[t]) || sigma_[t] <= 0)
            throw std::invalid_argument("cuda::PairLJCutCoulLong: finite "
                                        "epsilon>=0 and sigma>0 required");
}
PairLJCutCoulLong::~PairLJCutCoulLong() = default;
PairLJCutCoulLong::PairLJCutCoulLong(PairLJCutCoulLong&&) noexcept = default;
PairLJCutCoulLong&
PairLJCutCoulLong::operator=(PairLJCutCoulLong&&) noexcept = default;
PairLJCutCoulLong& PairLJCutCoulLong::set_special_bonds_amber() & {
    amber_ = true;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_special_bonds_amber() && {
    amber_ = true;
    return std::move(*this);
}

void PairLJCutCoulLong::build(const BuildContext& ctx) {
    const auto& f = ctx.frame;
    if (f.n_ghost() != 0)
        throw std::invalid_argument(
            "cuda::PairLJCutCoulLong: single GPU required");
    state_ = std::make_unique<State>();
    auto& s = *state_;
    const int n = f.n_atoms();
    std::vector<int> types(n);
    ATV_CUDA_CHECK(cudaMemcpy(types.data(), f.atoms.type, n * sizeof(int),
                              cudaMemcpyDeviceToHost));
    std::vector<work_t> sig(n), eps(n);
    for (int i = 0; i < n; ++i) {
        const int t = types[i];
        if (t < 0 || std::size_t(t) >= epsilon_.size())
            throw std::invalid_argument(
                "cuda::PairLJCutCoulLong: atom type outside coefficient table");
        sig[i] = work_t(sigma_[t] * 0.5);
        eps[i] = work_t(2 * std::sqrt(epsilon_[t]));
    }
    upload(s.sigma, sig);
    upload(s.epsilon, eps);
    s.slots = ((f.capacity() + 31) / 32) * 32;
    if (s.slots) {
        ATV_CUDA_CHECK(cudaMalloc(&s.sorted, s.slots * sizeof(Atom)));
        ATV_CUDA_CHECK(cudaMalloc(&s.images, std::size_t(s.slots / 32)
                                                 * sizeof(ClusterImage)));
    }
    std::vector<std::vector<int>> graph(n);
    if (amber_) {
        const int nb = f.n_bonds();
        std::vector<int> bi(nb), bj(nb);
        if (nb) {
            ATV_CUDA_CHECK(cudaMemcpy(bi.data(), f.bonds.atomi,
                                      nb * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            ATV_CUDA_CHECK(cudaMemcpy(bj.data(), f.bonds.atomj,
                                      nb * sizeof(int),
                                      cudaMemcpyDeviceToHost));
        }
        for (int b = 0; b < nb; ++b) {
            if (bi[b] < 0 || bj[b] < 0 || bi[b] >= n || bj[b] >= n)
                throw std::invalid_argument(
                    "cuda::PairLJCutCoulLong: bond index outside Frame");
            graph[bi[b]].push_back(bj[b]);
            graph[bj[b]].push_back(bi[b]);
        }
    }
    std::vector<int> offsets(n + 1), partners;
    std::vector<Exception> pairs;
    for (int i = 0; i < n; ++i) {
        std::map<int, int> depth;
        depth[i] = 0;
        std::vector<int> queue{i};
        for (std::size_t p = 0; p < queue.size(); ++p) {
            const int u = queue[p], d = depth[u];
            if (d == 3)
                continue;
            for (int j : graph[u])
                if (!depth.count(j)) {
                    depth[j] = d + 1;
                    queue.push_back(j);
                }
        }
        depth.erase(i);
        for (auto [j, d] : depth) {
            partners.push_back(j);
            if (i < j)
                pairs.push_back({i, j, work_t(d == 3 ? 0.5 : 0),
                                 work_t(d == 3 ? 5.0 / 6.0 : 0)});
        }
        offsets[i + 1] = int(partners.size());
    }
    upload(s.offsets, offsets);
    upload(s.partners, partners);
    upload(s.pairs, pairs);
    s.count = int(pairs.size());
    const PairList* list = ctx.runtime.md.pair_list;
    if (!list)
        throw std::invalid_argument(
            "cuda::PairLJCutCoulLong: build NeighborList before pair scratch");
    auto& packed = s.packed;
    ATV_CUDA_CHECK(cudaMallocHost(&packed.host_count, sizeof(int)));
    // A row contributes at most 32 columns per CSR entry, hence its packed
    // tile count cannot exceed its original entry count. The existing CSR
    // reserve is also an upper bound for all packed arrays and partials.
    packed.capacity = list->cluster_pair_capacity;
    const int max_clusters = s.slots / 32;
    ATV_CUDA_CHECK(
        cudaMalloc(&packed.counts, (max_clusters + 1) * sizeof(int)));
    ATV_CUDA_CHECK(
        cudaMalloc(&packed.offsets, (max_clusters + 1) * sizeof(int)));
    packed.scan_bytes = internal::packed_tiles_scan_bytes(max_clusters);
    if (packed.scan_bytes)
        ATV_CUDA_CHECK(cudaMalloc(&packed.scan_temp, packed.scan_bytes));
    if (packed.capacity) {
        ATV_CUDA_CHECK(
            cudaMalloc(&packed.ci, std::size_t(packed.capacity) * sizeof(int)));
        ATV_CUDA_CHECK(cudaMalloc(&packed.j, std::size_t(packed.capacity) * 32
                                                 * sizeof(int)));
        ATV_CUDA_CHECK(
            cudaMalloc(&packed.masks, std::size_t(packed.capacity) * 32
                                          * sizeof(std::uint32_t)));
    }
    const int tile_blocks =
        static_cast<int>((static_cast<long long>(list->cluster_pair_capacity)
                          + warps_per_block(kTileBlock) - 1)
                         / warps_per_block(kTileBlock));
    s.exception_blocks = s.count > 0 ? grid_size(s.count) : 0;
    s.partial_capacity = tile_blocks + s.exception_blocks;
    if (s.partial_capacity > 0)
        ATV_CUDA_CHECK(
            cudaMalloc(&s.partial, static_cast<std::size_t>(s.partial_capacity)
                                       * 7 * sizeof(prec_t)));
    // Ordinary tile force excludes every Amber exception. The sparse pass
    // restores scaled LJ and the real-space Ewald exclusion correction.
    ctx.runtime.md.excl_offsets = s.offsets;
    ctx.runtime.md.excl_partners = s.partners;
    ctx.runtime.md.s_excl = 0.0;
    ctx.runtime.md.exclusions_baked = false;
}
void PairLJCutCoulLong::compute(const LaunchContext& ctx) {
    const PairList* p = ctx.runtime.md.pair_list;
    if (!p || p->num_clusters == 0)
        return;
    if (p->num_cluster_pairs <= 0)
        throw std::runtime_error("cuda::PairLJCutCoulLong: run NeighborList "
                                 "before force evaluation");
    auto& s = *state_;
    auto& f = ctx.frame;
    if (!ctx.runtime.md.exclusions_baked)
        throw std::runtime_error(
            "cuda::PairLJCutCoulLong: NeighborList must bake exception masks "
            "before force evaluation");
    const int slots = p->num_clusters * 32;
    if (slots > s.slots)
        throw std::runtime_error(
            "cuda::PairLJCutCoulLong: Frame grew after build");
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    if (p->rebuild_generation == 0)
        throw std::runtime_error(
            "cuda::PairLJCutCoulLong: neighbor masks have not been published");
    if (s.packed_generation != p->rebuild_generation) {
        s.packed_count = internal::pack_tiles(*p, s.packed, stream);
        s.packed_generation = p->rebuild_generation;
    }
    const int tile_blocks =
        static_cast<int>((static_cast<long long>(s.packed_count)
                          + warps_per_block(kTileBlock) - 1)
                         / warps_per_block(kTileBlock));
    const int partial_count = tile_blocks + s.exception_blocks;
    if (partial_count > s.partial_capacity)
        throw std::runtime_error(
            "cuda::PairLJCutCoulLong: neighbor scratch grew after pair build");
    gather<<<grid_size(slots), kDefaultBlock, 0, stream>>>(
        p->cl_orig_index, slots, f.atoms.x, f.atoms.y, f.atoms.z,
        f.atoms.charge, s.sigma, s.epsilon, s.sorted, s.images,
        ctx.runtime.host_box, work_t(cutoff_ * cutoff_));
    ATV_CUDA_CHECK(cudaGetLastError());
    auto run = [&](auto o, auto v) {
        constexpr bool O = decltype(o)::value, V = decltype(v)::value;
        if (tile_blocks) {
            fused_tile<O, V><<<tile_blocks, kTileBlock, 0, stream>>>(
                s.sorted, p->cl_orig_index, s.packed.ci, s.packed.j,
                s.packed.masks, s.packed_count, ctx.runtime.host_box, s.images,
                work_t(alpha_), work_t(cutoff_ * cutoff_), f.atoms.fx,
                f.atoms.fy, f.atoms.fz, s.partial, s.partial_capacity);
            ATV_CUDA_CHECK(cudaGetLastError());
        }
        if (s.count) {
            exceptions<V><<<grid_size(s.count), kDefaultBlock, 0, stream>>>(
                s.pairs, s.count, f.atoms.x, f.atoms.y, f.atoms.z,
                f.atoms.charge, s.sigma, s.epsilon, ctx.runtime.host_box,
                work_t(alpha_), f.atoms.fx, f.atoms.fy, f.atoms.fz, s.partial,
                s.partial_capacity, tile_blocks);
            ATV_CUDA_CHECK(cudaGetLastError());
        }
        if (partial_count) {
            reduce_partials<V>
                <<<grid_size(partial_count), kDefaultBlock, 0, stream>>>(
                    s.partial, partial_count, s.partial_capacity,
                    &f.scalars->energy, f.scalars->stress);
            ATV_CUDA_CHECK(cudaGetLastError());
        }
    };
    if (ctx.runtime.host_box.ortho) {
        if (stress_)
            run(std::true_type{}, std::true_type{});
        else
            run(std::true_type{}, std::false_type{});
    } else {
        if (stress_)
            run(std::false_type{}, std::true_type{});
        else
            run(std::false_type{}, std::false_type{});
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}
void PairLJCutCoulLong::destroy(const DestroyContext& ctx) {
    if (state_ && ctx.runtime.md.excl_offsets == state_->offsets) {
        ctx.runtime.md.excl_offsets = nullptr;
        ctx.runtime.md.excl_partners = nullptr;
        ctx.runtime.md.s_excl = 1.0;
        ctx.runtime.md.exclusions_baked = false;
    }
    state_.reset();
}
} // namespace ATV::cuda
