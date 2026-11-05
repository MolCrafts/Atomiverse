/// @file cuda/potentials/pair_lj_cut_coul_long.cu
/// @brief Fused packed-J tile force and sparse Ewald exception correction.
/// Algorithm reference: OpenMM nonbonded.cu, findInteractingBlocks.cu and
/// nonbondedExceptions.cc,
/// commit 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e. Uses the published ideas
/// of parameter fusion, warp data exchange and a separate exception pass;
/// implementation is ATV-native, with no copied OpenMM source.
/// Typed LJ follows LAMMPS `lj/cut/coul/long` `init_one` semantics through
/// the shared `core/lj_type_table.h` (provenance there); each typed pair reads
/// its record from a read-only device table (LAMMPS `lib/gpu` keeps per-type
/// lj1..lj4 / cutsq / offset the same way). The exception pass applies the
/// LAMMPS `PairLJCutCoulLong::compute` cutoffs (`src/KSPACE/
/// pair_lj_cut_coul_long.cpp`, checkout `e8c4c278a8`): LJ only for
/// r² < cut_ljsq(i, j), Coulomb including the −(1 − factor_coul) qq/r
/// correction only for r² < cut_coulsq. The LJ tail correction is the LAMMPS
/// `init_one` etail/ptail (`core/lj_type_table.h`) applied as in
/// `src/compute_pe.cpp` (etail/V) and `ComputePressure::virial_compute`
/// (`src/compute_pressure.cpp`, ptail/V on the diagonal):
///
///   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
///   https://www.lammps.org/, Sandia National Laboratories
///   LAMMPS development team: developers@lammps.org
///
///   Copyright (2003) Sandia Corporation.  Under the terms of Contract
///   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
///   certain rights in this software.  This software is distributed under
///   the GNU General Public License.

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/lj_type_table.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/energy_term.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include "internal/lj_tail.cuh"
#include "internal/packed_tiles.cuh"
#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ATV::cuda {
namespace {
// 128 threads x 4 resident blocks: 256/2, 128/3 and 128/6 measured no
// better on GH200 (256/2 is 7% slower in double at 49,680 atoms).
constexpr int kTileBlock = 128;
constexpr int kTileMinBlocks = 4;
constexpr force_t kForceInfinity = std::numeric_limits<force_t>::infinity();
constexpr force_t kForceEpsilon = std::numeric_limits<force_t>::epsilon();
using ForceBox = BoxGeometry<force_t>;

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

__global__ void count_packed_tiles(PairList source, int* counts, int* valid) {
    const int ci = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (ci > source.num_clusters)
        return;
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int columns = 0, entries = 0;
    if (ci < source.num_clusters)
        for (int p = source.cl_row_ptr[ci]; p < source.cl_row_ptr[ci + 1];
             ++p) {
            const unsigned used = cg::reduce(
                warp, upper_mask(source, p, ci, lane), cg::bit_or<unsigned>{});
            columns += __popc(used);
            entries += used != 0;
        }
    if (lane == 0) {
        counts[ci] = (columns + 31) / 32;
        if (valid != nullptr && entries)
            atomicAdd(valid, entries);
    }
}

__global__ void fill_packed_tiles(PairList source,
                                  internal::PackedTileWorkspace out,
                                  ForceFault* fault) {
    const int ci = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (ci >= source.num_clusters)
        return;
    // A packed tile count never exceeds the CSR entry count, which the
    // neighbor list already bounds by the same reserve; guard regardless.
    const int total = out.offsets[source.num_clusters];
    if (total > out.capacity) {
        if (ci == 0 && lane == 0)
            raise_capacity_fault(fault, total);
        return;
    }
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int tile = out.offsets[ci], column = 0;
    unsigned packed_bits = 0;
    for (int p = source.cl_row_ptr[ci]; p < source.cl_row_ptr[ci + 1]; ++p) {
        const unsigned row = upper_mask(source, p, ci, lane);
        unsigned columns = cg::reduce(warp, row, cg::bit_or<unsigned>{});
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

// Per-atom Lorentz–Berthelot tile parameters (the OpenMM-benchmarked path):
// σ_ij = σ_i/2 + σ_j/2, 4ε_ij = (2√ε_i)(2√ε_j).
struct Atom {
    force_t x, y, z, q, sigma_half, epsilon_root;
};
// Typed tile parameters: the pair record comes from the device type table.
struct TypedAtom {
    force_t x, y, z, q;
    int type;
};
template <bool Typed>
using TileAtom = std::conditional_t<Typed, TypedAtom, Atom>;

// One resolved (type_i, type_j) record of `LJTypeTable` in force_t
// (LAMMPS lj1..lj4, cut_ljsq, offset; the ATV r^-4 constants).
struct PairCoeff {
    force_t lj1, lj2, lj3, lj4, cut_ljsq, offset;
    force_t r4_a, r4_cutsq, r4_rc5inv, r4_e0;
};
struct Exception {
    int i, j, ab; // atom rows and table index of (type_i, type_j)
    force_t lj_scale, coul_scale;
};
struct ClusterImage {
    force_t center[3];
    int single;
};

// OpenMM's singlePeriodicCopy criterion, evaluated from live I coordinates
// rather than rebuild-time bounds. Thus a reused Hilbert cluster needs no
// assumption about its motion within the neighbor skin. Wide clusters safely
// keep the per-pair minimum image calculation.
template <class A>
__device__ __forceinline__ ClusterImage cluster_image(const A& a, int ai,
                                                      const ForceBox& box,
                                                      force_t cut2) {
    const force_t position[3] = {a.x, a.y, a.z};
    ClusterImage image{};
    const force_t cutoff = sqrt(cut2);
#pragma unroll
    for (int d = 0; d < 3; ++d) {
        force_t lo = ai >= 0 ? position[d] : kForceInfinity;
        force_t hi = ai >= 0 ? position[d] : -kForceInfinity;
#pragma unroll
        namespace cg = cooperative_groups;
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());
        lo = cg::reduce(warp, lo, cg::less<force_t>{});
        hi = cg::reduce(warp, hi, cg::greater<force_t>{});
        const force_t length = force_t(box.cell[4 * d]);
        const force_t halfwidth = (hi - lo) * force_t(0.5);
        const force_t margin = force_t(32) * kForceEpsilon
                               * (fabs(lo) + fabs(hi) + fabs(length) + cutoff);
        // Every cutoff neighbor then lies strictly inside the single box
        // centered here. The guard is warp-uniform, including padded lanes.
        if (!(halfwidth + cutoff + margin < length * force_t(0.5)))
            return image;
        image.center[d] = lo + halfwidth;
    }
    image.single = 1;
    return image;
}

template <class A>
__device__ __forceinline__ void
apply_cluster_image(A& a, const ClusterImage& image, const ForceBox& box) {
    a.x -= box.cell[0] * nearbyint((a.x - image.center[0]) * box.inv_l[0]);
    a.y -= box.cell[4] * nearbyint((a.y - image.center[1]) * box.inv_l[1]);
    a.z -= box.cell[8] * nearbyint((a.z - image.center[2]) * box.inv_l[2]);
}

template <bool Typed>
__global__ void gather(const int* index, int slots, const pos_t* x,
                       const pos_t* y, const pos_t* z, const force_t* q,
                       const force_t* sigma, const force_t* epsilon,
                       const int* type, TileAtom<Typed>* sorted,
                       ClusterImage* images, Box box_arg, force_t cut2) {
    const ForceBox box = box_arg.current().as<force_t>();
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= slots)
        return;
    const int i = index[s];
    TileAtom<Typed> atom{};
    if (i >= 0) {
        if constexpr (Typed)
            atom = TypedAtom{x[i], y[i], z[i], q[i], type[i]};
        else
            atom = Atom{x[i], y[i], z[i], q[i], sigma[i], epsilon[i]};
    }
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
template <bool Virial, int BlockSize>
__device__ void tally(acc_t e, const acc_t* v, acc_t* partial,
                      int partial_stride, int slot) {
    // The tile producer uses kTileBlock; exceptions use kDefaultBlock.
    // The caller supplies its compile-time block size to CUB.
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

// Tile partials occupy [0, live chunks) of a reserve of `tile_slots`;
// exception partials follow the reserve. Logical index i enumerates the live
// tile chunks, then the exceptions, so each block sums the same partials in
// the same order as a contiguous layout would.
template <bool Energy, bool Virial>
__global__ void reduce_partials(const acc_t* partial, const int* tile_count,
                                int tile_slots, int exception_slots, int stride,
                                EnergyTerm energy, acc_t* stress) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunks = (*tile_count + warps_per_block(kTileBlock) - 1)
                       / warps_per_block(kTileBlock);
    constexpr int components = Virial ? 7 : 1;
    using Values = ::cuda::std::array<acc_t, components>;
    using Reduce = cub::BlockReduce<Values, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    Values values{};
    if (i < chunks + exception_slots) {
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

// Typed LJ + r^-4 of one pair from its table record (read-only cache):
// LAMMPS forcelj * r2inv and evdwl - offset inside cut_ljsq, the r^-4 term
// inside its own cutoff. Adds to the pair force fr = -(dU/dr)/r and energy.
template <bool Energy>
__device__ __forceinline__ void typed_lj(const PairCoeff* c, force_t r2,
                                         force_t inv2, force_t scale,
                                         force_t& fr, force_t& en) {
    if (r2 < __ldg(&c->cut_ljsq)) {
        const force_t r6inv = inv2 * inv2 * inv2;
        fr += scale * inv2 * r6inv * (__ldg(&c->lj1) * r6inv - __ldg(&c->lj2));
        if constexpr (Energy)
            en += scale
                  * (r6inv * (__ldg(&c->lj3) * r6inv - __ldg(&c->lj4))
                     - __ldg(&c->offset));
    }
    if (r2 < __ldg(&c->r4_cutsq)) {
        const R4Term<force_t> t = r4_attraction<force_t>(
            r2, __ldg(&c->r4_a), __ldg(&c->r4_rc5inv), __ldg(&c->r4_e0));
        fr += scale * t.fpair;
        if constexpr (Energy)
            en += scale * t.energy;
    }
}

template <bool Ortho, bool Energy, bool Virial, bool Typed>
__global__ __launch_bounds__(kTileBlock, kTileMinBlocks) void fused_tile(
    const TileAtom<Typed>* atoms, const int* index, const int* packed_ci,
    const int* packed_j, const std::uint32_t* masks, const int* tile_count,
    Box box_arg, const ClusterImage* images, force_t alpha, force_t cut2,
    force_t coul_cut2, force_t inv_rc2, bool shift, const PairCoeff* table,
    int n_types, fixed_t* fx, fixed_t* fy, fixed_t* fz, ForceFault* fault,
    acc_t* partial, int partial_stride) {
    const ForceBox box = box_arg.current().as<force_t>();
    // A warp owns one I cluster and 32 packed J atoms. Baked masks already
    // contain only upper-triangle interactions. Pair math runs in force_t;
    // energy and virial accumulate in acc_t per interaction, and each lane
    // adds its I and J forces to the fixed-point columns once per tile.
    // The tile count lives on the device; blocks stride over chunks of
    // `warps` tiles and tally each chunk into its own partial slot.
    // One block per reserved chunk; a block past the live count exits before
    // any barrier (uniform per block). A striding loop here cost 17%.
    const int num_pairs = *tile_count;
    const int warps = blockDim.x / 32;
    const int chunk = blockIdx.x;
    if (chunk * warps >= num_pairs)
        return;
    const int lane = threadIdx.x % 32;
    {
        const int p = chunk * warps + threadIdx.x / 32;
        acc_t e = 0, v[6] = {};
        DeferredForceAdds<6> adds;
        if (p < num_pairs) {
            const int si = packed_ci[p] * 32 + lane;
            const std::size_t slot = static_cast<std::size_t>(p) * 32 + lane;
            const int sj = packed_j[slot];
            const int ai = index[si], bj = sj >= 0 ? index[sj] : -1;
            TileAtom<Typed> a = atoms[si],
                            b = sj >= 0 ? atoms[sj] : TileAtom<Typed>{};
            const unsigned mask = masks[slot];
            force_t fi[3] = {}, fj[3] = {};
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
                const PairCoeff* row = nullptr;
                if constexpr (Typed)
                    row = table + static_cast<std::size_t>(a.type) * n_types;
                for (int step = 0; step < 32; ++step) {
                    const int jlane = (lane + step) & 31;
                    const int aj = __shfl_sync(0xffffffffu, bj, jlane);
                    force_t dx = __shfl_sync(0xffffffffu, b.x, jlane) - a.x;
                    force_t dy = __shfl_sync(0xffffffffu, b.y, jlane) - a.y;
                    force_t dz = __shfl_sync(0xffffffffu, b.z, jlane) - a.z;
                    const force_t qj = __shfl_sync(0xffffffffu, b.q, jlane);
                    force_t sig = 0, eps = 0;
                    int tj = 0;
                    if constexpr (Typed) {
                        tj = __shfl_sync(0xffffffffu, b.type, jlane);
                    } else {
                        sig = a.sigma_half
                              + __shfl_sync(0xffffffffu, b.sigma_half, jlane);
                        eps = a.epsilon_root
                              * __shfl_sync(0xffffffffu, b.epsilon_root, jlane);
                    }
                    if constexpr (Ortho) {
                        if (!single_copy) {
                            dx -= box.cell[0] * nearbyint(dx * box.inv_l[0]);
                            dy -= box.cell[4] * nearbyint(dy * box.inv_l[1]);
                            dz -= box.cell[8] * nearbyint(dz * box.inv_l[2]);
                        }
                    } else
                        box.min_image(dx, dy, dz);
                    const force_t r2 = dx * dx + dy * dy + dz * dz;
                    force_t contribution[3] = {};
                    if (ai >= 0 && aj >= 0 && ai != aj && (mask & (1u << jlane))
                        && r2 > 0 && r2 < cut2) {
                        const force_t inv = rsqrt(r2), inv2 = inv * inv;
                        force_t fr = 0, en = 0;
                        if constexpr (Typed) {
                            if (r2 < coul_cut2) {
                                const force_t ar = alpha * (r2 * inv);
                                const force_t ex = exp(-ar * ar),
                                              er = erfcx(ar) * ex;
                                const force_t qq =
                                    force_t(units::md_qqr2e) * a.q * qj * inv;
                                fr = qq
                                     * (er
                                        + force_t(1.1283791670955126)
                                              * (ar * ex))
                                     * inv2;
                                en = qq * er;
                            }
                            typed_lj<Energy>(row + tj, r2, inv2, force_t(1), fr,
                                             en);
                        } else {
                            const force_t ar = alpha * (r2 * inv);
                            const force_t sr2 = sig * sig * inv2,
                                          sr6 = sr2 * sr2 * sr2;
                            const force_t ex = exp(-ar * ar),
                                          er = erfcx(ar) * ex;
                            const force_t qq =
                                force_t(units::md_qqr2e) * a.q * qj * inv;
                            fr = (eps * (12 * sr6 - 6) * sr6
                                  + qq
                                        * (er
                                           + force_t(1.1283791670955126)
                                                 * (ar * ex)))
                                 * inv2;
                            if constexpr (Energy) {
                                en = eps * sr6 * (sr6 - 1) + qq * er;
                                if (shift) {
                                    // LAMMPS offset at the common cutoff.
                                    const force_t sc2 = sig * sig * inv_rc2,
                                                  sc6 = sc2 * sc2 * sc2;
                                    en -= eps * sc6 * (sc6 - 1);
                                }
                            }
                        }
                        contribution[0] = fr * dx;
                        contribution[1] = fr * dy;
                        contribution[2] = fr * dz;
                        for (int d = 0; d < 3; ++d)
                            fi[d] -= contribution[d];
                        if constexpr (Energy)
                            e += acc_t(en);
                        if constexpr (Virial) {
                            v[0] += acc_t(fr * dx * dx);
                            v[1] += acc_t(fr * dy * dy);
                            v[2] += acc_t(fr * dz * dz);
                            v[3] += acc_t(fr * dx * dy);
                            v[4] += acc_t(fr * dx * dz);
                            v[5] += acc_t(fr * dy * dz);
                        }
                    }
                    // Send each contribution back to its original j lane. Tail
                    // lanes and excluded pairs still reach every shuffle.
                    const int source = (lane - step) & 31;
                    for (int d = 0; d < 3; ++d)
                        fj[d] +=
                            __shfl_sync(0xffffffffu, contribution[d], source);
                }
            }
            // Issue the six atomics now; their wraparound check runs after the
            // energy/virial reduction below.
            fixed_t* const columns[3] = {fx, fy, fz};
#pragma unroll
            for (int d = 0; d < 3; ++d) {
                adds.add(d, columns[d], ai, fi[d], fault);
                adds.add(3 + d, columns[d], bj, fj[d], fault);
            }
        }
        if constexpr (Energy || Virial)
            tally<Virial, kTileBlock>(e, v, partial, partial_stride, chunk);
        adds.check(fault);
    }
}

// Special-bond exceptions (LAMMPS factor_lj / factor_coul with a kspace
// partner, PairLJCutCoulLong::compute): factor_lj * (LJ - offset) inside the
// pair's LJ cutoff and factor_lj * r^-4 inside its cutoff, plus, inside the
// Coulomb cutoff only, qq (factor_coul - erf(alpha r))/r — the unscaled erfc
// part minus the (1 - factor_coul) plain share the reciprocal sum added. A
// bonded pair beyond every cutoff contributes nothing, as on the CPU, whose
// neighbor list never offers it.
template <bool Energy, bool Virial>
__global__ void exceptions(const Exception* pairs, int count, const pos_t* x,
                           const pos_t* y, const pos_t* z, const force_t* q,
                           const PairCoeff* table, Box box_arg, force_t alpha,
                           force_t coul_cut2, fixed_t* fx, fixed_t* fy,
                           fixed_t* fz, ForceFault* fault, acc_t* partial,
                           int partial_stride, int partial_offset) {
    const ForceBox box = box_arg.current().as<force_t>();
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    acc_t e = 0, v[6] = {};
    DeferredForceAdds<6> adds;
    if (p < count) {
        const Exception t = pairs[p];
        const int i = t.i, j = t.j;
        force_t dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
        box.min_image(dx, dy, dz);
        const force_t r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 0) {
            const force_t inv2 = 1 / r2, inv = 1 / sqrt(r2), ar = alpha / inv;
            force_t fr = 0, en = 0;
            if (t.lj_scale != 0)
                typed_lj<Energy>(table + t.ab, r2, inv2, t.lj_scale, fr, en);
            if (r2 < coul_cut2) {
                const force_t qq = force_t(units::md_qqr2e) * q[i] * q[j] * inv;
                const force_t screened = t.coul_scale - erf(ar);
                fr += qq
                      * (screened
                         + force_t(1.1283791670955126) * ar * exp(-ar * ar))
                      * inv2;
                en += qq * screened;
            }
            const force_t f[3] = {fr * dx, fr * dy, fr * dz};
            adds.add(0, fx, i, -f[0], fault);
            adds.add(1, fy, i, -f[1], fault);
            adds.add(2, fz, i, -f[2], fault);
            adds.add(3, fx, j, f[0], fault);
            adds.add(4, fy, j, f[1], fault);
            adds.add(5, fz, j, f[2], fault);
            if constexpr (Energy)
                e = acc_t(en);
            if constexpr (Virial) {
                v[0] = acc_t(fr * dx * dx);
                v[1] = acc_t(fr * dy * dy);
                v[2] = acc_t(fr * dz * dz);
                v[3] = acc_t(fr * dx * dy);
                v[4] = acc_t(fr * dx * dz);
                v[5] = acc_t(fr * dy * dz);
            }
        }
    }
    if constexpr (Energy || Virial)
        tally<Virial, kDefaultBlock>(e, v, partial, partial_stride,
                                     partial_offset + blockIdx.x);
    adds.check(fault);
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

void internal::pack_tiles(const PairList& source,
                          const PackedTileWorkspace& work, ForceFault* fault,
                          cudaStream_t stream) {
    if (work.valid != nullptr)
        ATV_CUDA_CHECK(cudaMemsetAsync(work.valid, 0, sizeof(int), stream));
    count_packed_tiles<<<(source.num_clusters + 4) / 4, kTileBlock, 0,
                         stream>>>(source, work.counts, work.valid);
    ATV_CUDA_CHECK(cudaGetLastError());
    std::size_t bytes = work.scan_bytes;
    ATV_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        work.scan_temp, bytes, work.counts, work.offsets,
        source.num_clusters + 1, stream));
    if (source.num_clusters > 0) {
        fill_packed_tiles<<<(source.num_clusters + 3) / 4, kTileBlock, 0,
                            stream>>>(source, work, fault);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}

struct PairLJCutCoulLong::State {
    void* sorted = nullptr; // Atom[] or TypedAtom[] (see `typed`)
    ClusterImage* images = nullptr;
    force_t* sigma = nullptr;
    force_t* epsilon = nullptr;
    PairCoeff* table = nullptr;
    int n_types = 0;
    bool typed = false;
    int* offsets = nullptr;
    int* partners = nullptr;
    Exception* pairs = nullptr;
    int count = 0, slots = 0;
    acc_t* partial = nullptr; // SoA: energy followed by six virial planes.
    int partial_capacity = 0;
    int exception_blocks = 0;
    internal::PackedTileWorkspace packed;
    int tile_slots = 0; // tile-chunk partials reserved ahead of exceptions
    int tile_grid = 0;  // one block per reserved tile chunk
    int packed_clusters = 0;
    RebuildHooks* hooks = nullptr; ///< Provider hooks holding our repack.
    ~State() {
        cudaFree(sorted);
        cudaFree(images);
        cudaFree(sigma);
        cudaFree(epsilon);
        cudaFree(table);
        cudaFree(offsets);
        cudaFree(partners);
        cudaFree(pairs);
        cudaFree(partial);
        cudaFree(packed.counts);
        cudaFree(packed.offsets);
        cudaFree(packed.ci);
        cudaFree(packed.j);
        cudaFree(packed.masks);
        cudaFree(packed.scan_temp);
        cudaFree(packed.valid);
    }
};
PairLJCutCoulLong::PairLJCutCoulLong(std::vector<LJTypeCoeff> coeffs,
                                     double alpha, double cutoff)
    : Pair(cutoff),
      // Refuses an empty table, a bad row and a non-positive cutoff.
      table_(std::move(coeffs), cutoff), alpha_(alpha), cut_coul_(cutoff) {
    if (!std::isfinite(alpha) || alpha <= 0)
        throw std::invalid_argument(
            "cuda::PairLJCutCoulLong: positive finite alpha required");
    refresh_cutoff_();
    // Same default as cpu::PairLJCutCoulLong.
    set_special_bonds_amber();
}
PairLJCutCoulLong::~PairLJCutCoulLong() = default;
PairLJCutCoulLong::PairLJCutCoulLong(PairLJCutCoulLong&&) noexcept = default;
PairLJCutCoulLong&
PairLJCutCoulLong::operator=(PairLJCutCoulLong&&) noexcept = default;

void PairLJCutCoulLong::refresh_cutoff_() {
    // LAMMPS init_one returns max(cut_lj[i][j], cut_coul); the neighbor list
    // and the tile cutoff test cover the largest of those.
    cutoff_ = std::max(cut_coul_, table_.max_cutoff());
}
PairLJCutCoulLong& PairLJCutCoulLong::set_special_bonds_amber() & {
    // LAMMPS `special_bonds amber`
    return set_special_bonds(0.0, 0.0, 0.5, 0.0, 0.0, 5.0 / 6.0);
}
PairLJCutCoulLong PairLJCutCoulLong::set_special_bonds_amber() && {
    set_special_bonds_amber();
    return std::move(*this);
}
PairLJCutCoulLong&
PairLJCutCoulLong::set_special_bonds(double lj12, double lj13, double lj14,
                                     double c12, double c13, double c14) & {
    for (double w : {lj12, lj13, lj14, c12, c13, c14})
        if (!std::isfinite(w))
            throw std::invalid_argument(
                "cuda::PairLJCutCoulLong: special-bond weights must be finite");
    lj_[1] = lj12;
    lj_[2] = lj13;
    lj_[3] = lj14;
    coul_[1] = c12;
    coul_[2] = c13;
    coul_[3] = c14;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_special_bonds(double lj12, double lj13,
                                                       double lj14, double c12,
                                                       double c13,
                                                       double c14) && {
    set_special_bonds(lj12, lj13, lj14, c12, c13, c14);
    return std::move(*this);
}
PairLJCutCoulLong& PairLJCutCoulLong::set_energy_shift(bool on) & {
    shift_ = on;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_energy_shift(bool on) && {
    shift_ = on;
    return std::move(*this);
}
PairLJCutCoulLong& PairLJCutCoulLong::set_type_coeff(const LJTypeCoeff& row) & {
    table_.set_coeff(row);
    refresh_cutoff_();
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_type_coeff(const LJTypeCoeff& row) && {
    set_type_coeff(row);
    return std::move(*this);
}
PairLJCutCoulLong& PairLJCutCoulLong::set_tail_correction(bool on) & {
    tail_ = on;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_tail_correction(bool on) && {
    tail_ = on;
    return std::move(*this);
}
PairLJCutCoulLong&
PairLJCutCoulLong::set_r4_attraction(std::vector<R4TypeCoeff> rows) & {
    table_.set_r4(std::move(rows));
    refresh_cutoff_();
    return *this;
}
PairLJCutCoulLong
PairLJCutCoulLong::set_r4_attraction(std::vector<R4TypeCoeff> rows) && {
    set_r4_attraction(std::move(rows));
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
    if (n > 0 && !f.atoms.type)
        throw std::invalid_argument(
            std::string("cuda::PairLJCutCoulLong: the per-type LJ table "
                        "needs the force-field type column; ")
            + atom_field_how_to("type"));
    std::vector<int> types(n);
    if (n)
        ATV_CUDA_CHECK(cudaMemcpy(types.data(), f.atoms.type, n * sizeof(int),
                                  cudaMemcpyDeviceToHost));
    int max_type = -1;
    for (int t : types) {
        if (t < 0)
            throw std::invalid_argument(
                "cuda::PairLJCutCoulLong: atom types must be >= 0");
        max_type = std::max(max_type, t);
    }
    // LAMMPS init_one over every type the Frame carries; a type no row
    // names has no LJ site.
    table_.build(max_type + 1, shift_);
    const int nt = table_.n_types();
    s.n_types = nt;
    {
        std::vector<PairCoeff> host(table_.pairs().size());
        for (std::size_t k = 0; k < host.size(); ++k) {
            const LJPairCoeff& c = table_.pairs()[k];
            host[k] = PairCoeff{force_t(c.lj1),       force_t(c.lj2),
                                force_t(c.lj3),       force_t(c.lj4),
                                force_t(c.cut_ljsq),  force_t(c.offset),
                                force_t(c.r4_a),      force_t(c.r4_cutsq),
                                force_t(c.r4_rc5inv), force_t(c.r4_e0)};
        }
        upload(s.table, host);
    }
    // LAMMPS init_one tail over this build's global type counts.
    tail_pref_ = LJTail{};
    if (tail_) {
        std::vector<double> counts(static_cast<std::size_t>(nt), 0.0);
        for (int t : types)
            counts[static_cast<std::size_t>(t)] += 1.0;
        tail_pref_ = table_.tail(counts);
    }
    // The per-atom (sigma/2, 2 sqrt(eps)) tile form reproduces the table
    // only for pure Lorentz-Berthelot diagonals at one cutoff (which is the
    // Coulomb cutoff, so the shift offset is per-atom computable too).
    s.typed = !table_.lorentz_berthelot_uniform();
    typed_ = s.typed;
    if (!s.typed) {
        std::vector<force_t> sig(n), eps(n);
        for (int i = 0; i < n; ++i) {
            const LJPairCoeff& d = table_(types[i], types[i]);
            sig[i] = force_t(d.sigma * 0.5);
            eps[i] = force_t(2 * std::sqrt(d.epsilon));
        }
        upload(s.sigma, sig);
        upload(s.epsilon, eps);
    }
    s.slots = ((f.capacity() + 31) / 32) * 32;
    if (s.slots) {
        const std::size_t atom_bytes =
            s.typed ? sizeof(TypedAtom) : sizeof(Atom);
        ATV_CUDA_CHECK(cudaMalloc(&s.sorted, s.slots * atom_bytes));
        ATV_CUDA_CHECK(cudaMalloc(&s.images, std::size_t(s.slots / 32)
                                                 * sizeof(ClusterImage)));
    }
    // Exceptions: the nearest bond-graph shell (1, 2, 3) of a pair decides
    // its weights; only shells whose (LJ, Coulomb) weights differ from (1, 1)
    // leave the tiles. A pure (1, 1, 1, 1, 1, 1) setup has none.
    bool any_special = false;
    for (int d = 1; d <= 3; ++d)
        any_special = any_special || lj_[d] != 1.0 || coul_[d] != 1.0;
    std::vector<std::vector<int>> graph(n);
    if (any_special) {
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
            if (lj_[d] == 1.0 && coul_[d] == 1.0)
                continue; // an ordinary pair: stays in the tiles
            partners.push_back(j);
            if (i < j)
                pairs.push_back({i, j, types[i] * nt + types[j],
                                 force_t(lj_[d]), force_t(coul_[d])});
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
    ATV_CUDA_CHECK(cudaMalloc(&packed.valid, sizeof(int)));
    ATV_CUDA_CHECK(cudaMemset(packed.valid, 0, sizeof(int)));
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
    s.tile_slots =
        static_cast<int>((static_cast<long long>(list->cluster_pair_capacity)
                          + warps_per_block(kTileBlock) - 1)
                         / warps_per_block(kTileBlock));
    // One block per reserved chunk; blocks past the live (device) count
    // exit. A resident striding grid measured 15% slower on GH200.
    s.tile_grid = std::max(1, s.tile_slots);
    s.exception_blocks = s.count > 0 ? grid_size(s.count) : 0;
    s.partial_capacity = s.tile_slots + s.exception_blocks;
    if (s.partial_capacity > 0)
        ATV_CUDA_CHECK(
            cudaMalloc(&s.partial, static_cast<std::size_t>(s.partial_capacity)
                                       * 7 * sizeof(acc_t)));
    // Ordinary tile force excludes every exception. The sparse pass restores
    // the scaled LJ / r^-4 and the real-space Ewald exclusion correction.
    ctx.runtime.md.excl_offsets = s.offsets;
    ctx.runtime.md.excl_partners = s.partners;
    ctx.runtime.md.s_excl = 0.0;
    ctx.runtime.md.exclusions_baked = false;
    // Repack inside every neighbor rebuild, after its masks are final: no
    // device decision or launch of our own (the list owns the cadence).
    if (!list->hooks)
        throw std::invalid_argument("cuda::PairLJCutCoulLong: neighbor list "
                                    "publishes no rebuild hooks");
    s.hooks = list->hooks;
    State* state = &s;
    ForceFault* fault = ctx.frame.force_fault;
    s.hooks->add(state, [state, list, fault](void* stream) {
        internal::pack_tiles(*list, state->packed, fault,
                             static_cast<cudaStream_t>(stream));
    });
    s.packed_clusters = list->num_clusters;
}
void PairLJCutCoulLong::compute(const LaunchContext& ctx) {
    const PairList* p = ctx.runtime.md.pair_list;
    if (!p || p->num_clusters == 0)
        return;
    if (p->cl_row_ptr == nullptr || p->generation == nullptr)
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

    if (p->hooks != s.hooks)
        throw std::runtime_error("cuda::PairLJCutCoulLong: neighbor list "
                                 "replaced after pair build");
    const int* tile_count = s.packed.offsets + p->num_clusters;
    // Typed tiles test each pair's own LJ / r^-4 cutoff and the Coulomb one
    // inside the reported (largest) cutoff. The per-atom Lorentz-Berthelot
    // tile has no per-pair test: every interacting LJ pair of such a table
    // sits at the global cutoff, which is the Coulomb cutoff, so that is its
    // pair cutoff. The reported cutoff can only be larger through epsilon = 0
    // rows, which must not extend Coulomb (or any LJ pair) beyond cut_coul.
    const double pair_cut = s.typed ? cutoff_ : cut_coul_;
    const force_t cut2 = force_t(pair_cut * pair_cut);
    const force_t coul_cut2 = force_t(cut_coul_ * cut_coul_);
    const force_t inv_rc2 = force_t(1.0 / (cut_coul_ * cut_coul_));

    auto run = [&](auto o, auto e, auto v, auto t) {
        constexpr bool O = decltype(o)::value, E = decltype(e)::value,
                       V = decltype(v)::value, T = decltype(t)::value;
        if (s.tile_slots)
            fused_tile<O, E, V, T><<<s.tile_grid, kTileBlock, 0, stream>>>(
                static_cast<const TileAtom<T>*>(s.sorted), p->cl_orig_index,
                s.packed.ci, s.packed.j, s.packed.masks, tile_count,
                ctx.runtime.host_box, s.images, force_t(alpha_), cut2,
                coul_cut2, inv_rc2, shift_, s.table, s.n_types, f.atoms.fx,
                f.atoms.fy, f.atoms.fz, f.force_fault, s.partial,
                s.partial_capacity);
        if (s.count)
            exceptions<E, V><<<grid_size(s.count), kDefaultBlock, 0, stream>>>(
                s.pairs, s.count, f.atoms.x, f.atoms.y, f.atoms.z,
                f.atoms.charge, s.table, ctx.runtime.host_box, force_t(alpha_),
                coul_cut2, f.atoms.fx, f.atoms.fy, f.atoms.fz, f.force_fault,
                s.partial, s.partial_capacity, s.tile_slots);
        if (s.partial_capacity && (E || V))
            reduce_partials<E, V>
                <<<grid_size(s.partial_capacity), kDefaultBlock, 0, stream>>>(
                    s.partial, tile_count, s.tile_slots, s.exception_blocks,
                    s.partial_capacity,
                    energy_term(f.scalars, EnergyKind::Pair, E),
                    f.scalars->stress);
    };
    // Per-step LAMMPS eflag/vflag.
    const bool energy = ctx.energy_requested, virial = ctx.stress_requested;
    auto by_flags = [&](auto o, auto t) {
        using T = std::true_type;
        using F = std::false_type;
        if (energy && virial)
            run(o, T{}, T{}, t);
        else if (energy)
            run(o, T{}, F{}, t);
        else if (virial)
            run(o, F{}, T{}, t);
        else
            run(o, F{}, F{}, t);
    };
    auto by_kernel = [&](auto t) {
        constexpr bool T = decltype(t)::value;
        gather<T><<<grid_size(slots), kDefaultBlock, 0, stream>>>(
            p->cl_orig_index, slots, f.atoms.x, f.atoms.y, f.atoms.z,
            f.atoms.charge, s.sigma, s.epsilon, f.atoms.type,
            static_cast<TileAtom<T>*>(s.sorted), s.images, ctx.runtime.host_box,
            cut2);
        if (ctx.runtime.host_box.ortho)
            by_flags(std::true_type{}, t);
        else
            by_flags(std::false_type{}, t);
    };
    if (s.typed)
        by_kernel(std::true_type{});
    else
        by_kernel(std::false_type{});
    if (tail_)
        internal::launch_lj_tail(
            ctx.runtime.host_box, tail_pref_,
            energy_term(f.scalars, EnergyKind::Pair, energy),
            virial ? f.scalars->stress : nullptr, stream);
    ATV_CUDA_CHECK(cudaGetLastError());
}

int PairLJCutCoulLong::packed_tiles() const {
    if (!state_ || !state_->packed.offsets)
        return 0;
    int count = 0;
    ATV_CUDA_CHECK(cudaMemcpy(&count,
                              state_->packed.offsets + state_->packed_clusters,
                              sizeof(int), cudaMemcpyDeviceToHost));
    return count;
}

int PairLJCutCoulLong::valid_tiles() const {
    if (!state_ || !state_->packed.valid)
        return 0;
    int count = 0;
    ATV_CUDA_CHECK(cudaMemcpy(&count, state_->packed.valid, sizeof(int),
                              cudaMemcpyDeviceToHost));
    return count;
}

bool PairLJCutCoulLong::typed_kernel() const { return typed_; }

void PairLJCutCoulLong::destroy(const DestroyContext& ctx) {
    // Detach the repack while the neighbor list is certainly alive.
    if (state_ && state_->hooks)
        state_->hooks->remove(state_.get());
    if (state_ && ctx.runtime.md.excl_offsets == state_->offsets) {
        ctx.runtime.md.excl_offsets = nullptr;
        ctx.runtime.md.excl_partners = nullptr;
        ctx.runtime.md.s_excl = 1.0;
        ctx.runtime.md.exclusions_baked = false;
    }
    state_.reset();
}
} // namespace ATV::cuda
