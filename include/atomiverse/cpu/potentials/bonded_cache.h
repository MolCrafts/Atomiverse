/// @file cpu/potentials/bonded_cache.h
/// @brief `BondedTupleCache` — resolved-tuple cache shared by the bonded
///        force kernels (BondHarmonic / AngleHarmonic / DihedralMultiHarmonic).
///
/// The bonded kernels store partner atoms as **global tags** and used to
/// resolve every tag through the `tag_to_local` hash map on **every step**
/// (1 lookup per bond partner, 2 per angle, 3 per dihedral/charmm — each a
/// cache-missing `unordered_map::find`). Local row indices are stable
/// between topology-relevant events, so the resolution is hoisted here:
/// each kernel keeps one cache of fully-resolved local-index tuples and
/// re-resolves only when the cache goes stale.
///
/// **Staleness contract** (same one the NeighborList CSR relies on):
/// - `DomainDecomposition::neighbor_generation` bumps on migration
///   (owner rows reshuffled).
/// - `DomainDecomposition::halo_generation` bumps on every halo *full
///   rebuild* (ghost slots reshuffled). Position-only halo refreshes keep
///   slot indices stable.
/// - Single-rank: rows never move; the cache is built once (`n` guards
///   driver re-binds to a different frame).
///
/// A tuple whose partner does not resolve on this rank (outside the halo)
/// is dropped at cache-build time — identical semantics to the previous
/// per-step `resolve() < 0` skip, except the drop tally advances once per
/// rebuild instead of once per step.
#pragma once

#include <cstdint>
#include <vector>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Cache of resolved bonded tuples (flat int array, fixed stride).
///
/// Layout: `stride` ints per tuple, kernel-defined (e.g. bond: `{i, j,
/// type}`; angle: `{i, j, k, type}`; dihedral: `{i, j, k, l, type}`).
struct BondedTupleCache {
    /// @brief Flattened resolved tuples (local row indices + type).
    std::vector<int> flat;

    /// @brief Tuples dropped at the last rebuild (partner outside halo).
    std::uint64_t dropped_at_build = 0;

    /// @brief True when the cache must be rebuilt before use.
    /// @param ctx Launch context (reads `dd` generation counters).
    bool stale(const cpu::LaunchContext& ctx) const noexcept {
        const auto* dd = ctx.runtime.md.dd;
        if (!built_)
            return true;
        if (dd == nullptr)
            return false; // no DD: rows never move after first build
        return dd->neighbor_generation != neigh_gen_
               || dd->halo_generation != halo_gen_;
    }

    /// @brief Record the generation stamps of the build just performed.
    /// @param ctx Launch context (reads `dd` generation counters).
    void mark_built(const cpu::LaunchContext& ctx) noexcept {
        const auto* dd = ctx.runtime.md.dd;
        built_ = true;
        if (dd != nullptr) {
            neigh_gen_ = dd->neighbor_generation;
            halo_gen_ = dd->halo_generation;
        }
    }

    /// @brief Drop contents and force a rebuild on next use.
    void clear() {
        flat.clear();
        flat.shrink_to_fit();
        dropped_at_build = 0;
        built_ = false;
    }

  private:
    std::uint64_t neigh_gen_ = 0;
    std::uint64_t halo_gen_ = 0;
    bool built_ = false;
};

} // namespace md
} // namespace cpu
} // namespace ATV
