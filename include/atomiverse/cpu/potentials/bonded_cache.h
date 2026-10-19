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
/// - Frame identity, atom-block generation and owned/ghost tag sequence
///   detect rebinding, topology/schema changes and serial row permutations.
///
/// Missing partners are recorded while rebuilding; require_complete() checks
/// every rank before the partial tuple set can be used for a force evaluation.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/owned_range.h"

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
    bool stale(const cpu::LaunchContext& ctx) const {
        const auto* dd = ctx.runtime.md.dd;
        const int n_force = force_count(ctx);
        if (!built_ || frame_ != &ctx.frame || owned_ != owned_count(ctx)
            || tags_.size() != static_cast<std::size_t>(n_force)
            || block_generation_ != ctx.frame.block("atoms").generation())
            return true;
        if (dd
            && (dd->neighbor_generation != neigh_gen_
                || dd->halo_generation != halo_gen_))
            return true;
        const auto tags = ctx.frame.span(kAtom_tag);
        return !std::equal(tags_.begin(), tags_.end(), tags.data());
    }

    /// @brief Record the Frame and row identity used to resolve these tuples.
    void mark_built(const cpu::LaunchContext& ctx) {
        const auto* dd = ctx.runtime.md.dd;
        built_ = true;
        frame_ = &ctx.frame;
        owned_ = owned_count(ctx);
        block_generation_ = ctx.frame.block("atoms").generation();
        const auto tags = ctx.frame.span(kAtom_tag);
        tags_.assign(tags.data(), tags.data() + force_count(ctx));
        if (dd != nullptr) {
            neigh_gen_ = dd->neighbor_generation;
            halo_gen_ = dd->halo_generation;
        }
    }

    /// @brief Fail together before any rank computes an incomplete force field.
    /// Called by every rank even when its cache has no local tuples.
    void require_complete(const cpu::LaunchContext& ctx,
                          const char* kernel) const {
        int missing = dropped_at_build != 0;
#ifdef ATV_HAS_MPI
        if (multi_rank_dd(ctx.runtime))
            MPI_Allreduce(MPI_IN_PLACE, &missing, 1, MPI_INT, MPI_MAX,
                          ctx.runtime.md.dd->cart_comm);
#endif
        if (missing)
            throw std::runtime_error(
                std::string(kernel)
                + ": bonded partner is absent from owned/ghost rows; "
                  "check topology tags and increase the communication halo");
    }

    /// @brief Drop contents and force a rebuild on next use.
    void clear() {
        flat.clear();
        flat.shrink_to_fit();
        dropped_at_build = 0;
        built_ = false;
        frame_ = nullptr;
        tags_.clear();
    }

  private:
    const cpu::Frame* frame_ = nullptr;
    int owned_ = 0;
    std::uint64_t block_generation_ = 0;
    std::vector<TagInt> tags_;
    std::uint64_t neigh_gen_ = 0;
    std::uint64_t halo_gen_ = 0;
    bool built_ = false;
};

} // namespace md
} // namespace cpu
} // namespace ATV
