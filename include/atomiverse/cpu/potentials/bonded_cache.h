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
/// **Staleness contract:** `Frame::row_revision()` changes whenever a row
/// may hold a different atom — migration (`CommBrick::exchange`), ghost
/// rebuilds (`borders`), spatial sort, MC insert/delete, rebinding — and
/// stays put across position-only halo refreshes. Frame identity and the
/// atom-block generation catch rebinding and topology/schema changes. The
/// check is O(1); it replaced a per-step O(n_force) tag comparison.
///
/// Missing partners are recorded while rebuilding; require_complete() checks
/// every rank before the partial tuple set can be used for a force evaluation.
#pragma once

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
        return !built_ || frame_ != &ctx.frame
               || row_revision_ != ctx.frame.row_revision()
               || block_generation_ != ctx.frame.block("atoms").generation();
    }

    /// @brief Record the Frame and row identity used to resolve these tuples.
    void mark_built(const cpu::LaunchContext& ctx) {
        built_ = true;
        frame_ = &ctx.frame;
        row_revision_ = ctx.frame.row_revision();
        block_generation_ = ctx.frame.block("atoms").generation();
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
    }

  private:
    const cpu::Frame* frame_ = nullptr;
    std::uint64_t row_revision_ = 0;
    std::uint64_t block_generation_ = 0;
    bool built_ = false;
};

} // namespace md
} // namespace cpu
} // namespace ATV
