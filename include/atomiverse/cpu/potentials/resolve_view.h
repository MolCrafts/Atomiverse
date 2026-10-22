/// @file cpu/potentials/resolve_view.h
/// @brief `ResolveView` — the single narrow accessor force kernels use to
///        learn how many atoms a rank owns/holds and to resolve a global
///        atom tag to a local row index, with a built-in bounds guard.
///
/// Frame owns both owned and ghost row extents for every execution topology.
///
/// Tag resolution. Partner atoms (bonded tuples, special-bonds exclusions)
/// are stored as global tags and resolved to local rows via a tag→local map
/// (`dd->tag_to_local` multi-rank, Frame-local tag index single-rank). The map
/// can retain a STALE entry for a tag whose ghost row was not repopulated by
/// the current halo (e.g. a bonded partner that stretched beyond the halo
/// under deformation). `resolve()` therefore rejects any hit outside
/// `[0, n_force)` and returns -1 ("not present on this rank") — callers
/// already treat -1 as a skip. This is the fail-safe that turns a garbage
/// coordinate read into a clean drop.
#pragma once

#include "atomiverse/cpu/component/owned_range.h"
#include <cstdint>
#include <unordered_map>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Owned/ghost counts + bounded tag→local resolution for force
///        kernels. Cheap value type; build one per `compute()` call.
class ResolveView {
  public:
    explicit ResolveView(const cpu::LaunchContext& ctx)
        : n_local_(ctx.frame.nlocal()), n_force_(ctx.frame.n_total()) {
        const auto* dd = ctx.runtime.md.dd;
        if (cpu::multi_rank_dd(ctx.runtime)) {
            map_ = &dd->tag_to_local;
        } else {
            map_ = &ctx.frame.meta.get(kTagIndex);
        }
    }

    /// @brief Owned-atom count: kernels iterate owners in `[0, n_local())`.
    int n_local() const noexcept { return n_local_; }

    /// @brief Owned + ghost count: force accumulators and reduction loops
    ///        size to `[0, n_force())`. Partner rows may be ghosts in
    ///        `[n_local(), n_force())`.
    int n_force() const noexcept { return n_force_; }

    /// @brief Resolve a global tag to a local row, or -1 if not present /
    ///        out of range. The `[0, n_force_)` bound rejects stale map
    ///        entries that point at rows the current halo never populated.
    int resolve(std::int64_t tag) const noexcept {
        const auto it = map_->find(tag);
        if (it == map_->end())
            return -1;
        const int j = it->second;
        return (j >= 0 && j < n_force_) ? j : -1;
    }

  private:
    const std::unordered_map<TagInt, int>* map_;
    int n_local_;
    int n_force_;
};

} // namespace md
} // namespace cpu
} // namespace ATV
