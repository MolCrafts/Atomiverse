/// @file cpu/dynamics/electrode_rows.h
/// @brief `ElectrodeRows` — the electrode-site row list of the constant-
///        potential solvers, kept valid across atom sorting and migration.
///
/// The solvers (ConstPotBO, ConstPotCG, DipoleElectrodeJoint, and MaZe
/// through its ConstPotBO) collect their electrode sites once at build and
/// index the elastance matrix, the ASPC history and the charge vector by
/// site. Sites are therefore ordered by **tag**, which no reorder changes;
/// the rows they live in are re-resolved only when the Frame's rows move
/// (spatial sort, migration, halo rebuild) — the same staleness contract the
/// bonded kernels use. A site order tied to rows would point at the wrong
/// atoms after the first sort, which the Runtime performs at setup.
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/bonded_cache.h"
#include "atomiverse/cpu/potentials/resolve_view.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Tag-ordered electrode sites and their current rows.
class ElectrodeRows {
  public:
    /// @brief Owned rows in ascending tag order — the stable site order.
    /// @param frame Frame whose owned rows `[0, nlocal)` are listed.
    /// @return Row indices sorted by `kAtom_tag`.
    static std::vector<int> owned_by_tag(const Frame& frame) {
        std::vector<int> rows(static_cast<std::size_t>(frame.nlocal()));
        std::iota(rows.begin(), rows.end(), 0);
        const TagInt* tag = frame.span(kAtom_tag);
        std::sort(rows.begin(), rows.end(),
                  [tag](int a, int b) { return tag[a] < tag[b]; });
        return rows;
    }

    /// @brief Record the tags of the collected sites.
    /// @param frame Frame the rows index.
    /// @param rows Site rows, in site order.
    void bind(const Frame& frame, const std::vector<int>& rows) {
        const TagInt* tag = frame.span(kAtom_tag);
        tags_.resize(rows.size());
        for (std::size_t a = 0; a < rows.size(); ++a)
            tags_[a] = tag[rows[a]];
        cache_.clear();
    }

    /// @brief Re-resolve `rows` from the recorded tags if the rows moved.
    /// @param ctx Launch context.
    /// @param rows Site rows, rewritten in place (site order unchanged).
    /// @param who Component name for the error message.
    /// @throws std::runtime_error if a site is not an owned row here.
    void refresh(const LaunchContext& ctx, std::vector<int>& rows,
                 const char* who) {
        if (!cache_.stale(ctx))
            return;
        // Rows unchanged (no sort / migration since bind): keep them. This
        // is also the path of a component launched directly, outside a
        // Runtime, where no tag index has been built.
        const TagInt* tag = ctx.frame.span(kAtom_tag);
        const int n_local = ctx.frame.nlocal();
        bool same = rows.size() == tags_.size();
        for (std::size_t a = 0; same && a < rows.size(); ++a)
            same =
                rows[a] >= 0 && rows[a] < n_local && tag[rows[a]] == tags_[a];
        if (same) {
            cache_.mark_built(ctx);
            return;
        }
        const ResolveView rv(ctx);
        for (std::size_t a = 0; a < tags_.size(); ++a) {
            const int row = rv.resolve(tags_[a]);
            if (row < 0 || row >= rv.n_local())
                throw std::runtime_error(std::string(who)
                                         + ": electrode site tag "
                                         + std::to_string(tags_[a])
                                         + " is not an owned row on this rank");
            rows[a] = row;
        }
        cache_.mark_built(ctx);
    }

  private:
    std::vector<TagInt> tags_;
    BondedTupleCache cache_; ///< row-identity staleness only
};

} // namespace md
} // namespace cpu
} // namespace ATV
