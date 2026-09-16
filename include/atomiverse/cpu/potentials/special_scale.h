/// @file cpu/potentials/special_scale.h
/// @brief Shared LAMMPS special_bonds scaling table for pair styles.
///
/// Builds a per-owner sorted (partner_tag, scale) list from
/// `kAtom_nspecial` / `kAtom_special`. Scales multiply pair energy and force
/// (0 = skip, 0.5 = Amber 1–4, 1 = full).
///
/// Hot path: #scale_owner(local_i, partner_tag) — O(log k) binary search on
/// that owner's special list (k ≲ 10), no hash map.
#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Per-owner special-bond weight lookup (LAMMPS special_bonds).
struct SpecialScaleTable {
    double s12 = 1.0;
    double s13 = 1.0;
    double s14 = 1.0;

    /// by_owner[i] = sorted (partner_tag, scale) for scale != 1.
    std::vector<std::vector<std::pair<std::int64_t, double>>> by_owner;
    /// tag → local owner index (for scale(ti,tj) without local i).
    std::unordered_map<std::int64_t, int> tag_to_owner;

    void clear() {
        by_owner.clear();
        tag_to_owner.clear();
    }

    bool active() const { return s12 != 1.0 || s13 != 1.0 || s14 != 1.0; }

    int n_owners() const { return static_cast<int>(by_owner.size()); }

    /// @brief Rebuild from topology of owner rows [0, n_owners).
    void rebuild(const Frame& f, int n_owners) {
        by_owner.clear();
        tag_to_owner.clear();
        if (!active() || n_owners <= 0)
            return;
        by_owner.resize(static_cast<std::size_t>(n_owners));
        tag_to_owner.reserve(static_cast<std::size_t>(n_owners) * 2);

        const auto nspecial = f.span(kAtom_nspecial);
        const auto special = f.span(kAtom_special);
        const TagInt* tag = f.span(kAtom_tag);

        for (int i = 0; i < n_owners; ++i) {
            tag_to_owner[tag[i]] = i;
            auto& list = by_owner[static_cast<std::size_t>(i)];
            list.clear();
            const int n12 = nspecial(i, 0);
            const int n13 = nspecial(i, 1);
            const int n14 = nspecial(i, 2);
            auto push = [&](int s, double scale) {
                if (scale == 1.0)
                    return;
                const std::int64_t tj = special(i, s);
                if (tj == tag[i])
                    return;
                list.push_back({tj, scale});
            };
            for (int s = 0; s < n12; ++s)
                push(s, s12);
            for (int s = n12; s < n13; ++s)
                push(s, s13);
            for (int s = n13; s < n14; ++s)
                push(s, s14);

            if (list.size() < 2)
                continue;
            std::sort(
                list.begin(), list.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            // Prefer nearer tier if duplicate partners.
            std::size_t w = 0;
            for (std::size_t r = 0; r < list.size(); ++r) {
                if (w == 0 || list[w - 1].first != list[r].first)
                    list[w++] = list[r];
            }
            list.resize(w);
        }
    }

    /// @brief Scale for owner local index i and partner tag tj (default 1).
    double scale_owner(int i, std::int64_t tj) const noexcept {
        if (static_cast<std::size_t>(i) >= by_owner.size())
            return 1.0;
        const auto& list = by_owner[static_cast<std::size_t>(i)];
        if (list.empty())
            return 1.0;
        // Linear scan for tiny k (typical ≲ 12) beats lower_bound overhead.
        if (list.size() <= 16) {
            for (const auto& p : list)
                if (p.first == tj)
                    return p.second;
            return 1.0;
        }
        auto lb = std::lower_bound(list.begin(), list.end(), tj,
                                   [](const std::pair<std::int64_t, double>& p,
                                      std::int64_t t) { return p.first < t; });
        if (lb != list.end() && lb->first == tj)
            return lb->second;
        return 1.0;
    }

    /// @brief Scale for pair tags (ti,tj); default 1.0 when not special.
    double scale(std::int64_t ti, std::int64_t tj) const {
        auto it = tag_to_owner.find(ti);
        if (it == tag_to_owner.end())
            return 1.0;
        return scale_owner(it->second, tj);
    }
};

} // namespace md
} // namespace cpu
} // namespace ATV
