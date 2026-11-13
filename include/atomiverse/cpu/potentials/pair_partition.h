/// @file cpu/potentials/pair_partition.h
/// @brief Thread team and row partition for a half pair-list kernel.
///
/// A half list's rows are far from equal: under an all-pairs (or long-cutoff)
/// list row i holds roughly N - i partners, so `schedule(static)` over rows
/// hands the first thread most of the work. And a kernel with per-thread force
/// buffers pays a zero and a reduce pass over every row it may touch, per
/// thread, per launch — over `n_local + n_ghost` rows, which for a long cutoff
/// in a small box is many times the atom count even when no ghost has a pair.
///
/// `PairRowPartition` is rebuilt whenever the list a kernel walks changes
/// (the neighbor list's version, or a kernel's own filtered copy) and answers:
///  - whether the pair count is worth the whole team against a thread's
///    fixed cost, its fork and its chunk buffer over the touched rows
///    (`team`), so a small system runs the single-thread direct path instead
///    of waking the team;
///  - how many chunks to cut the rows into (`chunks`) and which contiguous
///    rows each chunk takes so the pair counts balance (`range`);
///  - the highest row the list references (`n_touch`), which bounds the
///    buffer zero/reduce passes.
///
/// `md::PairLoop` hands chunks out dynamically (a thread on a slower core, or
/// one the OS time-shares with other work, simply takes fewer), but every
/// chunk owns its `md::ForcePass` slot and its energy/virial partials, and
/// those are summed in chunk order. The result therefore depends on the chunk
/// count only — never on which thread ran which chunk — so a run is
/// bit-reproducible.
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "atomiverse/cpu/device/team.h"

namespace ATV {
namespace cpu {
namespace md {

class PairRowPartition {
  public:
    /// Fixed cost of one more thread (wake-up, its share of the fork, the
    /// barriers and the join), in half-list entries: ≈ 2–4 µs at the
    /// ≈ 1–2 ns a chunked entry costs a cheap kernel.
    static constexpr std::int64_t kForkEntries = 2048;
    /// Cost of one row of a chunk's force buffer, in half-list entries: it is
    /// zeroed on the chunk's first touch and reduced by the flush (≈ 48
    /// bytes of traffic, about one entry's geometry).
    static constexpr std::int64_t kEntriesPerBufferRow = 1;
    /// Most chunks per thread, when buffers are cheap: enough slack for
    /// dynamic pickup to absorb a slow thread.
    static constexpr int kChunksPerThread = 4;
    /// A thread takes extra chunks only while their buffers stay below
    /// 1/kBufferShare of its entries.
    static constexpr std::int64_t kBufferShare = 8;

    /// @brief Team for `total` half-list entries that scatter into rows
    ///        `[0, n_touch)`: all `max_threads`, or one (`device::team_for`).
    ///
    /// Every thread costs a fork and at least one chunk buffer of `n_touch`
    /// rows whatever its share, so the list goes threaded only once two
    /// threads' shares of the entries each cover that fixed cost. A small
    /// dense system with a long cutoff (few atoms, many ghost rows)
    /// therefore stays serial longer than a large one with the same entry
    /// count.
    static int team_for(std::int64_t total, int n_touch, int max_threads) {
        return device::team_for(total,
                                kForkEntries
                                    + std::int64_t{std::max(n_touch, 1)}
                                          * kEntriesPerBufferRow,
                                max_threads);
    }

    /// @brief Chunks for `total` entries over a team of `team` (1 for a
    ///        team of one): a multiple of the team, up to kChunksPerThread
    ///        per thread while their buffers stay a small share of the work.
    static int chunks_for(std::int64_t total, int n_touch, int team) {
        if (team <= 1)
            return 1;
        const std::int64_t buffer = kBufferShare
                                    * std::int64_t{std::max(n_touch, 1)}
                                    * kEntriesPerBufferRow;
        const std::int64_t per_thread = total / team;
        const auto per = static_cast<int>(
            std::clamp<std::int64_t>(per_thread / buffer, 1, kChunksPerThread));
        return team * per;
    }

    /// @brief Rebuild from a CSR half list.
    /// @param inum       Number of rows walked.
    /// @param ilist      Row index of the ii-th walked row.
    /// @param numneigh   Partners per row (indexed by row).
    /// @param firstneigh Offset of each row's partners in `neighbors`.
    /// @param neighbors  Flat partner rows.
    void rebuild(int inum, const int* ilist, const int* numneigh,
                 const int* firstneigh, const int* neighbors) {
        prefix_.assign(static_cast<std::size_t>(inum) + 1, 0);
        int touch = 0;
        for (int ii = 0; ii < inum; ++ii) {
            const int i = ilist[ii];
            const int jn = numneigh[i];
            const int* nb = neighbors + firstneigh[i];
            touch = std::max(touch, i + 1);
            for (int jj = 0; jj < jn; ++jj)
                touch = std::max(touch, nb[jj] + 1);
            prefix_[static_cast<std::size_t>(ii) + 1] =
                prefix_[static_cast<std::size_t>(ii)] + jn;
        }
        n_touch_ = touch;
    }

    /// @brief Total pairs in the list.
    std::int64_t total() const { return prefix_.empty() ? 0 : prefix_.back(); }

    /// @brief One past the highest row (owner or partner) the list names.
    int n_touch() const { return n_touch_; }

    /// @brief Team for this list: `max_threads` or one.
    int team(int max_threads) const {
        return team_for(total(), n_touch_, max_threads);
    }

    /// @brief Chunks for a team of `team` (1 for a team of one).
    int chunks(int team) const { return chunks_for(total(), n_touch_, team); }

    /// @brief Walked rows `[begin, end)` of chunk `t` out of `team` chunks,
    ///        cut so every chunk gets about `total() / team` pairs.
    std::pair<int, int> range(int t, int team) const {
        const int inum = static_cast<int>(prefix_.size()) - 1;
        if (inum <= 0 || team <= 1)
            return {0, std::max(inum, 0)};
        return {cut_(t, team), cut_(t + 1, team)};
    }

  private:
    int cut_(int t, int team) const {
        const int inum = static_cast<int>(prefix_.size()) - 1;
        if (t <= 0)
            return 0;
        if (t >= team)
            return inum;
        const std::int64_t target = total() * t / team;
        const auto it =
            std::lower_bound(prefix_.begin(), prefix_.end(), target);
        return static_cast<int>(it - prefix_.begin());
    }

    std::vector<std::int64_t> prefix_;
    int n_touch_ = 0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
