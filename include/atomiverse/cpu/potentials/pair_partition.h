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
///  - how many threads the pair count is worth (`team`), so a small system runs
///    the single-thread direct path instead of waking the whole team;
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

namespace ATV {
namespace cpu {
namespace md {

class PairRowPartition {
  public:
    /// Pairs below which one more thread is not worth its wake-up, its share
    /// of the buffer zero/reduce and the fork/join barrier.
    /// About 100 µs of pair work per thread at a few ns per pair.
    static constexpr std::int64_t kMinPairsPerThread = 49152;
    /// Chunks per thread when buffers are cheap: enough slack for dynamic
    /// pickup to absorb a slow thread.
    static constexpr int kChunksPerThread = 4;
    /// Cap on chunk-buffer rows (chunks x touched rows): past it a large
    /// system gets one chunk per thread, whose balance it does not need.
    static constexpr std::int64_t kMaxChunkRows = std::int64_t{1} << 20;

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

    /// @brief Threads worth using for this list, at most `max_threads`.
    int team(int max_threads) const {
        const std::int64_t by_work = total() / kMinPairsPerThread;
        return static_cast<int>(
            std::clamp<std::int64_t>(by_work, 1, std::max(max_threads, 1)));
    }

    /// @brief Chunks for a team of `team` (1 for a team of one).
    int chunks(int team) const {
        if (team <= 1)
            return 1;
        const std::int64_t many = std::int64_t{team} * kChunksPerThread;
        return many * std::max(n_touch_, 1) <= kMaxChunkRows
                   ? static_cast<int>(many)
                   : team;
    }

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
