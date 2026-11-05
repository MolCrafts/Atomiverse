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
/// Kernels hand chunks out dynamically (a thread on a slower core, or one the
/// OS time-shares with other work, simply takes fewer), but every chunk owns
/// its force buffer and its energy/virial slot, and those are summed in chunk
/// order. The result therefore depends on the chunk count only — never on
/// which thread ran which chunk — so a run is bit-reproducible.
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

/// @brief Per-chunk force buffers and energy/virial sums for a pair kernel
///        run in `PairRowPartition` chunks.
///
/// Each chunk scatters into its own buffer rows `[0, n_touch)` and writes its
/// own sums; `reduce_forces` and `sum` then combine them in chunk order, so
/// the result does not depend on which thread took which chunk. Storage only
/// grows (no allocation once the chunk count and touched rows stabilise).
class PairChunkScratch {
  public:
    /// Energy + Voigt virial (xx yy zz xy xz yz) per chunk.
    static constexpr int kSums = 7;

    /// @brief Size for `n_chunks` chunks over `n_touch` rows and clear the
    ///        sums. Call before the parallel region.
    void prepare(int n_chunks, int n_touch) {
        if (static_cast<int>(fx_.size()) < n_chunks) {
            fx_.resize(n_chunks);
            fy_.resize(n_chunks);
            fz_.resize(n_chunks);
        }
        for (int c = 0; c < n_chunks; ++c) {
            if (static_cast<int>(fx_[c].size()) < n_touch) {
                fx_[c].resize(n_touch);
                fy_[c].resize(n_touch);
                fz_[c].resize(n_touch);
            }
        }
        sums_.assign(static_cast<std::size_t>(n_chunks) * kSums, 0.0);
        n_chunks_ = n_chunks;
        n_touch_ = n_touch;
    }

    /// @brief Chunk `c`'s force rows, zeroed. Call by the thread running it.
    void begin_chunk(int c, double*& fx, double*& fy, double*& fz) {
        fx = fx_[c].data();
        fy = fy_[c].data();
        fz = fz_[c].data();
        std::fill(fx, fx + n_touch_, 0.0);
        std::fill(fy, fy + n_touch_, 0.0);
        std::fill(fz, fz + n_touch_, 0.0);
    }

    /// @brief Chunk `c`'s energy + virial slot (`kSums` doubles).
    double* sums(int c) { return sums_.data() + std::size_t(c) * kSums; }

    /// @brief Add every chunk's rows into the frame, chunk order per row.
    ///        Work-shares over rows: call from inside the parallel region,
    ///        after a barrier that follows the last chunk.
    void reduce_forces(double* fx, double* fy, double* fz) const {
#pragma omp for schedule(static)
        for (int i = 0; i < n_touch_; ++i) {
            double sx = 0, sy = 0, sz = 0;
            for (int c = 0; c < n_chunks_; ++c) {
                sx += fx_[c][i];
                sy += fy_[c][i];
                sz += fz_[c][i];
            }
            fx[i] += sx;
            fy[i] += sy;
            fz[i] += sz;
        }
    }

    /// @brief Sum slot `k` over the chunks, in chunk order.
    double sum(int k) const {
        double s = 0;
        for (int c = 0; c < n_chunks_; ++c)
            s += sums_[std::size_t(c) * kSums + k];
        return s;
    }

    std::size_t memory_bytes() const {
        std::size_t n = sums_.capacity() * sizeof(double);
        for (const auto* v : {&fx_, &fy_, &fz_})
            for (const auto& b : *v)
                n += b.capacity() * sizeof(double);
        return n;
    }

    void clear() {
        fx_.clear();
        fy_.clear();
        fz_.clear();
        sums_.clear();
        n_chunks_ = 0;
        n_touch_ = 0;
    }

  private:
    std::vector<std::vector<double>> fx_, fy_, fz_;
    std::vector<double> sums_;
    int n_chunks_ = 0;
    int n_touch_ = 0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
