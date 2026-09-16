/// @file cpu/device/chunked_range.h
/// @brief `cpu::device::ChunkedRange` — a thread-count-independent partition of
///        an index range, the unit of a **reproducible** OpenMP reduction.
///
/// **Why this exists.** Floating-point addition is not associative, so a sum
/// split across threads depends on *how* it was split. OpenMP's own
/// `reduction(+ : x)` and the hand-rolled "per-thread buffer + `#pragma omp
/// critical` merge" both leave that split to the runtime: with a dynamic
/// schedule the chunk→thread assignment varies between two runs of the same
/// binary, and the merge order of the partials varies with thread completion
/// order. The result is a sum that differs run to run at a fixed thread count,
/// and again between thread counts.
///
/// A `ChunkedRange` fixes the split. The range `[0, n)` is cut into `chunks()`
/// contiguous pieces whose boundaries are a function of `n` alone — never of
/// `omp_get_num_threads()`. Each chunk accumulates into its own slot, and the
/// caller then folds the slots in ascending chunk order. The schedule may stay
/// `dynamic` for load balance: which thread runs a chunk no longer influences
/// the arithmetic, because a chunk's partial is never mixed with another's
/// until the fixed-order fold.
///
/// ```cpp
/// const cpu::device::ChunkedRange r(ng, 64);
/// std::vector<double> part(static_cast<std::size_t>(r.chunks()), 0.0);
/// #pragma omp parallel for schedule(dynamic, 1)
/// for (long c = 0; c < r.chunks(); ++c) {
///     double s = 0.0;
///     for (long g = r.begin(c); g < r.end(c); ++g)
///         s += f(g);
///     part[static_cast<std::size_t>(c)] = s;
/// }
/// double total = 0.0;                      // fixed order, any thread count
/// for (double p : part)
///     total += p;
/// ```
///
/// The sum this produces is *not* the sequential left-to-right sum — it is the
/// fold of `chunks()` partial sums. That is intended: it is one specific,
/// reproducible order, which is what a regression gate can hold still. It is
/// also, for large `n`, more accurate than the sequential sum, for the same
/// reason pairwise summation is.
#pragma once

#include <algorithm>
#include <cstddef>

namespace ATV {
namespace cpu {
namespace device {

/// @brief Chunk count for a reproducible reduction whose accumulator is a
///        *block* (a matrix, a grid) rather than a scalar.
///
/// A scalar reduction can afford as many chunks as it likes; a block reduction
/// pays `chunks x block_bytes` of accumulator. This caps the count so that
/// budget stays bounded, as a function of the **block size only** — putting the
/// thread count into it would hand the arithmetic straight back to the runtime,
/// which is the defect this whole header exists to remove.
///
/// @param block_bytes Size of one accumulator block [bytes].
/// @return Chunk count (dimensionless), in `[1, 64]`.
inline long block_reduce_chunks(std::size_t block_bytes) {
    constexpr std::size_t kBudgetBytes = 64u << 20; // 64 MiB of accumulators
    constexpr long kMaxChunks = 64;
    if (block_bytes == 0)
        return kMaxChunks;
    return std::max<long>(
        1, std::min<long>(kMaxChunks,
                          static_cast<long>(kBudgetBytes / block_bytes)));
}

/// @brief Fixed contiguous partition of `[0, n)` into at most `max_chunks`
///        pieces, independent of the OpenMP thread count.
///
/// Chunk `c` covers `[begin(c), end(c))`; the pieces are contiguous, disjoint,
/// and cover the whole range. Sizes differ by at most one element.
class ChunkedRange {
  public:
    /// @brief Partition `[0, n)`.
    /// @param n Range length (dimensionless count); `n <= 0` yields zero chunks.
    /// @param max_chunks Upper bound on the number of pieces (dimensionless);
    ///        clamped to at least 1. Pick it well above the widest thread count
    ///        the range will run on so a dynamic schedule still balances, and
    ///        below whatever per-chunk accumulator memory the caller can hold.
    ChunkedRange(long n, long max_chunks)
        : n_(n > 0 ? n : 0),
          k_(n_ > 0 ? std::min(n_, std::max<long>(1, max_chunks)) : 0) {}

    /// @brief Number of chunks (dimensionless); 0 for an empty range.
    long chunks() const { return k_; }

    /// @brief First index of chunk `c` (dimensionless), `0 <= c < chunks()`.
    long begin(long c) const { return n_ * c / k_; }

    /// @brief One past the last index of chunk `c` (dimensionless).
    long end(long c) const { return n_ * (c + 1) / k_; }

  private:
    long n_; ///< Range length.
    long k_; ///< Chunk count.
};

} // namespace device
} // namespace cpu
} // namespace ATV
