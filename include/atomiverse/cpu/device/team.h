/// @file cpu/device/team.h
/// @brief `cpu::device::team_for` — the thread count of a work-sized OpenMP
///        region: the whole team, or the calling thread alone.
///
/// libgomp (GCC's OpenMP runtime) keeps its thread pool at the size of the
/// last team it opened. A region that opens fewer threads than the one before
/// ends the surplus threads, and the next region that opens more starts new
/// ones. Measured on a two-socket EPYC node (GCC 14.3, `OMP_PROC_BIND=close`,
/// `OMP_PLACES=cores`): a steady 64-thread region costs 14 µs and a steady
/// 16-thread one 5 µs, while alternating 64- and 16-thread regions cost
/// 0.9 ms each and alternating 16- and 7-thread ones 0.09 ms each; a region
/// whose `if` clause is false leaves the pool alone. An MD step whose regions
/// each sized their own team from their own work paid that churn on most of
/// its regions, every step.
///
/// So a region sizes nothing but the choice between the two: it opens every
/// thread once its work fills two threads' minimum shares and runs on the
/// calling thread below that. Consecutive threaded regions then always open
/// the same team, whatever their work.
#pragma once

#include <algorithm>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace device {

/// @brief Threads the calling thread's next parallel region opens by default.
inline int max_team() noexcept {
#ifdef _OPENMP
    return std::max(omp_get_max_threads(), 1);
#else
    return 1;
#endif
}

/// @brief Team for a region of `work` units in which one thread repays its
///        fork, join and private buffers at `per_thread` units: all of
///        `team` once the work fills two such shares, otherwise one thread.
/// @param work Units of work in the region (any unit, ≥ 0).
/// @param per_thread Units one thread must get to be worth opening (> 0).
/// @param team The whole team (≥ 1).
inline int team_for(std::int64_t work, std::int64_t per_thread,
                    int team) noexcept {
    return work >= 2 * std::max<std::int64_t>(per_thread, 1) ? std::max(team, 1)
                                                             : 1;
}

/// @brief `team_for(work, per_thread, max_team())`.
inline int team_for(std::int64_t work, std::int64_t per_thread) noexcept {
    return team_for(work, per_thread, max_team());
}

} // namespace device
} // namespace cpu
} // namespace ATV
