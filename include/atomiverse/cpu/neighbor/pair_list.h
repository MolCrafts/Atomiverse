/// @file cpu/neighbor/pair_list.h
/// @brief PairList -- LAMMPS-style CSR neighbor list format for CPU backend.
///
/// Shifted-ghost displacement contract follows LAMMPS src/npair_bin.cpp and
/// src/comm_brick.cpp, checkout a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b.
/// Copyright (2003) Sandia Corporation. Under Contract DE-AC04-94AL85000,
/// the U.S. Government retains certain rights. Distributed under the
/// GNU General Public License (GPL-2.0-only).
#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief LAMMPS-style CSR neighbor list (half or full).
///
/// Force components iterate:
/// @code{.cpp}
/// for (int ii = 0; ii < pairs.inum; ++ii) {
///     int i = pairs.ilist[ii];
///     int jnum = pairs.numneigh[i];
///     int off = pairs.firstneigh[i];
///     for (int jj = 0; jj < jnum; ++jj) {
///         int j = pairs.neighbors[off + jj];
///         // compute pair (i,j)
///     }
/// }
/// @endcode
///
/// **Storage discipline.** All CSR arrays are `std::vector<int>` with
/// amortised `resize()` growth. `NeighborList::launch` writes the first
/// `inum` entries in place on each rebuild, so steady-state rebuilds
/// never enter the allocator once capacity has stabilised.
struct PairList {
    int inum = 0; ///< Number of I atoms with neighbor lists.

    /// @brief Number of populated centre rows, independent of half/full counting.
    ///
    /// `inum` limits force-loop centres to owners. Providers may populate extra
    /// ghost-centred rows only when a consumer explicitly requests them.
    /// SW needs owned full rows; EAM uses an owned half list and communicates
    /// its density and embedding derivatives instead of reconstructing ghosts.
    int n_rows = 0;
    std::vector<int> ilist;    ///< I atom indices [capacity].
    std::vector<int> numneigh; ///< Number of neighbors per atom [capacity].
    std::vector<int>
        firstneigh; ///< Offset into neighbors[] per atom [capacity].
    std::vector<int> neighbors; ///< Flat neighbor indices [total_pairs].
    int total_pairs = 0;        ///< Total entries in neighbors[].

    // Every neighbor j names an explicit Cartesian row, including periodic
    // images prepared by Comm. Displacement is always r_i - r_j; neither
    // consumers nor derived lists wrap or apply a second image transform.

    /// @brief Monotonic rebuild counter. `NeighborList::launch` bumps this by
    /// 1 every time it actually re-enumerates the list. Consumers that cache
    /// derived views (e.g. LJ's exclusion-filtered neighbor list) compare
    /// against their last-seen value to decide whether their cache is stale.
    /// Zero at `NeighborList::build`; never reset.
    unsigned long long version = 0;

    /// @brief Resident heap bytes owned by this CSR. Uses `size()` (live,
    /// written elements ≈ resident pages) rather than `capacity()` — reserved-
    /// but-unwritten slack is lazily mapped and not resident, so `size()` is
    /// the figure that subtracts cleanly against peak RSS in the driver memory
    /// summary.
    std::size_t bytes() const {
        return (ilist.size() + numneigh.size() + firstneigh.size()
                + neighbors.size())
               * sizeof(int);
    }

    /// @brief Zero-overhead pair iteration (half list).
    /// @tparam F Callable with signature void(int i, int j[, int entry]).
    template <typename F>
    void for_each_pair(F&& func) const {
        for (int ii = 0; ii < inum; ++ii) {
            int i = ilist[ii];
            int jnum_i = numneigh[i];
            int offset = firstneigh[i];
            for (int jj = 0; jj < jnum_i; ++jj) {
                int j = neighbors[offset + jj];
                if constexpr (std::is_invocable_v<F&, int, int, int>)
                    func(i, j, offset + jj);
                else
                    func(i, j);
            }
        }
    }

    /// @brief OpenMP-parallel pair iteration.
    /// @tparam F Callable with signature void(int i, int j[, int entry]).
    template <typename F>
    void for_each_pair_omp(F&& func) const {
#pragma omp parallel for schedule(static)
        for (int ii = 0; ii < inum; ++ii) {
            int i = ilist[ii];
            int jnum_i = numneigh[i];
            int offset = firstneigh[i];
            for (int jj = 0; jj < jnum_i; ++jj) {
                int j = neighbors[offset + jj];
                if constexpr (std::is_invocable_v<F&, int, int, int>)
                    func(i, j, offset + jj);
                else
                    func(i, j);
            }
        }
    }
};

} // namespace md
} // namespace cpu
} // namespace ATV
