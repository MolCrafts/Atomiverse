/// @file cpu/neighbor/pair_list.h
/// @brief PairList -- LAMMPS-style CSR neighbor list format for CPU backend.
#pragma once

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

    /// @brief Rows that actually carry neighbour data — owners AND ghosts.
    ///
    /// `inum` is trimmed to the OWNER count so a force kernel iterates only atoms it
    /// is responsible for. But the CSR is built for every active row, ghosts included,
    /// and a many-body style needs those: EAM's second pass reads `F'(rho_j)` for a
    /// ghost `j`, so its first pass must fill `rho` for ghost rows too. Without this
    /// field there was no way to ask "how many rows are populated?" without
    /// second-guessing the decomposition.
    int n_rows = 0;
    std::vector<int> ilist;    ///< I atom indices [capacity].
    std::vector<int> numneigh; ///< Number of neighbors per atom [capacity].
    std::vector<int>
        firstneigh; ///< Offset into neighbors[] per atom [capacity].
    std::vector<int> neighbors; ///< Flat neighbor indices [total_pairs].
    int total_pairs = 0;        ///< Total entries in neighbors[].

    /// @brief Per-pair PBC image shift applied to neighbor j, parallel to
    /// `neighbors[]`. The minimum-image displacement is
    /// `d = x_i - (x_j + shift_j)`, so force kernels skip the per-pair
    /// min-image when `has_shifts` is set. The shift is fixed for the list's
    /// lifetime (the skin guarantees the chosen image stays valid between
    /// rebuilds) -- this is LAMMPS's ghost-image trick encoded as a shift.
    /// Only the single-rank serial build populates these today; other build
    /// paths leave `has_shifts=false` and consumers fall back to min-image.
    std::vector<double> shift_x;
    std::vector<double> shift_y;
    std::vector<double> shift_z;
    bool has_shifts = false; ///< True when shift_{x,y,z} are populated.

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
                   * sizeof(int)
               + (shift_x.size() + shift_y.size() + shift_z.size())
                     * sizeof(double);
    }

    /// @brief Zero-overhead pair iteration (half list).
    /// @tparam F Callable with signature void(int i, int j).
    template <typename F>
    void for_each_pair(F&& func) const {
        for (int ii = 0; ii < inum; ++ii) {
            int i = ilist[ii];
            int jnum_i = numneigh[i];
            int offset = firstneigh[i];
            for (int jj = 0; jj < jnum_i; ++jj) {
                int j = neighbors[offset + jj];
                func(i, j);
            }
        }
    }

    /// @brief OpenMP-parallel pair iteration.
    /// @tparam F Callable with signature void(int i, int j).
    template <typename F>
    void for_each_pair_omp(F&& func) const {
#pragma omp parallel for schedule(static)
        for (int ii = 0; ii < inum; ++ii) {
            int i = ilist[ii];
            int jnum_i = numneigh[i];
            int offset = firstneigh[i];
            for (int jj = 0; jj < jnum_i; ++jj) {
                int j = neighbors[offset + jj];
                func(i, j);
            }
        }
    }
};

} // namespace md
} // namespace cpu
} // namespace ATV
