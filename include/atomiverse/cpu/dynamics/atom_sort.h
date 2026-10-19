/// @file cpu/dynamics/atom_sort.h
/// @brief LAMMPS `atom_modify sort` — spatial reordering of **owned** atoms.
///
/// ## Placement (copy LAMMPS)
///
/// | Concern | Owns |
/// |---|---|
/// | **Which rank owns which atoms** | `DomainDecomposition` (MPI brick / RCB) |
/// | **Spatial sort within a rank** | **this module**, triggered at **reneighbor** |
/// | **Neighbor CSR construction** | `NeighborList` / `CellList` |
///
/// LAMMPS `src/verlet.cpp` invokes `Atom::sort()` between exchange and borders
/// at the first reneighbor after Nfreq steps (`atom_modify sort Nfreq binsize`).
/// Runtime follows this ordering for serial and MPI. Reordering owned rows
/// rebuilds tag maps and invalidates local-index caches before ghost creation.
/// Port reference: a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b.
/// Copyright (2003) Sandia Corporation. Under Contract DE-AC04-94AL85000,
/// the U.S. Government retains certain rights. GPL-2.0-only.
#pragma once

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {

// DomainDecomposition lives in md::mpi (see domain.h).
using DomainDecomposition = mpi::DomainDecomposition;

/// @brief Runtime state for LAMMPS `atom_modify sort Nfreq binsize`.
///
/// Owned by `NeighborList`, published non-owning as `ctx.md.atom_sort`.
/// Runtime applies it between exchange and borders for every topology.
/// The clock is the Driver's `LaunchContext::step`, not a
/// counter incremented on every communication opportunity — Minimize
/// line-search re-evaluates forces at a frozen step and must not walk
/// the cadence.
struct AtomSortState {
    /// @brief Sort at the first reneighbor after this many MD steps.
    ///        LAMMPS default is typically 1000; use 1 to sort every rebuild.
    int nfreq = 1000;
    /// @brief Bin edge [Å]. `≤ 0` → `0.5 * neighbor_cutoff` (LAMMPS auto).
    double binsize = 0.0;
    /// @brief Master enable (LAMMPS `atom_modify sort` vs no-sort).
    bool enabled = true;
    /// @brief The loop's current step, stamped from `LaunchContext::step`.
    int step = 0;
    /// @brief Next sort step; zero requests sorting during initial setup.
    int next_sort = 0;

    /// @brief Resolve binsize (Å) from `binsize` + neighbor cutoff.
    double resolved_binsize(double neighbor_cutoff) const {
        if (binsize > 0.0)
            return binsize;
        // LAMMPS: when binsize = 0.0, use 0.5 * neighbor cutoff.
        return 0.5 * neighbor_cutoff;
    }

    /// @brief True when this opportunity should run a spatial sort.
    bool due() const {
        if (!enabled || nfreq <= 0)
            return false;
        return step >= next_sort;
    }

    void mark_sorted() { next_sort = step + (nfreq > 0 ? nfreq : 1); }

    /// @brief Apply cadence and request sorting during initial setup.
    /// @param nfreq_ Sort at the first reneighbor after this many steps.
    /// @param binsize_ Bin edge [Å]; `≤ 0` → auto from neighbor cutoff.
    /// @param enabled_ Master enable.
    void configure(int nfreq_, double binsize_ = 0.0, bool enabled_ = true) {
        nfreq = nfreq_;
        binsize = binsize_;
        enabled = enabled_;
        next_sort = 0;
    }
};

/// @brief Spatially bin owned atoms `[0, n_local)` and reorder migrate columns.
///
/// Only touches `kColMigrate` columns via `Block::copy_row` cycles (same
/// migration discipline as halo/migrate). Ghost rows are left in place.
///
/// @param f        Frame (mutated).
/// @param n_local  Owner count (do not include ghosts).
/// @param binsize  Spatial bin edge [Å] (must be > 0).
/// @return true if any atom moved (false if already sorted / n_local < 2).
bool sort_owned_atoms(Frame& f, int n_local, double binsize);

/// @brief After a successful sort: rebuild owner tag map + bump generation.
///
/// Call only when ghosts are about to be rebuilt (CommBrick full borders) or
/// when there are no ghosts (serial). Zeroing `n_ghost` without invalidating
/// the swap table makes CommBrick layout checks fail.
///
/// A reorder invalidates every cached row index in the process, and the two
/// places that hold them are the DD's `tag_to_local` and the **Frame's own tag
/// index**. Those are not the same lifetime: the Frame index is published even
/// when no decomposition exists, and a component that resolves tags per step
/// (`FixSettle`) reads it, not the DD's. So the Frame side is republished
/// unconditionally — an early return on `dd == nullptr` left it naming the
/// pre-sort rows, which is silent corruption rather than a failure. (No
/// in-tree pipeline reaches here with a null `dd` today; that is exactly why
/// it would not have been noticed.)
///
/// Runtime invokes this after exchange and sorting, immediately before borders
/// reconstructs the communicator's send lists. A direct caller can supply its
/// Halo to invalidate stored row indices immediately.
///
/// @param dd Domain decomposition (may be null — the DD half is then skipped).
/// @param f  Frame after sort.
/// @param swaps Halo swap table whose stored row indices the reorder just
///        invalidated, or null when borders immediately rebuilds it.
/// @param clear_ghosts If true, drop ghost bookkeeping (pre-borders path).
inline void notify_sorted(DomainDecomposition* dd, Frame& f, comm::Halo* swaps,
                          bool clear_ghosts = true) {
    // Sendlists / ghost counts name rows that have just moved; the swap
    // geometry (faces, neighbours, pbc flags, slab bands) survives a
    // reorder untouched.
    if (clear_ghosts && swaps)
        swaps->invalidate();
    if (dd) {
        // Rebuild owner half of tag_to_local; drop the ghost half when borders
        // redo it.
        dd->tag_to_local.clear();
        if (clear_ghosts) {
            dd->ghost_tags.clear();
            f.set_nghost(0);
            ++dd->halo_generation; // ghost slots invalid
        }
        const int n = f.nlocal();
        // Always rebuild owner tag map after reorder (bonded + gather).
        dd->tag_to_local.reserve(static_cast<std::size_t>(n));
        const TagInt* tag = f.span(kAtom_tag);
        for (int i = 0; i < n; ++i)
            dd->tag_to_local.emplace(tag[i], i);
        // Invalidate BondedTupleCache and any other local-index caches.
        ++dd->neighbor_generation;
    } else if (clear_ghosts) {
        f.set_nghost(0);
    }
    const int n_owned = f.nlocal();
    f.clear_tag_index();
    f.rebuild_tag_index(0, n_owned);
}

} // namespace md
} // namespace cpu
} // namespace ATV
