/// @file cpu/md/atom_sort.h
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
/// LAMMPS (`atom_vec.cpp` / `neighbor.cpp`): reordering is *not* domain
/// decomposition. `Atom::sort()` bins owned atoms and permutes the per-proc
/// atom arrays so that atoms in the same spatial bin are adjacent; the call
/// is made from `Neighbor` at the first reneighbor after `Nfreq` steps
/// (`atom_modify sort Nfreq binsize`). That keeps force + neighbor walks
/// cache-friendly without changing ownership.
///
/// Atomiverse mirrors that: `NeighborList` calls #sort_owned_atoms immediately
/// before rebuilding the cell list / CSR. After a reorder it must
/// (1) rebuild `DomainDecomposition::tag_to_local` for owners and
/// (2) bump `neighbor_generation` so `BondedTupleCache` rebuilds local-index
/// tuples (slots moved). Ghost rows are never sorted (they live in
/// `[n_local, n_local+n_ghost)` and are rewritten by halo).
#pragma once

#include <cmath>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/swap_list.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {

// DomainDecomposition lives in md::mpi (see domain.h).
using DomainDecomposition = mpi::DomainDecomposition;

/// @brief LAMMPS `atom_modify sort Nfreq binsize` parameters.
struct AtomSortParams {
    /// @brief Sort at the first reneighbor after this many MD steps.
    ///        LAMMPS default is typically 1000; use 1 to sort every rebuild.
    int nfreq = 1000;
    /// @brief Bin edge [Å]. `≤ 0` → `0.5 * neighbor_cutoff` (LAMMPS auto).
    double binsize = 0.0;
    /// @brief Master enable (LAMMPS `atom_modify sort` vs no-sort).
    bool enabled = true;
};

/// @brief Runtime state for atom_modify sort cadence.
///
/// Multi-rank: owned by `DomainDecomposition` and applied in `HaloSend`
/// **before** borders (LAMMPS: exchange → sort → borders). Serial: applied
/// in `NeighborList` at reneighbor (no halo).
struct AtomSortState {
    AtomSortParams params{};
    /// @brief Global MD step estimate (incremented once per sort opportunity).
    int step = 0;
    /// @brief Next `step` at which sort may fire. Default = default nfreq so
    ///        the first sort is after `nfreq` steps (LAMMPS `nextsort` init),
    ///        not on step 0.
    int next_sort = 1000;

    /// @brief Resolve binsize (Å) from params + neighbor cutoff.
    double resolved_binsize(double neighbor_cutoff) const {
        if (params.binsize > 0.0)
            return params.binsize;
        // LAMMPS: when binsize = 0.0, use 0.5 * neighbor cutoff.
        return 0.5 * neighbor_cutoff;
    }

    /// @brief True when this opportunity should run a spatial sort.
    bool due() const {
        if (!params.enabled || params.nfreq <= 0)
            return false;
        return step >= next_sort;
    }

    /// @brief @deprecated alias for #due (reneighbor-gated callers).
    bool due_at_reneighbor() const { return due(); }

    void mark_sorted() {
        next_sort = step + (params.nfreq > 0 ? params.nfreq : 1);
    }

    /// @brief Apply params and schedule first sort at `nfreq` (not step 0).
    void configure(AtomSortParams p) {
        params = p;
        next_sort = (p.nfreq > 0) ? p.nfreq : 1;
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
/// Call only when ghosts are about to be rebuilt (HaloSend full borders) or
/// when there are no ghosts (serial). Zeroing `n_ghost` without invalidating
/// the swap table makes HaloRecv layout checks fail.
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
/// The swap-table parameter is **mandatory**, without a default, so that
/// every call site has to state which halo table a reorder just invalidated
/// rather than silently forgetting one: the halo path passes its own table,
/// the serial `NeighborList` path passes `nullptr` because no swap table
/// exists there. Non-const: `SwapList::invalidate()` is a mutation of the
/// owner's table, and no `const_cast` is permitted anywhere for it.
///
/// @param dd Domain decomposition (may be null — the DD half is then skipped).
/// @param f  Frame after sort.
/// @param swaps Halo swap table whose stored row indices the reorder just
///        invalidated, or null when this path has no halo (serial).
/// @param clear_ghosts If true, drop ghost bookkeeping (pre-borders path).
inline void notify_sorted(DomainDecomposition* dd, Frame& f,
                          comm::SwapList* swaps, bool clear_ghosts = true) {
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
            dd->n_ghost = 0;
            f.set_nghost(0);
            ++dd->halo_generation; // ghost slots invalid
        }
        const int n = dd->n_local > 0 ? dd->n_local : f.nlocal();
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
    const int n_owned =
        f.nlocal() > 0 ? f.nlocal() : (dd ? dd->n_local : f.nlocal());
    f.clear_tag_index();
    f.rebuild_tag_index(0, n_owned);
}

} // namespace md
} // namespace cpu
} // namespace ATV
