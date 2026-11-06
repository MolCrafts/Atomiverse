/// @file cpu/neighbor/cell_list.h
/// @brief CellList -- LAMMPS-style linked-cell spatial index for CPU backend.
///
/// Cartesian bins and distance-pruned stencils over explicit Frame rows.
/// Comm supplies periodic ghost coordinates; the index never wraps coordinates
/// or manufactures images. Owned rows are centres, ghosts are neighbours.
/// DD subdomain bounds are fractional; the grid is their Cartesian bounding
/// box padded by the greater of neighbor reach and communication shell.
// Thread-owned row arenas and full-to-half derivation follow LAMMPS
// src/OPENMP/npair_bin_omp.cpp, src/npair_halffull.cpp,
// src/nbin_standard.cpp, src/nstencil_bin.cpp and src/npair_bin.cpp,
// a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b. Copyright (2003) Sandia
// Corporation. Under Contract DE-AC04-94AL85000, the U.S. Government retains
// certain rights. Distributed under the GNU General Public License.
#pragma once

#include <vector>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

namespace ATV {
namespace cpu {
namespace md {

namespace mpi {
struct DomainDecomposition;
} // namespace mpi

/// @brief Cell list spatial index -- LAMMPS bin style.
///
/// Algorithm:
/// 1. Bin raw atom/image rows into cells of approximately half the reach
/// 2. Precompute stencil of cell offsets
/// 3. Build CSR half-neighbor list via stencil search
class CellList {
  public:
    /// @param cutoff Interaction cutoff [Angstrom].
    /// @param skin Verlet skin width [length]. Negative = the LAMMPS default
    ///        of the run's `units` (lj 0.3 until a driver applies its style
    ///        through `set_skin`).
    explicit CellList(double cutoff, double skin = -1.0);

    /// @brief Replace the skin (the driver's LAMMPS `units` default).
    void set_skin(double skin);

    /// Prepare bins around the complete fractional cell plus reach. Input
    /// positions are raw Cartesian rows; periodic images must already exist.
    void init(const cpu::Frame& frame, double cutoff_skin);

    /// Prepare a local Cartesian window around DD's fractional subdomain.
    /// Counts/capacity belong to Frame; DD supplies only partition geometry.
    void init_extended(const cpu::Frame& frame, double cutoff_skin,
                       const mpi::DomainDecomposition* dd, int capacity);

    /// @brief Full rebuild: bin atoms, build stencil.
    /// @param frame Frame to bin.
    /// @param n_active Number of active atoms (owners + ghosts) to iterate.
    ///        `-1` means `frame.n_total()`.
    void build(const cpu::Frame& frame, int n_active = -1);

    /// @brief True when the grid extents no longer match the live box /
    /// subdomain bounds (NPT rescale or RCB rebalance since last build).
    ///
    /// This is the predicate `build()` uses to resize bins. It is true for
    /// any cell-matrix change, including a Berendsen step of order 1e-4 Å.
    /// That is not a reason to throw the Verlet list away: see
    /// `geometry_forces_rebuild`.
    bool grid_stale(const cpu::Frame& frame) const;

    /// @brief Largest Cartesian change of a lattice vector since the last
    /// `build()` [Angstrom]. Zero when the cell matrix matches.
    double lattice_drift(const cpu::Frame& frame) const;

    /// @brief Geometry the Verlet skin cannot absorb, so the list must be
    /// rebuilt before the next force.
    ///
    /// A cell rescaling whose lattice vectors moved by at most `skin()`
    /// returns false. NPT updates the box every step; treating that as a
    /// stale grid forced a full binning, halo, and pair-list rebuild on
    /// every step and disabled `neigh_modify delay/check`. Halo-pad and
    /// subdomain changes still return true. `build()` consults
    /// `grid_stale()` and resizes the bins whenever a rebuild does run.
    bool geometry_forces_rebuild(const cpu::Frame& frame) const;

    /// @brief Produce LAMMPS-style half-neighbor list (CSR) into a fresh
    /// `PairList`. Newton on. Thin wrapper around `refresh_half` for
    /// one-shot callers; steady-state rebuilds should use `refresh_half`
    /// directly so their `PairList`'s amortized buffer capacity is
    /// preserved across calls.
    /// @param frame Frame for distance evaluation.
    /// @param n_active Outer-loop / array-size bound; `-1` = `frame.n_total()`.
    ///        Caller (e.g. `NeighborList`) typically trims `inum = n_local` after.
    PairList build_half(const cpu::Frame& frame, int n_active = -1) const;

    /// @brief Produce full-neighbor list (CSR) into a fresh `PairList`. See
    /// `build_half` for the capacity-preservation rationale.
    /// @param frame Frame for distance evaluation.
    /// @param n_active Outer-loop / array-size bound; `-1` = `frame.n_total()`.
    PairList build_full(const cpu::Frame& frame, int n_active = -1) const;

    /// @brief Rebuild the half-neighbor list **in place** into an existing
    /// `PairList`. Reuses `out.ilist / numneigh / firstneigh / neighbors`
    /// allocations: the xtensor arrays are sized to `max(n_active, prior
    /// capacity)` via xtensor's resize semantics (never shrinks); the
    /// `std::vector<int>` neighbors arena grows via `resize()` with the
    /// standard amortized-O(1) reallocation policy, so steady-state calls
    /// are allocation-free once a typical neighbor count has been reached.
    /// @param frame Frame for distance evaluation.
    /// @param n_active Number of i-atoms this rebuild should cover.
    /// @param out PairList whose buffers are overwritten in place.
    /// @param n_local Owned centres; -1 uses frame.nlocal(). Ghost rows
    /// participate as neighbours and Newton ownership uses image coordinates.
    void refresh_half(const cpu::Frame& frame, int n_active, PairList& out,
                      int n_local = -1) const;

    /// @brief Rebuild the full-neighbor list **in place** into an existing
    /// `PairList`. See `refresh_half` for the capacity-preservation
    /// rationale.
    /// @param frame Frame for distance evaluation.
    /// @param n_active Number of active atoms included in the spatial index.
    /// @param out PairList whose buffers are overwritten in place.
    /// @param n_rows Number of centre rows to enumerate; -1 means all active atoms.
    void refresh_full(const cpu::Frame& frame, int n_active, PairList& out,
                      int n_rows = -1) const;

    /// @brief Derive Newton-on half rows from an existing full list, without
    /// revisiting spatial bins (LAMMPS NPairHalffull).
    /// @param frame Coordinates used to select the owner of a ghost pair.
    /// @param n_active Number of owned and ghost atoms in the spatial index.
    /// @param full Full list covering every owned centre at the same reach.
    /// @param out Distinct output list, with its storage retained across calls.
    /// @param n_local Owned atom count; -1 means all atoms are local.
    void derive_half(const cpu::Frame& frame, int n_active,
                     const PairList& full, PairList& out,
                     int n_local = -1) const;

    /// @brief Flat bin index a frame row was assigned at the last
    /// `build()`, i.e. `cx * ncy * ncz + cy * ncz + cz` in this rank's
    /// grid (which, on decomposed axes, is a lattice-aligned window around
    /// the subdomain — bin 0 is not the box origin there).
    ///
    /// Read-only introspection on the binning, the one observable that
    /// distinguishes a correctly padded window from one whose out-of-window
    /// rows were clamped into the edge cell; the published `PairList` looks
    /// the same either way because the distance filter rejects the merged
    /// candidates. The grid is observable independently of list construction.
    /// @param row Frame row index, in `[0, n_active)` of the last `build()`.
    /// @return Flat cell index in `[0, total_cells)`; unspecified before
    ///         the first `build()`.
    int cell_of(int row) const { return atom_cell_[row]; }

    /// @brief Verlet skin width [Angstrom] currently held by this strategy.
    /// Used by infrastructure auto-injection (e.g. MD::install_defaults
    /// extending the halo cutoff by skin so atoms in the skin band stay
    /// ghosts across full-rebuild boundaries — without this, ghosts
    /// flicker in/out of the cutoff as atoms drift, injecting per-rebuild
    /// energy jumps into the NPT extended Hamiltonian).
    double skin() const { return skin_; }

    /// @brief Resident heap bytes owned by the cell index (per-atom bin
    /// bookkeeping, the cell count/offset arrays, stencils, and the multi-thread scratch). Consumed by the built-in
    /// driver memory summary (the "Neighbor" category, with the CSR PairList).
    std::size_t bytes() const;

  private:
    double cutoff_, skin_, cutoff_skin_;

    // Grid
    int ncx_ = 0, ncy_ = 0, ncz_ = 0;    ///< Cells per axis.
    double csx_ = 0, csy_ = 0, csz_ = 0; ///< Cell dimensions.
    int total_cells_ = 0;
    bool triclinic_ = false;
    // Cartesian global-bin origin; zero on the orthorhombic fast path.
    double bbox_lo_[3] = {0, 0, 0};
    double bbox_extent_[3] = {0, 0, 0};
    /// @brief Global bin index of the first local cell on each axis.
    /// Coordinates are binned globally before subtracting this integer offset.
    int bin_lo_[3] = {0, 0, 0};
    int global_bins_[3] = {0, 0, 0}; ///< Canonical box bin counts.
    /// @brief Bookkeeping capacity. Per-atom arrays sized to this; ≥ any
    /// `n_active` ever passed.
    int capacity_ = 0;

    /// @brief Domain decomposition (multi-rank only; nullptr single-rank).
    /// Extended axes derive the grid extent from `dd_->sublo/subhi`.
    const mpi::DomainDecomposition* dd_ = nullptr;

    /// @brief `dd_->sublo/subhi` captured at the last `compute_grid()`.
    /// `grid_stale()` compares against the live values so NPT rescaling
    /// and RCB rebalancing both trigger a grid re-derivation.
    double sublo_at_grid_[3] = {0.0, 0.0, 0.0};
    double subhi_at_grid_[3] = {0.0, 0.0, 0.0};

    /// @brief `halo_pad_()` captured at the last `compute_grid()`
    /// [Angstrom]. `grid_stale()` compares against the live value so a
    /// halo that deepens its ghost shell mid-run (a shrinking box raising
    /// the hop count, a bonded extension growing) re-derives the window
    /// instead of binning into a now-too-narrow one. Not a latch: dropping
    /// the shell back makes the grid fresh again.
    double halo_pad_at_grid_ = 0.0;

    /// @brief Extended-axis window margin [Angstrom]: the deeper of this
    /// index's Verlet reach (`cutoff_skin_`) and the halo's real ghost
    /// shell (`dd_->ghost_shell_width`). Single-rank (`dd_ == nullptr`)
    /// is the Verlet reach alone.
    /// @return Margin added on each side of the subdomain [Angstrom].
    double halo_pad_() const noexcept;

    // Binning
    xt::xtensor<int, 1> atom_cell_;    ///< Atom -> cell index [capacity].
    xt::xtensor<int, 1> cell_count_;   ///< Atoms per cell [total_cells].
    xt::xtensor<int, 1> cell_offset_;  ///< Prefix sum [total_cells].
    xt::xtensor<int, 1> binned_atoms_; ///< Sorted atom indices [capacity].

    // Cartesian stencil offsets: interior rows add the cached flat offset;
    // boundary rows use the triples for per-axis wrapping or clipping.
    /// @brief Stencil offset triple. Used by `build_half`/`build_full`
    /// together with each cell's (cx,cy,cz) decomposition + PBC wrap.
    struct StencilOffset {
        int di;
        int dj;
        int dk;
        int flat; // LAMMPS ibin + stencil[k], for interior Cartesian rows
    };
    std::vector<StencilOffset> half_stencil_;
    std::vector<StencilOffset> full_stencil_;
    /// @brief Per-axis stencil reach in cells (`ceil(cutoff_skin / cs_d)`);
    /// 2 with the half-reach bins, 1 on the nc==1 floor.
    int stencil_reach_[3] = {1, 1, 1};
    /// Cell matrix defining the current Cartesian grid.
    Mat3 box_at_last_build_{};

    /// @brief Thread-owned contiguous neighbor rows, retained between rebuilds.
    /// Only neighbor j is stored: each thread owns a contiguous centre range,
    /// and firstneigh holds row offsets into this arena until final assembly.
    mutable std::vector<std::vector<int>> thread_neighbors_;
    /// @brief Prefix offsets of the thread arenas in the final public CSR.
    mutable std::vector<int> thread_offsets_;

    /// @brief Per-rebuild bin-scatter cursor: a `std::vector<int>` of
    /// length `total_cells_` reused across `bin_atoms` calls. Sized
    /// monotonically in `bin_atoms` itself; the contents are
    /// overwritten from `cell_offset_` at the start of each call so
    /// the per-cell scatter offsets stay in lockstep with the prefix
    /// sum. Replaces the per-call `auto write_ptr = cell_offset_;`
    /// xtensor copy that previously leaked one heap allocation per
    /// neighbor-list rebuild into the MD hot loop.
    std::vector<int> bin_write_ptr_;

    void compute_grid(const cpu::Frame& frame);
    void bin_atoms(const cpu::Frame& frame, int n_active);
    void build_stencil();
};

} // namespace md
} // namespace cpu
} // namespace ATV
