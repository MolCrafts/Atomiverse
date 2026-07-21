/// @file cpu/neighbor/cell_list.h
/// @brief CellList -- LAMMPS-style linked-cell spatial index for CPU backend.
///
/// O(N) neighbor list construction via cell binning + stencil search.
/// Produces LAMMPS-style CSR PairList (half or full).
///
/// **Multi-rank ghost-aware mode** (`init_extended`):
/// the bin grid is widened by `cutoff_skin` on each side along every axis
/// `d` with `dims[d] > 1`, so ghost atoms whose positions lie outside the
/// box (after PBC pre-shift by `HaloSend`) bin into dedicated
/// halo cells instead of wrapping back onto owner cells. Iteration bounds
/// are explicit: `bin_atoms` walks `[0, n_active)` (= owners + current
/// ghosts), and the half-list outer loop walks `[0, n_active)` so callers
/// can later trim `inum = n_local` for force kernels.
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
/// 1. Bin atoms into cells of side >= cutoff + skin
/// 2. Precompute stencil of cell offsets
/// 3. Build CSR half-neighbor list via stencil search
/// 4. Track max displacement for rebuild check
class CellList {
  public:
    /// @param cutoff Interaction cutoff [Angstrom].
    /// @param skin Verlet skin width [Angstrom]. Negative = auto (0.3 * cutoff).
    explicit CellList(double cutoff, double skin = -1.0);

    /// @brief One-time setup (single-rank): grid spans `[0, L)` per axis,
    /// ghost positions are wrapped during binning.
    /// @param frame Frame whose box defines the grid.
    /// @param cutoff_skin Interaction cutoff plus Verlet skin [Angstrom].
    void init(const cpu::Frame& frame, double cutoff_skin);

    /// @brief One-time setup (multi-rank): along every axis `d` with
    /// `dims[d] > 1` the grid covers only THIS RANK's subdomain plus a
    /// `cutoff_skin` halo margin on each side — `[sublo[d] - cutoff_skin,
    /// subhi[d] + cutoff_skin)` — not the global box. Ghost atoms (PBC
    /// pre-shifted by `HaloSend`) bin into the halo cells without
    /// wrap-around. Single-rank axes (`dims[d] == 1`) keep the global
    /// `[0, L)` range and are wrapped during binning.
    ///
    /// Sizing the grid to the subdomain keeps both the cell arrays and the
    /// per-rebuild cell loops O(local atoms): the previous global-box grid
    /// cost every rank O(N_total) cells — ~0.3 MB and three full sweeps per
    /// added rank at 200k atoms/rank, the dominant per-rank RSS growth and
    /// the multi-node rebuild bottleneck at 512-1024 ranks.
    /// @param frame Frame whose box defines the grid.
    /// @param cutoff_skin Interaction cutoff plus Verlet skin [Angstrom].
    /// @param dd Domain decomposition (dims + this rank's sublo/subhi).
    ///        Borrowed pointer; must outlive this CellList's use. The grid
    ///        re-derives itself from `dd->sublo/subhi` whenever they move
    ///        (NPT rescale, RCB rebalance).
    /// @param capacity Maximum number of active atoms (owners + ghost slack);
    ///        per-atom bookkeeping arrays are sized to this.
    void init_extended(const cpu::Frame& frame, double cutoff_skin,
                       const mpi::DomainDecomposition* dd, int capacity);

    /// @brief Full rebuild: bin atoms, build stencil.
    /// @param frame Frame to bin.
    /// @param n_active Number of active atoms (owners + ghosts) to iterate.
    ///        `-1` means `frame.nlocal()` (single-rank legacy).
    void build(const cpu::Frame& frame, int n_active = -1);

    /// @brief Check if any atom moved more than skin/2.
    /// @param frame Frame whose positions are compared to last `build()`.
    /// @param n_active Number of active atoms to scan; `-1` = `frame.nlocal()`.
    bool needs_rebuild(const cpu::Frame& frame, int n_active = -1) const;

    /// @brief Rebuild if needed, else no-op.
    /// @param frame Frame to potentially rebin.
    /// @param n_active Number of active atoms; `-1` = `frame.nlocal()`.
    void update(const cpu::Frame& frame, int n_active = -1);

    /// @brief Produce LAMMPS-style half-neighbor list (CSR) into a fresh
    /// `PairList`. Newton on. Thin wrapper around `refresh_half` for
    /// one-shot callers; steady-state rebuilds should use `refresh_half`
    /// directly so their `PairList`'s amortized buffer capacity is
    /// preserved across calls.
    /// @param frame Frame for distance evaluation.
    /// @param n_active Outer-loop / array-size bound; `-1` = `frame.nlocal()`.
    ///        Caller (e.g. `NeighborList`) typically trims `inum = n_local` after.
    PairList build_half(const cpu::Frame& frame, int n_active = -1) const;

    /// @brief Produce full-neighbor list (CSR) into a fresh `PairList`. See
    /// `build_half` for the capacity-preservation rationale.
    /// @param frame Frame for distance evaluation.
    /// @param n_active Outer-loop / array-size bound; `-1` = `frame.nlocal()`.
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
    /// @param n_local Owner count (only rows in `[0, n_local)` are this
    ///        rank's owned atoms; the rest are halo ghosts). When
    ///        `>= 0` (multi-rank), pairs whose `j` is a ghost are
    ///        deduplicated across ranks by atom tag — only the rank
    ///        whose owner's tag is smaller adds the pair, exactly
    ///        LAMMPS `npair_half_bin_newton::build` (newton on,
    ///        `tag[j] < tag[i]` skip). Without this, the same
    ///        owner-ghost LJ pair is computed on BOTH ranks (each
    ///        sees its own atom as the local-index-smaller side) and
    ///        E_pair / virial are 1.3-2× LAMMPS, driving NPT into a
    ///        runaway barostat. Default `-1` (= single-rank mode):
    ///        pure local-index half-list, byte-identical to legacy.
    void refresh_half(const cpu::Frame& frame, int n_active, PairList& out,
                      int n_local = -1) const;

    /// @brief Rebuild the full-neighbor list **in place** into an existing
    /// `PairList`. See `refresh_half` for the capacity-preservation
    /// rationale.
    /// @param frame Frame for distance evaluation.
    /// @param n_active Number of i-atoms this rebuild should cover.
    /// @param out PairList whose buffers are overwritten in place.
    void refresh_full(const cpu::Frame& frame, int n_active,
                      PairList& out) const;

    /// @brief Number of active atoms at the last `build()` / `refresh_*`
    /// call, or `-1` before the first build. `NeighborList::launch` uses
    /// this to detect whether an upstream `Migrate` has changed the
    /// owner count and force a rebuild even when the displacement skin
    /// check would have skipped it.
    int n_at_last_build() const { return n_at_last_build_; }

    /// @brief Verlet skin width [Angstrom] currently held by this strategy.
    /// Used by infrastructure auto-injection (e.g. MD::install_defaults
    /// extending the halo cutoff by skin so atoms in the skin band stay
    /// ghosts across full-rebuild boundaries — without this, ghosts
    /// flicker in/out of the cutoff as atoms drift, injecting per-rebuild
    /// energy jumps into the NPT extended Hamiltonian).
    double skin() const { return skin_; }

    /// @brief Resident heap bytes owned by the cell index (per-atom bin
    /// bookkeeping, the cell count/offset arrays, displacement-save arrays,
    /// stencils, and the multi-thread scratch). Consumed by the built-in
    /// driver memory summary (the "Neighbor" category, with the CSR PairList).
    std::size_t bytes() const;

  private:
    double cutoff_, skin_, cutoff_skin_;

    // Grid
    int ncx_ = 0, ncy_ = 0, ncz_ = 0;    ///< Cells per axis.
    double csx_ = 0, csy_ = 0, csz_ = 0; ///< Cell dimensions.
    int total_cells_ = 0;
    /// @brief Grid origin per axis [Angstrom]. `0.0` for single-rank axes;
    /// `-cutoff_skin` for axes extended to cover ghost halo.
    double origin_[3] = {0.0, 0.0, 0.0};
    /// @brief Per-axis flag: `true` if grid is extended for multi-rank ghosts
    /// (skip PBC wrap during binning); `false` for single-rank (wrap into `[0, L)`).
    bool axis_extended_[3] = {false, false, false};

    /// @brief Minimum-image `d{x,y,z}` on the NON-extended axes only.
    ///
    /// Halo-extended axes carry ghosts pre-placed at their true periodic
    /// image; wrapping them folds a dims==2 overlap band's two images of
    /// the same atom onto one separation and double-counts the pair. Raw
    /// separations are exact there (owners span <= L/2 per rank).
    void min_image_unextended_(const cpu::Box& box, double& dx, double& dy,
                               double& dz) const noexcept {
        for (int d = 0; d < 3; ++d) {
            if (axis_extended_[d])
                continue;
            const double L = box.length(d);
            const double half = 0.5 * L;
            double& v = d == 0 ? dx : (d == 1 ? dy : dz);
            while (v > half)
                v -= L;
            while (v < -half)
                v += L;
        }
    }
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

    // Binning
    xt::xtensor<int, 1> atom_cell_;    ///< Atom -> cell index [capacity].
    xt::xtensor<int, 1> cell_count_;   ///< Atoms per cell [total_cells].
    xt::xtensor<int, 1> cell_offset_;  ///< Prefix sum [total_cells].
    xt::xtensor<int, 1> binned_atoms_; ///< Sorted atom indices [capacity].

    // Stencil. Stored as (di, dj, dk) triples so that opposite-edge
    // wraps under PBC can be applied per-axis at lookup time. The flat
    // index is recomposed in `build_half` as `cx*ncy*ncz + cy*ncz + cz`,
    // wrapping each axis into `[0, ncN)` when the axis is *not* extended
    // (i.e. single-rank or single-rank-along-this-axis under MPI).
    /// @brief Stencil offset triple. Used by `build_half`/`build_full`
    /// together with each cell's (cx,cy,cz) decomposition + PBC wrap.
    struct StencilOffset {
        int di;
        int dj;
        int dk;
    };
    std::vector<StencilOffset> half_stencil_;
    std::vector<StencilOffset> full_stencil_;

    // Displacement tracking
    xt::xtensor<double, 1> x_save_, y_save_, z_save_;

    /// @brief Number of active atoms at last build. `needs_rebuild` forces
    /// a rebuild when `n_active` changes (e.g. after Migrate), since
    /// `binned_atoms_` would otherwise reference indices beyond the new
    /// `n_active` and produce out-of-bounds reads in `build_half`.
    int n_at_last_build_ = -1;

    /// @brief All three box edge lengths at last build. `needs_rebuild`
    /// triggers a rebuild when ANY edge changes (e.g. an NPT barostat or an
    /// anisotropic MC volume move scaled it), because the cell grid
    /// (`csx_/csy_/csz_`) was sized for the old box and would otherwise
    /// silently drop pair candidates after the box grows.
    double box_at_last_build_[3] = {0.0, 0.0, 0.0};

    /// @brief Per-rebuild scratch state — promoted to member variables
    /// to honour the "no allocation in `launch()`" component contract
    /// (CLAUDE.md, Architecture / Component lifecycle).
    ///
    /// `refresh_half` / `refresh_full` were re-constructing
    /// `std::vector<std::vector<std::pair<int,int>>> thread_pairs(n_threads)`
    /// and per-thread `visited` vectors on every rebuild, producing
    /// O(n_threads) heap allocations per neighbor-list refresh. On the
    /// 288-thread aarch64 GH200 node a single rebuild leaked ~345
    /// allocations into the MD hot path. Promoting both buffers to
    /// members lets each rebuild clear the inner vectors (keeping
    /// capacity) and reuse them. Both fields are `mutable` because the
    /// rebuild functions are `const` views of the cell-list state.
    mutable std::vector<std::vector<std::pair<int, int>>> thread_pairs_;
    mutable std::vector<std::vector<int>> thread_visited_;

    /// @brief Per-rebuild CSR write cursor for the multi-thread scatter
    /// finalizer. Reused across `refresh_half`/`refresh_full` rebuilds so the
    /// scatter no longer does `auto write_ptr = result.firstneigh` (a full
    /// std::vector copy = one heap allocation per neighbor-list rebuild in the
    /// MD hot loop). `mutable` because the rebuild functions are `const`.
    mutable std::vector<int> csr_write_cursor_;

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
    /// @brief True when the grid extents no longer match the live box /
    /// subdomain bounds (NPT rescale or RCB rebalance since last build).
    bool grid_stale(const cpu::Frame& frame) const;
    void bin_atoms(const cpu::Frame& frame, int n_active);
    void build_stencil();
    void save_positions(const cpu::Frame& frame, int n_active);
};

} // namespace md
} // namespace cpu
} // namespace ATV
