/// @file cpu/neighbor/cell_list.cpp
/// @brief CellList implementation -- LAMMPS-style linked-cell neighbor list.

#include "atomiverse/cpu/neighbor/cell_list.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/domain.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

// ── CellList ─────────────────────────────────────────────────────

CellList::CellList(double cutoff, double skin)
    : cutoff_(cutoff), skin_(skin < 0 ? double(0.3) * cutoff : skin),
      cutoff_skin_(cutoff + (skin < 0 ? double(0.3) * cutoff : skin)) {}

void CellList::init(const cpu::Frame& frame, double cutoff_skin) {
    cutoff_skin_ = cutoff_skin;
    // Single-rank: no axis extension; whole-box grid; ghosts wrap into [0, L).
    dd_ = nullptr;
    for (int d = 0; d < 3; ++d) {
        axis_extended_[d] = false;
        origin_[d] = 0.0;
    }
    capacity_ = frame.nlocal();
    compute_grid(frame);
    build_stencil();

    atom_cell_ = xt::zeros<int>({capacity_});
    binned_atoms_ = xt::zeros<int>({capacity_});
    x_save_ = xt::zeros<double>({capacity_});
    y_save_ = xt::zeros<double>({capacity_});
    z_save_ = xt::zeros<double>({capacity_});
}

void CellList::init_extended(const cpu::Frame& frame, double cutoff_skin,
                             const mpi::DomainDecomposition* dd, int capacity) {
    cutoff_skin_ = cutoff_skin;
    // Multi-rank: every decomposed axis bins only this rank's subdomain
    // plus a max(cutoff_skin, dd->ghost_shell_width) halo margin (extent +
    // origin derived from dd->sublo/subhi inside compute_grid).
    dd_ = dd;
    for (int d = 0; d < 3; ++d) {
        axis_extended_[d] = (dd != nullptr && dd->dims[d] > 1);
        origin_[d] = 0.0; // set per-axis by compute_grid()
    }
    capacity_ = capacity;
    compute_grid(frame);
    build_stencil();

    atom_cell_ = xt::zeros<int>({capacity_});
    binned_atoms_ = xt::zeros<int>({capacity_});
    x_save_ = xt::zeros<double>({capacity_});
    y_save_ = xt::zeros<double>({capacity_});
    z_save_ = xt::zeros<double>({capacity_});
}

double CellList::halo_pad_() const noexcept {
    // The two reaches answer different questions: cutoff_skin_ is how far a
    // PAIR may be found, dd_->ghost_shell_width is how far a GHOST may
    // actually be placed (HaloSend's max(pair cutoff, bonded extension),
    // covered in as many hops as the subdomain width needs). Under a
    // multi-hop halo, or a bonded extension wider than cutoff + skin, the
    // second exceeds the first and rows land outside a cutoff_skin-only
    // window — where bin_atoms's clamp folds them into the edge cell
    // silently. Take the deeper one. Zero (no halo built yet, or
    // single-rank) leaves today's cutoff_skin margin untouched.
    return dd_ != nullptr ? std::max(cutoff_skin_, dd_->ghost_shell_width)
                          : cutoff_skin_;
}

void CellList::compute_grid(const cpu::Frame& frame) {
    // Per-axis grid extent. Extended (decomposed) axes cover only this
    // rank's subdomain plus a `halo_pad_()` margin on each side, so
    // both the cell arrays and every per-rebuild cell sweep stay
    // O(local atoms) instead of O(global box). Non-extended axes use the
    // global box edge as-is (positions wrap into [0, L) during binning).
    //
    // CRITICAL: the local window is a slice of the GLOBAL bin lattice —
    // the cell size comes from the global box edge and the origin snaps
    // to an integer multiple of it (LAMMPS Neighbor::setup_bins does the
    // same). The owner-ghost Newton dedup in refresh_half classifies a
    // pair as home-cell (tag tie-break on both ranks) or forward-cell
    // (exactly one rank scans it) from CELL INDices; that classification
    // is only rank-symmetric when cell boundaries coincide across ranks.
    // A free-floating origin (sublo - cutoff_skin, not lattice-aligned)
    // lets the same boundary pair land home-cell on one rank and
    // forward-cell on the other → double-counted or dropped pair forces
    // (measured: +2.5% step-1 pressure error on the 8-rank NPT gate).
    // PBC ghost pre-shifts (±L) preserve lattice alignment because L is
    // an exact integer multiple of the cell size by construction.
    //
    // The margin widens with the halo's ghost shell but the CELL EDGE does
    // not: it stays `L / floor(L / cutoff_skin)`, derived from the GLOBAL
    // box, so a deeper shell buys more cells rather than bigger ones and
    // the cross-rank lattice argued above survives untouched.
    int nc[3];
    double cs[3];
    const double pad = halo_pad_();
    for (int d = 0; d < 3; ++d) {
        const double L = frame.box.length(d);
        const int nc_global =
            std::max(int(1), static_cast<int>(std::floor(L / cutoff_skin_)));
        const double cs_d = L / nc_global;
        if (axis_extended_[d]) {
            // Lattice-aligned local window covering [sublo - pad, subhi + pad].
            const int lo_bin =
                static_cast<int>(std::floor((dd_->sublo[d] - pad) / cs_d));
            const int hi_bin =
                static_cast<int>(std::floor((dd_->subhi[d] + pad) / cs_d));
            nc[d] = hi_bin - lo_bin + 1;
            cs[d] = cs_d;
            origin_[d] = lo_bin * cs_d;
            sublo_at_grid_[d] = dd_->sublo[d];
            subhi_at_grid_[d] = dd_->subhi[d];
        } else {
            nc[d] = nc_global;
            cs[d] = cs_d;
            origin_[d] = 0.0;
            sublo_at_grid_[d] = 0.0;
            subhi_at_grid_[d] = L;
        }
    }

    halo_pad_at_grid_ = pad;

    ncx_ = nc[0];
    ncy_ = nc[1];
    ncz_ = nc[2];
    csx_ = cs[0];
    csy_ = cs[1];
    csz_ = cs[2];

    total_cells_ = ncx_ * ncy_ * ncz_;
    cell_count_ = xt::zeros<int>({total_cells_});
    cell_offset_ = xt::zeros<int>({total_cells_});
}

bool CellList::grid_stale(const cpu::Frame& frame) const {
    for (int d = 0; d < 3; ++d)
        if (frame.box.length(d) != box_at_last_build_[d])
            return true;
    if (dd_ != nullptr) {
        for (int d = 0; d < 3; ++d) {
            if (!axis_extended_[d])
                continue;
            if (dd_->sublo[d] != sublo_at_grid_[d]
                || dd_->subhi[d] != subhi_at_grid_[d])
                return true;
            // A halo that deepened its ghost shell since the grid was
            // derived (box shrank into more hops, bonded extension grew)
            // now places ghosts past the window; re-derive rather than let
            // bin_atoms clamp them into the edge cell. Only the extended
            // axes carry the margin, so an unextended-only grid ignores it.
            if (halo_pad_() != halo_pad_at_grid_)
                return true;
        }
    }
    return false;
}

void CellList::build_stencil() {
    half_stencil_.clear();
    full_stencil_.clear();

    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            for (int dk = -1; dk <= 1; ++dk) {
                full_stencil_.push_back({di, dj, dk});
                // Half stencil: include self (0,0,0), and any neighbor
                // whose flat offset > 0 in the original integer ordering
                // (lexicographic on (di, dj, dk)). The full-list filter
                // catches every pair twice; the half-list filter is the
                // same pair pruned to one direction.
                const int flat = di * ncy_ * ncz_ + dj * ncz_ + dk;
                if (flat > 0 || (di == 0 && dj == 0 && dk == 0))
                    half_stencil_.push_back({di, dj, dk});
            }
        }
    }
}

void CellList::bin_atoms(const cpu::Frame& frame, int n_active) {
    int n = (n_active < 0) ? frame.nlocal() : n_active;
    // Grow the per-atom arrays if the active count (n_local + n_ghost) exceeds
    // the capacity captured at init. Previously every rank held nlocal ==
    // global N so the init-time capacity always dominated; with right-sized
    // per-rank storage the ghost count can push `n` past it (and the frame
    // capacity can grow on a thin-subdomain halo overflow), so size on demand.
    if (n > capacity_) {
        capacity_ = n;
        atom_cell_ = xt::zeros<int>({capacity_});
        binned_atoms_ = xt::zeros<int>({capacity_});
        x_save_ = xt::zeros<double>({capacity_});
        y_save_ = xt::zeros<double>({capacity_});
        z_save_ = xt::zeros<double>({capacity_});
    }
    double lx = frame.box.length(0);
    double ly = frame.box.length(1);
    double lz = frame.box.length(2);

    // Clear cell counts
    std::fill(cell_count_.begin(), cell_count_.end(), int(0));

    // Hoisted once at the launch boundary — never resolve a span inside the
    // `#pragma omp parallel` region below (data race + the 13.7x rule).
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);

// Assign atoms to cells. Single-rank axes wrap into [0, L); extended
// axes leave the position alone so PBC-shifted ghost coordinates land
// in the dedicated halo cells around the slab.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double xi = sp_x[i];
        double yi = sp_y[i];
        double zi = sp_z[i];
        if (!axis_extended_[0]) {
            xi -= lx * std::floor(xi / lx);
        }
        if (!axis_extended_[1]) {
            yi -= ly * std::floor(yi / ly);
        }
        if (!axis_extended_[2]) {
            zi -= lz * std::floor(zi / lz);
        }

        int cx = static_cast<int>(std::floor((xi - origin_[0]) / csx_));
        int cy = static_cast<int>(std::floor((yi - origin_[1]) / csy_));
        int cz = static_cast<int>(std::floor((zi - origin_[2]) / csz_));
        cx = std::max(0, std::min(cx, ncx_ - 1));
        cy = std::max(0, std::min(cy, ncy_ - 1));
        cz = std::max(0, std::min(cz, ncz_ - 1));

        atom_cell_[i] = cx * ncy_ * ncz_ + cy * ncz_ + cz;
    }

    // Count atoms per cell (serial -- fast, O(n_cells) is small)
    for (int i = 0; i < n; ++i)
        cell_count_[atom_cell_[i]]++;

    // Prefix sum
    cell_offset_[0] = 0;
    for (int c = 1; c < total_cells_; ++c)
        cell_offset_[c] = cell_offset_[c - 1] + cell_count_[c - 1];

    // Scatter atoms into bins. `bin_write_ptr_` is member state sized
    // once in `compute_grid()` to match `cell_offset_`; copying through
    // it avoids the per-call `auto write_ptr = cell_offset_;` xtensor
    // allocation that otherwise leaks ~1 heap allocation into the MD
    // hot loop per neighbor-list rebuild (violating CLAUDE.md's
    // "no allocation in launch()" contract).
    if (bin_write_ptr_.size() < static_cast<std::size_t>(total_cells_))
        bin_write_ptr_.resize(static_cast<std::size_t>(total_cells_));
    for (int c = 0; c < total_cells_; ++c)
        bin_write_ptr_[c] = cell_offset_[c];
    for (int i = 0; i < n; ++i) {
        int c = atom_cell_[i];
        binned_atoms_[bin_write_ptr_[c]] = i;
        bin_write_ptr_[c]++;
    }
}

void CellList::save_positions(const cpu::Frame& frame, int n_active) {
    int n = (n_active < 0) ? frame.nlocal() : n_active;
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);
    for (int i = 0; i < n; ++i) {
        x_save_[i] = sp_x[i];
        y_save_[i] = sp_y[i];
        z_save_[i] = sp_z[i];
    }
}

void CellList::build(const cpu::Frame& frame, int n_active) {
    // The NPT barostat (or Deform) may have rescaled the box since the
    // cell grid was last sized. `compute_grid()` derives the per-axis
    // cell counts (`ncx_/ncy_/ncz_`) and cell sizes (`csx_/csy_/csz_`)
    // from the current box edge; the stencil depends on the cell
    // counts. If we re-bin into a grid sized for the *old* box, atoms
    // past the stale grid extent get their cell index clamped into the
    // boundary row (`bin_atoms` clamps to `[0, ncx_-1]`), so real
    // neighbour pairs land in cells the stale stencil never visits and
    // are silently dropped — the pair virial then decays toward zero
    // and an NPT run inflates without bound (see
    // tests/cpu/md/test_lammps_npt_parity.cpp). Resize the grid
    // whenever the box edge has moved since the last build — and, for the
    // subdomain-local multi-rank grid, whenever this rank's sublo/subhi
    // moved (RCB rebalance, or NPT rescale propagated through
    // compute_sublo_subhi).
    if (grid_stale(frame)) {
        compute_grid(frame);
        build_stencil();
    }
    bin_atoms(frame, n_active);
    save_positions(frame, n_active);
    n_at_last_build_ = (n_active < 0) ? frame.nlocal() : n_active;
    for (int d = 0; d < 3; ++d)
        box_at_last_build_[d] = frame.box.length(d);
}

bool CellList::needs_rebuild(const cpu::Frame& frame, int n_active) const {
    int n = (n_active < 0) ? frame.nlocal() : n_active;

    // Multi-rank: Migrate / HaloExchange may change n_local + n_ghost
    // between launches.  When that happens, `binned_atoms_` from the
    // previous build still references the OLD atom indices, including ones
    // beyond the current `n_active`. Reading those produces an out-of-
    // bounds access in `build_half`.  Force a rebuild whenever the active
    // count moves.
    if (n != n_at_last_build_)
        return true;

    // NPT: barostat may have scaled the box (or RCB moved this rank's
    // subdomain bounds), invalidating the cell grid. Force a rebuild.
    if (grid_stale(frame))
        return true;

    double max_disp_sq = 0;

    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);

    for (int i = 0; i < n; ++i) {
        double dx = sp_x[i] - x_save_[i];
        double dy = sp_y[i] - y_save_[i];
        double dz = sp_z[i] - z_save_[i];
        // Use minimum-image convention so atom wrapping across PBC
        // boundaries doesn't falsely signal a ~box-length displacement.
        frame.box.min_image(dx, dy, dz);
        double dsq = dx * dx + dy * dy + dz * dz;
        if (dsq > max_disp_sq)
            max_disp_sq = dsq;
    }

    // Rebuild when 2 * max_disp > skin
    return max_disp_sq * double(4) > skin_ * skin_;
}

void CellList::update(const cpu::Frame& frame, int n_active) {
    if (needs_rebuild(frame, n_active))
        build(frame, n_active);
}

namespace {

/// @brief Grow-never-shrink resize of a std::vector<int>. When
/// `n <= current size` this is a no-op; when `n > current size` it
/// hands off to `std::vector::resize`, which amortises allocation and
/// preserves existing contents. Used by `refresh_*` so the per-atom
/// CSR arrays keep a monotonically-non-decreasing capacity across
/// steady-state launches.
inline void ensure_at_least(std::vector<int>& arr, std::size_t n) {
    if (arr.size() < n)
        arr.resize(n);
}

} // namespace

void CellList::refresh_half(const cpu::Frame& frame, int n_active,
                            PairList& result, int n_local) const {
    const int n = (n_active < 0) ? static_cast<int>(frame.nlocal()) : n_active;
    // Multi-rank ghost-pair tie-breaker. `n_local < 0` → single-rank mode,
    // pure local-index filter. `n_local >= 0` → owner-ghost pairs are
    // deduplicated by tag, mirroring LAMMPS `newton on` convention.
    const bool tag_dedup = (n_local >= 0);
    const TagInt* __restrict tag_ptr =
        tag_dedup ? static_cast<const TagInt*>(frame.span(kAtom_tag)) : nullptr;
    const double rc_sq = cutoff_skin_ * cutoff_skin_;

    // Hoisted ONCE at the launch boundary, dominating both the serial and
    // multi-thread branches below (the multi-thread branch opens a
    // `#pragma omp parallel` region — span() may never be resolved inside
    // one, per the 13.7x rule).
    const StorageReal* __restrict x_ptr = frame.span(kAtom_x);
    const StorageReal* __restrict y_ptr = frame.span(kAtom_y);
    const StorageReal* __restrict z_ptr = frame.span(kAtom_z);

    ensure_at_least(result.ilist, static_cast<std::size_t>(n));
    ensure_at_least(result.numneigh, static_cast<std::size_t>(n));
    ensure_at_least(result.firstneigh, static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i)
        result.ilist[i] = i;
    result.inum = n;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // ── Serial fast path ────────────────────────────────────────────
    // Single-thread builds write the CSR directly in one pass: for each
    // atom i we reserve its slot in `neighbors[]`, iterate the stencil,
    // append each surviving j to a running write cursor, then commit
    // `numneigh[i]`. No per-thread pair buffers, no pair-count pass, no
    // prefix-sum pass, no scatter pass — the old path made 3 trips over
    // ~250k pairs plus an `std::vector<std::vector<pair<int,int>>>`
    // allocation every call. `neighbors` grows amortized O(1) the first
    // few times and then stays at capacity (~250k ints = 1 MB for 10k
    // atoms). `firstneigh[i]` is the live cursor at function entry, so
    // consumers that later index `neighbors[firstneigh[i] + k]` see the
    // exact slot we wrote. Matches LAMMPS `NPairHalfBin::build`.
    // Serial fast-path gate: when there are fewer atoms than threads,
    // the multi-thread path's per-thread `my_pairs` buffer would either
    // sit idle on most threads or grow from a tiny baseline on the few
    // that get any work — both produce sporadic heap allocations that
    // would otherwise be amortised out by capacity saturation. Falling
    // back to serial on these small workloads keeps the hot loop
    // allocation-free (CLAUDE.md "no allocation in launch()") and
    // avoids paying OMP fork/join overhead that was already net-negative
    // at this scale.
    if (n_threads == 1 || n < n_threads) {
        // Pre-reserve neighbors with a 32-per-atom heuristic so the
        // first rebuild saturates capacity above any realistic per-atom
        // half-list count (typical LJ + bonded chains: < 16/atom). After
        // saturation, every subsequent rebuild's `push_back` is a
        // pointer-bump store, honouring the "no allocation in launch()"
        // contract.
        const std::size_t want = static_cast<std::size_t>(n) * 32u;
        if (result.neighbors.capacity() < want)
            result.neighbors.reserve(want);
        result.neighbors.clear();
        // Consumers recompute the pair vector by minimum image every step
        // (Box::min_image, O(1)). Storing the rebuild-time per-pair shift and
        // reusing it across steps is UNSAFE: once an atom crosses a periodic
        // boundary between neighbor rebuilds, its stored image offset is stale,
        // and the LJ kernel then evaluates the wrong separation — single-rank
        // NVE energy conservation drifts (~0.7 rel) at OMP_NUM_THREADS=1 (this
        // serial path) while the multi-thread path, which never stored shifts,
        // is fine. Match the multi-thread contract: no stored shifts. Ghosts
        // are pre-placed at their nearest image by the halo, so min_image is a
        // no-op on owner-ghost pairs (multi-rank stays correct).
        result.has_shifts = false;

        // x_ptr/y_ptr/z_ptr hoisted at function entry (shared with the
        // multi-thread branch below).
        const double lx = frame.box.length(0);
        const double ly = frame.box.length(1);
        const double lz = frame.box.length(2);
        const double half_lx = 0.5 * lx;
        const double half_ly = 0.5 * ly;
        const double half_lz = 0.5 * lz;
        const int* __restrict atom_cell_ptr = atom_cell_.data();
        const int* __restrict cell_count_ptr = cell_count_.data();
        const int* __restrict cell_offset_ptr = cell_offset_.data();
        const int* __restrict binned_ptr = binned_atoms_.data();

        // Half-stencil (LAMMPS newton-on) fast path: iterate ~14 cells (home +
        // forward) instead of the full 27 with a `j<=i` reject on every
        // candidate. Each unordered pair is still emitted exactly once (home
        // cell: j>i; forward cells: all j -- the reverse offset is backward
        // from j and thus not in j's half stencil). Valid whenever no PBC
        // wrap-aliasing can fold two stencil offsets onto the same cell, i.e.
        // >=3 cells on every axis. This holds for single-rank (every axis
        // wraps) AND multi-rank: extended (decomposed) axes never wrap
        // (out-of-range stencil cells `continue`, see below), and the
        // non-extended axes still need >=3 cells.
        //
        // Multi-rank cross-rank dedup (tag_dedup): ghost atoms are iterated as
        // `i` too, but `inum` is trimmed to `n_local` downstream so only the
        // owner's scan is consumed. An owner-ghost pair is therefore picked up
        // by exactly the rank whose owner sees the partner in a *forward* cell
        // (the partner rank sees it as a *backward* cell and never scans it);
        // only when both sit in the *home* cell do both ranks see it, so the
        // tag tie-break is needed there alone -- gated by `reject_le_i` below.
        // This mirrors LAMMPS `npair_half_bin_newton` (coordinate tie-break in
        // the home bin, unconditional add in the forward stencil bins).
        // Small/aliased grids (<3 cells) fall back to the full stencil, where
        // the tie-break applies on every cell because both ranks scan all 27.
        const bool use_half = ncx_ >= 3 && ncy_ >= 3 && ncz_ >= 3;
        const std::vector<StencilOffset>& stencil =
            use_half ? half_stencil_ : full_stencil_;

        int visited[27];
        int cursor = 0;

        for (int i = 0; i < n; ++i) {
            result.firstneigh[i] = cursor;
            // Multi-rank: ghost rows (i >= n_local) are trimmed to inum =
            // n_local downstream and never consumed by the force kernel, so we
            // skip building them entirely. Every owner-ghost pair is already
            // emitted from the OWNER's scan (forward/home cells) or from the
            // partner rank's owner (backward cells), so this loses no pair.
            // This mirrors LAMMPS, whose half neighbor list iterates local
            // atoms only.
            //
            // The skip is UNCONDITIONAL on use_half (it applies on both the
            // half- and full-stencil paths). That consistency is essential: if
            // it were gated on use_half, a rank whose deforming/compressing
            // slab transiently drops below 3 cells/axis would flip to
            // full-stencil AND resume building ghost rows, while its peers
            // (still >=3 cells) skip them. That asymmetry makes the reverted
            // rank a straggler and every collective (halo, energy Allreduce)
            // blocks on it -- a measured Comm+Reduce blow-up in the 64-rank
            // (4x4x4, thin-slab) deform stage. Skipping ghost rows on every
            // rank regardless of stencil keeps the per-rank work balanced.
            if (tag_dedup && i >= n_local) {
                result.numneigh[i] = 0;
                continue;
            }
            const int ci = atom_cell_ptr[i];
            const int cz0 = ci % ncz_;
            const int cy0 = (ci / ncz_) % ncy_;
            const int cx0 = ci / (ncy_ * ncz_);
            const double xi = x_ptr[i];
            const double yi = y_ptr[i];
            const double zi = z_ptr[i];

            int visited_n = 0;

            for (auto s : stencil) {
                int cx = cx0 + s.di;
                int cy = cy0 + s.dj;
                int cz = cz0 + s.dk;

                // Per-cell PBC shift: every atom in a wrapped neighbour cell
                // shares the same image shift, so the inner pair loop applies
                // it with a single add instead of the 6-branch per-pair
                // min-image. Only consumed on the `use_half` (single-rank,
                // >=3-cell, non-extended) path, where one wrap is always
                // sufficient.
                double shx = 0.0, shy = 0.0, shz = 0.0;
                if (axis_extended_[0]) {
                    if (cx < 0 || cx >= ncx_)
                        continue;
                } else {
                    if (cx < 0) {
                        cx += ncx_;
                        shx = -lx;
                    } else if (cx >= ncx_) {
                        cx -= ncx_;
                        shx = lx;
                    }
                }
                if (axis_extended_[1]) {
                    if (cy < 0 || cy >= ncy_)
                        continue;
                } else {
                    if (cy < 0) {
                        cy += ncy_;
                        shy = -ly;
                    } else if (cy >= ncy_) {
                        cy -= ncy_;
                        shy = ly;
                    }
                }
                if (axis_extended_[2]) {
                    if (cz < 0 || cz >= ncz_)
                        continue;
                } else {
                    if (cz < 0) {
                        cz += ncz_;
                        shz = -lz;
                    } else if (cz >= ncz_) {
                        cz -= ncz_;
                        shz = lz;
                    }
                }

                const int cj = cx * ncy_ * ncz_ + cy * ncz_ + cz;

                // Small-grid PBC-wrap dedup. Stored as a fixed-size
                // int[27] so the compiler keeps it in a register file.
                bool seen = false;
                for (int v = 0; v < visited_n; ++v) {
                    if (visited[v] == cj) {
                        seen = true;
                        break;
                    }
                }
                if (seen)
                    continue;
                visited[visited_n++] = cj;

                const int start = cell_offset_ptr[cj];
                const int end = start + cell_count_ptr[cj];

                // Half-stencil: only the home cell (cj==ci) needs the j>i
                // reject; forward cells contribute every j exactly once.
                // Full-stencil: every cell rejects j<=i.
                const bool reject_le_i = !use_half || (cj == ci);

                for (int idx = start; idx < end; ++idx) {
                    int j = binned_ptr[idx];
                    if (j <= i && reject_le_i)
                        continue;
                    // Multi-rank ghost-pair tie-breaker: when j is a halo
                    // ghost (j >= n_local), only the rank whose owner has
                    // the smaller tag adds this pair. Without this, the
                    // same owner-ghost pair gets added on BOTH ranks (each
                    // sees its own atom as having the smaller local row),
                    // double-counting LJ energy + virial. Matches LAMMPS
                    // `npair_half_bin_newton::build` newton-on path.
                    //
                    // Gated by `reject_le_i`: the tie-break applies only where
                    // both ranks scan the pair symmetrically -- every cell in
                    // the full stencil, but ONLY the home cell in the half
                    // stencil. In a half-stencil forward cell the partner rank
                    // sees the pair as a backward cell and never scans it, so
                    // applying the tie-break there would drop the pair on both
                    // ranks. (`reject_le_i == !use_half || cj == ci` is exactly
                    // "this cell is scanned by both ranks".)
                    if (tag_dedup && reject_le_i && j >= n_local
                        && tag_ptr[j] < tag_ptr[i])
                        continue;

                    double dx, dy, dz;
                    if (use_half) {
                        // Per-cell shift already encodes the (single) wrap;
                        // one add, no per-pair branch.
                        dx = xi - (x_ptr[j] + shx);
                        dy = yi - (y_ptr[j] + shy);
                        dz = zi - (z_ptr[j] + shz);
                    } else {
                        dx = xi - x_ptr[j];
                        dy = yi - y_ptr[j];
                        dz = zi - z_ptr[j];
                        // Multi-wrap (see Box::min_image) — but ONLY on
                        // axes the halo did NOT extend. Ghosts sit at
                        // their true periodic image already; min-imaging
                        // an extended axis folds the two images a dims==2
                        // overlap band imports (same tag through both
                        // faces) onto ONE separation and double-counts
                        // the pair (observed: silent 16% LJ energy error,
                        // 2026-07-15). The raw dx rejects the far image
                        // by its true distance. Owner-owner pairs never
                        // span half the global box on a decomposed axis
                        // (sub <= L/2), so skipping the wrap there is
                        // exact. Cross-PBC ghosts on small grids (the
                        // 2x2x2 case this multi-wrap was added for) are
                        // ALSO extended-axis ghosts pre-placed at true
                        // images — covered by the same rule.
                        if (!axis_extended_[0]) {
                            while (dx > half_lx)
                                dx -= lx;
                            while (dx < -half_lx)
                                dx += lx;
                        }
                        if (!axis_extended_[1]) {
                            while (dy > half_ly)
                                dy -= ly;
                            while (dy < -half_ly)
                                dy += ly;
                        }
                        if (!axis_extended_[2]) {
                            while (dz > half_lz)
                                dz -= lz;
                            while (dz < -half_lz)
                                dz += lz;
                        }
                    }

                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 < rc_sq) {
                        // push_back on an empty-then-growing vector
                        // amortizes to O(1); after a few rebuilds
                        // capacity saturates and push_back is a
                        // pointer-bump store.
                        result.neighbors.push_back(j);
                        ++cursor;
                    }
                }
            }
            result.numneigh[i] = cursor - result.firstneigh[i];
        }
        result.total_pairs = cursor;
        return;
    }

    // ── Multi-thread path ───────────────────────────────────────────
    // Does not populate per-pair shifts; consumers fall back to min-image.
    result.has_shifts = false;
    // Thread-private pair buffers feed a pair-count + prefix-sum +
    // scatter finalizer. Slightly more work per pair than the serial
    // fast path but keeps scaling on OMP ≥ 2. The old single-path
    // implementation paid this overhead even with OMP=1 — that cost is
    // what the serial branch above reclaims.
    //
    // `thread_pairs_` / `thread_visited_` are *member* state: grow them
    // to the current OMP team size if needed, clear the inner vectors
    // (capacity preserved), and reuse. This is the "no allocation in
    // launch()" fix — the old code re-constructed n_threads vectors
    // per rebuild, leaking O(n_threads) heap allocations into the MD
    // hot path on every neighbor refresh.
    if (static_cast<int>(thread_pairs_.size()) < n_threads)
        thread_pairs_.resize(static_cast<std::size_t>(n_threads));
    if (static_cast<int>(thread_visited_.size()) < n_threads)
        thread_visited_.resize(static_cast<std::size_t>(n_threads));
    // Reserve each thread's pair buffer to a per-thread share of the
    // 32-per-atom half-list budget (the same heuristic the serial path uses
    // for `neighbors`). Without this the buffers grow lazily as pair counts
    // drift, leaking a heap allocation into the MD hot loop after warm-up
    // (caught by MdNoAlloc). Margin covers schedule(dynamic) imbalance.
    const std::size_t pair_cap_hint =
        static_cast<std::size_t>(n / n_threads + 1) * 32u;
    for (auto& tp : thread_pairs_) {
        tp.clear();
        if (tp.capacity() < pair_cap_hint)
            tp.reserve(pair_cap_hint);
    }
    for (auto& tv : thread_visited_) {
        tv.clear();
        if (tv.capacity() < 27u)
            tv.reserve(27); // 27-cell stencil; lazy one-shot grow per thread.
    }
    for (int i = 0; i < n; ++i)
        result.numneigh[i] = 0;

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& my_pairs = thread_pairs_[tid];
        auto& visited = thread_visited_[tid];

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            const int ci = atom_cell_[i];
            const int cz0 = ci % ncz_;
            const int cy0 = (ci / ncz_) % ncy_;
            const int cx0 = ci / (ncy_ * ncz_);

            visited.clear();

            for (auto s : full_stencil_) {
                int cx = cx0 + s.di;
                int cy = cy0 + s.dj;
                int cz = cz0 + s.dk;

                if (axis_extended_[0]) {
                    if (cx < 0 || cx >= ncx_)
                        continue;
                } else {
                    if (cx < 0)
                        cx += ncx_;
                    if (cx >= ncx_)
                        cx -= ncx_;
                }
                if (axis_extended_[1]) {
                    if (cy < 0 || cy >= ncy_)
                        continue;
                } else {
                    if (cy < 0)
                        cy += ncy_;
                    if (cy >= ncy_)
                        cy -= ncy_;
                }
                if (axis_extended_[2]) {
                    if (cz < 0 || cz >= ncz_)
                        continue;
                } else {
                    if (cz < 0)
                        cz += ncz_;
                    if (cz >= ncz_)
                        cz -= ncz_;
                }

                const int cj = cx * ncy_ * ncz_ + cy * ncz_ + cz;

                bool seen = false;
                for (int v : visited)
                    if (v == cj) {
                        seen = true;
                        break;
                    }
                if (seen)
                    continue;
                visited.push_back(cj);

                const int start = cell_offset_[cj];
                const int end = start + cell_count_[cj];

                for (int idx = start; idx < end; ++idx) {
                    int j = binned_atoms_[idx];
                    if (j <= i)
                        continue;
                    // Multi-rank ghost tie-breaker — see serial path
                    // comment above for the LAMMPS-newton-on rationale.
                    if (tag_dedup && j >= n_local && tag_ptr[j] < tag_ptr[i])
                        continue;

                    double dx = x_ptr[i] - x_ptr[j];
                    double dy = y_ptr[i] - y_ptr[j];
                    double dz = z_ptr[i] - z_ptr[j];
                    // Extended-axis-aware min-image (see the serial
                    // full-stencil comment): never wrap axes the halo
                    // extended, or dual-image ghosts double-count.
                    min_image_unextended_(frame.box, dx, dy, dz);

                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 < rc_sq) {
                        my_pairs.emplace_back(i, j);
                    }
                }
            }
        }
    }

    // Iterate only [0, n_threads) — `thread_pairs_` may be larger than
    // the current team if a prior rebuild ran on a wider team; those
    // extra slots are already cleared above but still skip them.
    for (int t = 0; t < n_threads; ++t)
        for (auto& [i, j] : thread_pairs_[t])
            result.numneigh[i]++;

    result.firstneigh[0] = 0;
    for (int i = 1; i < n; ++i)
        result.firstneigh[i] =
            result.firstneigh[i - 1] + result.numneigh[i - 1];

    result.total_pairs =
        (n > 0) ? result.firstneigh[n - 1] + result.numneigh[n - 1] : 0;
    result.neighbors.resize(static_cast<std::size_t>(result.total_pairs));

    // Member-state write cursor (no per-rebuild `auto write_ptr =
    // result.firstneigh` vector copy — that leaked one heap allocation per
    // neighbor-list rebuild into the MD hot loop). Sized monotonically; the
    // contents are reset from firstneigh each call.
    if (csr_write_cursor_.size() < static_cast<std::size_t>(n))
        csr_write_cursor_.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        csr_write_cursor_[i] = result.firstneigh[i];
    for (int t = 0; t < n_threads; ++t) {
        for (auto& [i, j] : thread_pairs_[t]) {
            result.neighbors[csr_write_cursor_[i]] = j;
            csr_write_cursor_[i]++;
        }
    }
}

PairList CellList::build_half(const cpu::Frame& frame, int n_active) const {
    PairList out;
    refresh_half(frame, n_active, out);
    return out;
}

void CellList::refresh_full(const cpu::Frame& frame, int n_active,
                            PairList& result) const {
    const int n = (n_active < 0) ? static_cast<int>(frame.nlocal()) : n_active;
    const double rc_sq = cutoff_skin_ * cutoff_skin_;

    // Hoisted ONCE at the launch boundary, dominating the `#pragma omp
    // parallel` region below (span() may never be resolved inside one).
    const StorageReal* __restrict x_ptr = frame.span(kAtom_x);
    const StorageReal* __restrict y_ptr = frame.span(kAtom_y);
    const StorageReal* __restrict z_ptr = frame.span(kAtom_z);

    ensure_at_least(result.ilist, static_cast<std::size_t>(n));
    ensure_at_least(result.numneigh, static_cast<std::size_t>(n));
    ensure_at_least(result.firstneigh, static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        result.ilist[i] = i;
        result.numneigh[i] = 0;
    }
    result.inum = n;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // Member-state scratch reuse — same contract as the `refresh_half`
    // version. See the long comment above for the "no allocation in
    // launch()" rationale.
    if (static_cast<int>(thread_pairs_.size()) < n_threads)
        thread_pairs_.resize(static_cast<std::size_t>(n_threads));
    if (static_cast<int>(thread_visited_.size()) < n_threads)
        thread_visited_.resize(static_cast<std::size_t>(n_threads));
    // Reserve each thread's pair buffer to a per-thread share of the
    // 32-per-atom half-list budget (the same heuristic the serial path uses
    // for `neighbors`). Without this the buffers grow lazily as pair counts
    // drift, leaking a heap allocation into the MD hot loop after warm-up
    // (caught by MdNoAlloc). Margin covers schedule(dynamic) imbalance.
    const std::size_t pair_cap_hint =
        static_cast<std::size_t>(n / n_threads + 1) * 32u;
    for (auto& tp : thread_pairs_) {
        tp.clear();
        if (tp.capacity() < pair_cap_hint)
            tp.reserve(pair_cap_hint);
    }
    for (auto& tv : thread_visited_) {
        tv.clear();
        if (tv.capacity() < 27u)
            tv.reserve(27);
    }

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& my_pairs = thread_pairs_[tid];
        auto& visited = thread_visited_[tid];

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            const int ci = atom_cell_[i];
            const int cz0 = ci % ncz_;
            const int cy0 = (ci / ncz_) % ncy_;
            const int cx0 = ci / (ncy_ * ncz_);

            visited.clear();

            for (auto s : full_stencil_) {
                int cx = cx0 + s.di;
                int cy = cy0 + s.dj;
                int cz = cz0 + s.dk;

                if (axis_extended_[0]) {
                    if (cx < 0 || cx >= ncx_)
                        continue;
                } else {
                    if (cx < 0)
                        cx += ncx_;
                    if (cx >= ncx_)
                        cx -= ncx_;
                }
                if (axis_extended_[1]) {
                    if (cy < 0 || cy >= ncy_)
                        continue;
                } else {
                    if (cy < 0)
                        cy += ncy_;
                    if (cy >= ncy_)
                        cy -= ncy_;
                }
                if (axis_extended_[2]) {
                    if (cz < 0 || cz >= ncz_)
                        continue;
                } else {
                    if (cz < 0)
                        cz += ncz_;
                    if (cz >= ncz_)
                        cz -= ncz_;
                }

                const int cj = cx * ncy_ * ncz_ + cy * ncz_ + cz;

                bool seen = false;
                for (int v : visited)
                    if (v == cj) {
                        seen = true;
                        break;
                    }
                if (seen)
                    continue;
                visited.push_back(cj);

                const int start = cell_offset_[cj];
                const int end = start + cell_count_[cj];

                for (int idx = start; idx < end; ++idx) {
                    int j = binned_atoms_[idx];
                    if (j == i)
                        continue;

                    double dx = x_ptr[i] - x_ptr[j];
                    double dy = y_ptr[i] - y_ptr[j];
                    double dz = z_ptr[i] - z_ptr[j];
                    // Extended-axis-aware min-image (see the serial
                    // full-stencil comment): never wrap axes the halo
                    // extended, or dual-image ghosts double-count.
                    min_image_unextended_(frame.box, dx, dy, dz);

                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 < rc_sq) {
                        my_pairs.emplace_back(i, j);
                    }
                }
            }
        }
    }

    for (int t = 0; t < n_threads; ++t)
        for (auto& [i, j] : thread_pairs_[t])
            result.numneigh[i]++;

    result.firstneigh[0] = 0;
    for (int i = 1; i < n; ++i)
        result.firstneigh[i] =
            result.firstneigh[i - 1] + result.numneigh[i - 1];

    result.total_pairs =
        (n > 0) ? result.firstneigh[n - 1] + result.numneigh[n - 1] : 0;
    result.neighbors.resize(static_cast<std::size_t>(result.total_pairs));

    // Member-state write cursor (no per-rebuild `auto write_ptr =
    // result.firstneigh` vector copy — that leaked one heap allocation per
    // neighbor-list rebuild into the MD hot loop). Sized monotonically; the
    // contents are reset from firstneigh each call.
    if (csr_write_cursor_.size() < static_cast<std::size_t>(n))
        csr_write_cursor_.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        csr_write_cursor_[i] = result.firstneigh[i];
    for (int t = 0; t < n_threads; ++t) {
        for (auto& [i, j] : thread_pairs_[t]) {
            result.neighbors[csr_write_cursor_[i]] = j;
            csr_write_cursor_[i]++;
        }
    }
}

PairList CellList::build_full(const cpu::Frame& frame, int n_active) const {
    PairList out;
    refresh_full(frame, n_active, out);
    return out;
}

std::size_t CellList::bytes() const {
    auto xt_bytes = [](const auto& a) {
        return a.size()
               * sizeof(typename std::decay_t<decltype(a)>::value_type);
    };
    std::size_t n = xt_bytes(atom_cell_) + xt_bytes(binned_atoms_)
                    + xt_bytes(cell_count_) + xt_bytes(cell_offset_)
                    + xt_bytes(x_save_) + xt_bytes(y_save_) + xt_bytes(z_save_);
    // `size()` (live elements) ≈ resident pages; reserved slack is lazily
    // mapped and not resident, so this subtracts cleanly against peak RSS.
    n += (half_stencil_.size() + full_stencil_.size()) * sizeof(StencilOffset);
    n += csr_write_cursor_.size() * sizeof(int);
    n += bin_write_ptr_.size() * sizeof(int);
    // Multi-thread scratch: per-thread (i,j) pair buffers + visited-cell lists.
    for (const auto& tp : thread_pairs_)
        n += tp.size() * sizeof(std::pair<int, int>);
    for (const auto& tv : thread_visited_)
        n += tv.size() * sizeof(int);
    return n;
}

} // namespace md
} // namespace cpu
} // namespace ATV
