/// @file cpu/neighbor/cell_list.cpp
/// @brief CellList implementation -- LAMMPS-style linked-cell neighbor list.
///
/// Global-bin / local-window indexing adapted from LAMMPS src/nbin.cpp
/// NBin::coord2bin and src/nbin_standard.cpp. Thread-owned neighbor rows
/// follow src/OPENMP/npair_bin_omp.cpp and src/OPENMP/npair_omp.h;
/// flat stencil offsets, triclinic full stencils and tag ownership follow src/nstencil_bin.cpp and
/// src/npair_bin.cpp; full-to-half assembly follows src/npair_halffull.cpp
/// (a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b).
/// Copyright (2003) Sandia Corporation. Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software. Distributed under the GNU General
/// Public License; this adaptation is GPL-2.0-only.

#include "atomiverse/cpu/neighbor/cell_list.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/cpu/communication/domain.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

// ── CellList ─────────────────────────────────────────────────────

CellList::CellList(double cutoff, double skin)
    : cutoff_(cutoff), skin_(skin < 0 ? units::UnitSystem::lj().skin : skin),
      cutoff_skin_(cutoff + skin_) {}

void CellList::set_skin(double skin) {
    skin_ = skin;
    cutoff_skin_ = cutoff_ + skin;
}

void CellList::init(const cpu::Frame& frame, double cutoff_skin) {
    cutoff_skin_ = cutoff_skin;
    // Whole-cell Cartesian window; caller supplies explicit image rows.
    dd_ = nullptr;
    for (int d = 0; d < 3; ++d)
        bin_lo_[d] = 0;
    capacity_ = frame.capacity();
    compute_grid(frame);
    build_stencil();

    atom_cell_ = xt::zeros<int>({capacity_});
    binned_atoms_ = xt::zeros<int>({capacity_});
}

void CellList::init_extended(const cpu::Frame& frame, double cutoff_skin,
                             const mpi::DomainDecomposition* dd, int capacity) {
    cutoff_skin_ = cutoff_skin;
    // Multi-rank: every decomposed axis bins only this rank's subdomain
    // plus a max(cutoff_skin, dd->ghost_shell_width) halo margin (extent +
    // integer bin offset derived from dd->sublo/subhi inside compute_grid).
    dd_ = dd;
    for (int d = 0; d < 3; ++d)
        bin_lo_[d] = 0;
    capacity_ = capacity;
    compute_grid(frame);
    build_stencil();

    atom_cell_ = xt::zeros<int>({capacity_});
    binned_atoms_ = xt::zeros<int>({capacity_});
}

double CellList::halo_pad_() const noexcept {
    // The two reaches answer different questions: cutoff_skin_ is how far a
    // PAIR may be found, dd_->ghost_shell_width is how far a GHOST may
    // actually be placed (CommBrick's max(pair cutoff, bonded extension),
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
    triclinic_ = !frame.box.is_orthorhombic();
    const auto bounds =
        frame.box.cartesian_bounds(Vec3{0, 0, 0}, Vec3{1, 1, 1});
    Vec3 local_lo, local_hi;
    {
        Vec3 lo, hi;
        for (int d = 0; d < 3; ++d) {
            const double reach = halo_pad_() / frame.box.face_height(d);
            lo[d] = (dd_ ? dd_->sublo[d] : 0.0) - reach;
            hi[d] = (dd_ ? dd_->subhi[d] : 1.0) + reach;
        }
        const auto local = frame.box.cartesian_bounds(lo, hi);
        local_lo = local.first;
        local_hi = local.second;
    }
    for (int d = 0; d < 3; ++d) {
        bbox_lo_[d] = bounds.first[d];
        bbox_extent_[d] = bounds.second[d] - bounds.first[d];
    }
    // Per-axis grid extent. Extended (decomposed) axes cover only this
    // rank's subdomain plus a `halo_pad_()` margin on each side, so
    // both the cell arrays and every per-rebuild cell sweep stay
    // O(local atoms) instead of O(global box). Without DD, the
    // whole-cell window uses the same padding and raw-coordinate rule.
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
    // Compute the global bin FIRST and subtract an integer window offset.
    // Subtracting a rank-dependent floating origin before division can round
    // an exact boundary into different bins on the owner and ghost ranks.
    // This follows LAMMPS NBin::coord2bin's global-to-local indexing.
    //
    // The margin widens with the halo's ghost shell but the CELL EDGE does
    // not: it stays `L / floor(L / (0.5 * cutoff_skin))`, derived from the GLOBAL
    // box, so a deeper shell buys more cells rather than bigger ones and
    // the cross-rank lattice argued above survives untouched.
    int nc[3];
    double cs[3];
    const double pad = halo_pad_();
    for (int d = 0; d < 3; ++d) {
        const double L = bbox_extent_[d];
        // LAMMPS `Neighbor::setup_bins`: bins are HALF the neighbor reach
        // (binsize = 0.5 * cutoff+skin), searched with a multi-cell
        // distance-pruned stencil. Half-reach bins cut the candidate
        // volume per atom by ~1.8x against one-reach bins — the stencil
        // covers the reach exactly instead of rounding it up to whole
        // fat cells.
        const int nc_global = std::max(
            int(1), static_cast<int>(std::floor(L / (0.5 * cutoff_skin_))));
        global_bins_[d] = nc_global;
        const double cs_d = L / nc_global;
        const double lo = local_lo[d] - bbox_lo_[d];
        const double hi = local_hi[d] - bbox_lo_[d];
        const int lo_bin = static_cast<int>(std::floor(lo / cs_d)) - 1;
        const int hi_bin = static_cast<int>(std::floor(hi / cs_d)) + 1;
        nc[d] = hi_bin - lo_bin + 1;
        cs[d] = cs_d;
        bin_lo_[d] = lo_bin;
        sublo_at_grid_[d] = dd_ ? dd_->sublo[d] : 0.0;
        subhi_at_grid_[d] = dd_ ? dd_->subhi[d] : 1.0;
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
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            if (frame.box.cell()(row, col) != box_at_last_build_(row, col))
                return true;
    if (dd_ != nullptr) {
        for (int d = 0; d < 3; ++d) {
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

double CellList::lattice_drift(const cpu::Frame& frame) const {
    const Mat3& now = frame.box.cell();
    double drift = 0.0;
    for (int col = 0; col < 3; ++col) {
        double acc = 0.0;
        for (int row = 0; row < 3; ++row) {
            const double d = now(row, col) - box_at_last_build_(row, col);
            acc += d * d;
        }
        drift = std::max(drift, std::sqrt(acc));
    }
    return drift;
}

bool CellList::geometry_forces_rebuild(const cpu::Frame& frame) const {
    // A pair just outside cutoff+skin enters the force cutoff only when a
    // lattice vector moves by more than the skin. Berendsen steps are far
    // smaller than that; the owner displacement check covers the affine
    // part. Bin resize happens in `build()` via `grid_stale()`, not here.
    if (lattice_drift(frame) > skin_)
        return true;
    if (dd_ == nullptr)
        return false;
    if (halo_pad_() != halo_pad_at_grid_)
        return true;
    for (int d = 0; d < 3; ++d) {
        if (dd_->sublo[d] != sublo_at_grid_[d]
            || dd_->subhi[d] != subhi_at_grid_[d])
            return true;
    }
    return false;
}

void CellList::build_stencil() {
    half_stencil_.clear();
    full_stencil_.clear();

    // Per-axis reach in cells. With half-reach bins this is 2; odd aspect
    // ratios or the nc==1 floor can make it 1 or 3.
    const int rx =
        std::max(1, static_cast<int>(std::ceil(cutoff_skin_ / csx_ - 1e-12)));
    const int ry =
        std::max(1, static_cast<int>(std::ceil(cutoff_skin_ / csy_ - 1e-12)));
    const int rz =
        std::max(1, static_cast<int>(std::ceil(cutoff_skin_ / csz_ - 1e-12)));
    stencil_reach_[0] = rx;
    stencil_reach_[1] = ry;
    stencil_reach_[2] = rz;

    const double rc_sq = cutoff_skin_ * cutoff_skin_;
    // Minimum distance between any two points of the home cell and the
    // offset cell (LAMMPS `Neighbor::bin_distance`): cells farther apart
    // than the reach can never hold a pair and are pruned outright.
    auto min_dist_sq = [&](int di, int dj, int dk) {
        const double ddx =
            di > 0 ? (di - 1) * csx_ : (di < 0 ? (di + 1) * csx_ : 0.0);
        const double ddy =
            dj > 0 ? (dj - 1) * csy_ : (dj < 0 ? (dj + 1) * csy_ : 0.0);
        const double ddz =
            dk > 0 ? (dk - 1) * csz_ : (dk < 0 ? (dk + 1) * csz_ : 0.0);
        return ddx * ddx + ddy * ddy + ddz * ddz;
    };

    for (int di = -rx; di <= rx; ++di) {
        for (int dj = -ry; dj <= ry; ++dj) {
            for (int dk = -rz; dk <= rz; ++dk) {
                if (min_dist_sq(di, dj, dk) > rc_sq)
                    continue;
                const int flat = (di * ncy_ + dj) * ncz_ + dk;
                full_stencil_.push_back({di, dj, dk, flat});
                // Half stencil: the home cell plus the lexicographically
                // FORWARD offsets. Explicit lexicographic order (not the
                // flat-index sign, which aliases once |dj|·ncz + |dk|
                // can exceed ncy·ncz on multi-cell reaches): every
                // unordered pair is emitted exactly once — home cell by
                // the j>i filter, forward cells unconditionally, and the
                // backward image of a forward offset is never scanned.
                const bool home = (di == 0 && dj == 0 && dk == 0);
                const bool forward = (di > 0) || (di == 0 && dj > 0)
                                     || (di == 0 && dj == 0 && dk > 0);
                if (home || forward)
                    half_stencil_.push_back({di, dj, dk, flat});
            }
        }
    }
}

void CellList::bin_atoms(const cpu::Frame& frame, int n_active) {
    int n = (n_active < 0) ? frame.n_total() : n_active;
    // Grow the per-atom arrays if the active count (n_local + n_ghost) exceeds
    // the capacity captured at init. Previously every rank held nlocal ==
    // global N so the init-time capacity always dominated; with right-sized
    // per-rank storage the ghost count can push `n` past it (and the frame
    // capacity can grow on a thin-subdomain halo overflow), so size on demand.
    if (n > capacity_) {
        capacity_ = n;
        atom_cell_ = xt::zeros<int>({capacity_});
        binned_atoms_ = xt::zeros<int>({capacity_});
    }
    const double lx = bbox_extent_[0];
    const double ly = bbox_extent_[1];
    const double lz = bbox_extent_[2];
    // Clear cell counts
    std::fill(cell_count_.begin(), cell_count_.end(), int(0));

    // Hoisted once at the launch boundary — never resolve a span inside the
    // `#pragma omp parallel` region below (data race + the 13.7x rule).
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);

    // LAMMPS NBin::coord2bin: distinguish the canonical box from its
    // images. A coordinate strictly below L must stay in the last box bin
    // even when division rounds to nbin; x == L starts the upper image.
    const auto global_bin = [](double x, double length, double size,
                               int count) {
        if (x >= length)
            return static_cast<int>((x - length) / size) + count;
        if (x >= 0)
            return std::min(static_cast<int>(x / size), count - 1);
        return static_cast<int>(x / size) - 1;
    };

// Assign explicit Cartesian owner and image rows to their physical cells.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double xi = sp_x[i];
        double yi = sp_y[i];
        double zi = sp_z[i];

        int cx = global_bin(xi - bbox_lo_[0], lx, csx_, global_bins_[0])
                 - bin_lo_[0];
        int cy = global_bin(yi - bbox_lo_[1], ly, csy_, global_bins_[1])
                 - bin_lo_[1];
        int cz = global_bin(zi - bbox_lo_[2], lz, csz_, global_bins_[2])
                 - bin_lo_[2];
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
    // hot loop per neighbor-list rebuild (violating docs/developer/project-contract.md's
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
    box_at_last_build_ = frame.box.cell();
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
    const int n = (n_active < 0) ? static_cast<int>(frame.n_total()) : n_active;
    // Ghost ownership follows LAMMPS Newton-on, independent of rank count.
    if (n_local < 0)
        n_local = frame.nlocal();
    const TagInt* __restrict tag_ptr = frame.span(kAtom_tag);
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
    result.inum = n_local;
    result.n_rows = n;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    const bool use_half = !triclinic_;
    const int* __restrict atom_cell_ptr = atom_cell_.data();
    const int* __restrict cell_count_ptr = cell_count_.data();
    const int* __restrict cell_offset_ptr = cell_offset_.data();
    const int* __restrict binned_ptr = binned_atoms_.data();
    const auto& stencil = use_half ? half_stencil_ : full_stencil_;
    const int n_owned_rows = std::min(n_local, n);

    // One row traversal for serial CSR and thread-private buffers. The
    // LAMMPS half/Newton stencil selects forward cells; only the home cell
    // requires the symmetric coordinate/index tie-break.
    const auto enumerate_row = [&](int i, const auto& append) {
        const int ci = atom_cell_ptr[i];
        const int cz0 = ci % ncz_;
        const int cy0 = (ci / ncz_) % ncy_;
        const int cx0 = ci / (ncy_ * ncz_);
        const double xi = x_ptr[i];
        const double yi = y_ptr[i];
        const double zi = z_ptr[i];

        // NPairBin: an interior stencil uses ibin + precomputed offset.
        // Boundary rows clip offsets to the allocated Cartesian window.
        const bool interior =
            cx0 >= stencil_reach_[0] && cx0 < ncx_ - stencil_reach_[0]
            && cy0 >= stencil_reach_[1] && cy0 < ncy_ - stencil_reach_[1]
            && cz0 >= stencil_reach_[2] && cz0 < ncz_ - stencil_reach_[2];

        for (auto s : stencil) {
            int cj = ci + s.flat;
            if (!interior) {
                const int cx = cx0 + s.di, cy = cy0 + s.dj, cz = cz0 + s.dk;
                if (cx < 0 || cx >= ncx_ || cy < 0 || cy >= ncy_ || cz < 0
                    || cz >= ncz_)
                    continue;
                cj = (cx * ncy_ + cy) * ncz_ + cz;
            }

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
                if (reject_le_i && j >= n_local) {
                    if (triclinic_) {
                        const TagInt itag = tag_ptr[i], jtag = tag_ptr[j];
                        const bool odd = ((itag ^ jtag) & TagInt{1}) != 0;
                        if ((itag > jtag && !odd) || (itag < jtag && odd))
                            continue;
                        if (itag == jtag) {
                            if (std::abs(z_ptr[j] - zi) > 0.01) {
                                if (z_ptr[j] < zi)
                                    continue;
                            } else if (std::abs(y_ptr[j] - yi) > 0.01) {
                                if (y_ptr[j] < yi)
                                    continue;
                            } else if (x_ptr[j] < xi)
                                continue;
                        }
                    } else {
                        if (x_ptr[j] < xi)
                            continue;
                        if (x_ptr[j] == xi && y_ptr[j] < yi)
                            continue;
                        if (x_ptr[j] == xi && y_ptr[j] == yi && z_ptr[j] < zi)
                            continue;
                    }
                }

                const double dx = xi - x_ptr[j];
                const double dy = yi - y_ptr[j];
                const double dz = zi - z_ptr[j];

                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < rc_sq) {
                    append(j);
                }
            }
        }
    };

    if (n_threads == 1 || n < n_threads) {
        const std::size_t want = static_cast<std::size_t>(n) * 32u;
        if (result.neighbors.capacity() < want)
            result.neighbors.reserve(want);
        result.neighbors.clear();
        int cursor = 0;
        for (int i = 0; i < n; ++i) {
            result.firstneigh[i] = cursor;
            if (i < n_owned_rows)
                enumerate_row(i, [&](int j) {
                    result.neighbors.push_back(j);
                    ++cursor;
                });
            result.numneigh[i] = cursor - result.firstneigh[i];
        }
        result.total_pairs = cursor;
        return;
    }

    // ── Multi-thread path ───────────────────────────────────────────
    // LAMMPS NPairBinOmp writes complete rows into each thread's own page.
    // Retain contiguous j-only arenas here, then bulk-copy them to the public
    // CSR. Prefix only thread totals; no pair-wise scatter or repeated i.
    //
    // `thread_neighbors_` is member state: grow it
    // to the current OMP team size if needed, clear the inner vectors
    // (capacity preserved), and reuse them across rebuilds.
    if (static_cast<int>(thread_neighbors_.size()) < n_threads)
        thread_neighbors_.resize(static_cast<std::size_t>(n_threads));
    ensure_at_least(thread_offsets_, static_cast<std::size_t>(n_threads + 1));
    // Reserve each thread's index arena to a per-thread share of the
    // 32-per-atom half-list budget (the same heuristic the serial path uses
    // for `neighbors`). Without this the buffers grow lazily as pair counts
    // drift, leaking a heap allocation into the MD hot loop after warm-up
    // (caught by MdNoAlloc). Capacities are retained across rebuilds.
    const std::size_t pair_cap_hint =
        static_cast<std::size_t>(n / n_threads + 1) * 32u;
    for (auto& tp : thread_neighbors_) {
        tp.clear();
        if (tp.capacity() < pair_cap_hint)
            tp.reserve(pair_cap_hint);
    }
    for (int i = 0; i < n; ++i)
        result.numneigh[i] = 0;

#pragma omp parallel num_threads(n_threads)
    {
        int tid = 0, team_size = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        team_size = omp_get_num_threads();
#endif
        auto& my_neighbors = thread_neighbors_[tid];

        const int chunk = 1 + n_owned_rows / team_size;
        const int begin = std::min(tid * chunk, n_owned_rows);
        const int end = std::min(begin + chunk, n_owned_rows);
        for (int i = begin; i < end; ++i) {
            result.firstneigh[i] = static_cast<int>(my_neighbors.size());
            enumerate_row(i, [&](int j) { my_neighbors.push_back(j); });
            result.numneigh[i] =
                static_cast<int>(my_neighbors.size()) - result.firstneigh[i];
        }
#pragma omp barrier
#pragma omp single
        {
            thread_offsets_[0] = 0;
            for (int t = 0; t < team_size; ++t)
                thread_offsets_[t + 1] =
                    thread_offsets_[t]
                    + static_cast<int>(thread_neighbors_[t].size());
            result.total_pairs = thread_offsets_[team_size];
            result.neighbors.resize(
                static_cast<std::size_t>(result.total_pairs));
        }
        const int offset = thread_offsets_[tid];
        std::copy(my_neighbors.begin(), my_neighbors.end(),
                  result.neighbors.begin() + offset);
        for (int i = begin; i < end; ++i)
            result.firstneigh[i] += offset;
    }
    // Ghost centres carry no half-list rows, but retain valid empty CSR offsets.
    for (int i = n_owned_rows; i < n; ++i)
        result.firstneigh[i] = result.total_pairs;
}

PairList CellList::build_half(const cpu::Frame& frame, int n_active) const {
    PairList out;
    refresh_half(frame, n_active, out);
    return out;
}

void CellList::refresh_full(const cpu::Frame& frame, int n_active,
                            PairList& result, int n_rows) const {
    const int n = (n_active < 0) ? static_cast<int>(frame.n_total()) : n_active;
    const int centres = n_rows < 0 ? frame.nlocal() : n_rows;
    if (n_rows < -1 || centres > n)
        throw std::invalid_argument("CellList full-list centre count must be "
                                    "-1 or within active atoms");
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
    result.inum = result.n_rows = centres;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // Member-state scratch reuse — same contract as the `refresh_half`
    // version. See the long comment above for the "no allocation in
    // launch()" rationale.
    if (static_cast<int>(thread_neighbors_.size()) < n_threads)
        thread_neighbors_.resize(static_cast<std::size_t>(n_threads));
    ensure_at_least(thread_offsets_, static_cast<std::size_t>(n_threads + 1));
    // Reserve each thread's index arena to a per-thread share of the
    // 32-per-atom half-list budget (the same heuristic the serial path uses
    // for `neighbors`). Without this the buffers grow lazily as pair counts
    // drift, leaking a heap allocation into the MD hot loop after warm-up
    // (caught by MdNoAlloc). Capacities are retained across rebuilds.
    const std::size_t pair_cap_hint =
        static_cast<std::size_t>(n / n_threads + 1) * 32u;
    for (auto& tp : thread_neighbors_) {
        tp.clear();
        if (tp.capacity() < pair_cap_hint)
            tp.reserve(pair_cap_hint);
    }

#pragma omp parallel num_threads(n_threads)
    {
        int tid = 0, team_size = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        team_size = omp_get_num_threads();
#endif
        auto& my_neighbors = thread_neighbors_[tid];

        const int chunk = 1 + centres / team_size;
        const int begin = std::min(tid * chunk, centres);
        const int end = std::min(begin + chunk, centres);
        for (int i = begin; i < end; ++i) {
            result.firstneigh[i] = static_cast<int>(my_neighbors.size());
            const int ci = atom_cell_[i];
            const int cz0 = ci % ncz_;
            const int cy0 = (ci / ncz_) % ncy_;
            const int cx0 = ci / (ncy_ * ncz_);

            const bool interior =
                cx0 >= stencil_reach_[0] && cx0 < ncx_ - stencil_reach_[0]
                && cy0 >= stencil_reach_[1] && cy0 < ncy_ - stencil_reach_[1]
                && cz0 >= stencil_reach_[2] && cz0 < ncz_ - stencil_reach_[2];

            for (auto s : full_stencil_) {
                int cj = ci + s.flat;
                if (!interior) {
                    const int cx = cx0 + s.di, cy = cy0 + s.dj, cz = cz0 + s.dk;
                    if (cx < 0 || cx >= ncx_ || cy < 0 || cy >= ncy_ || cz < 0
                        || cz >= ncz_)
                        continue;
                    cj = (cx * ncy_ + cy) * ncz_ + cz;
                }

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

                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 < rc_sq) {
                        my_neighbors.push_back(j);
                        ++result.numneigh[i];
                    }
                }
            }
        }
#pragma omp barrier
#pragma omp single
        {
            thread_offsets_[0] = 0;
            for (int t = 0; t < team_size; ++t)
                thread_offsets_[t + 1] =
                    thread_offsets_[t]
                    + static_cast<int>(thread_neighbors_[t].size());
            result.total_pairs = thread_offsets_[team_size];
            result.neighbors.resize(
                static_cast<std::size_t>(result.total_pairs));
        }
        const int offset = thread_offsets_[tid];
        std::copy(my_neighbors.begin(), my_neighbors.end(),
                  result.neighbors.begin() + offset);
        for (int i = begin; i < end; ++i)
            result.firstneigh[i] += offset;
    }
    for (int i = centres; i < n; ++i)
        result.firstneigh[i] = result.total_pairs;
}

void CellList::derive_half(const cpu::Frame& frame, int n_active,
                           const PairList& full, PairList& result,
                           int n_local) const {
    const int n = n_active < 0 ? frame.n_total() : n_active;
    const int owned = n_local < 0 ? frame.nlocal() : n_local;
    if (&full == &result || owned > n || full.inum < owned
        || full.n_rows < owned)
        throw std::invalid_argument("CellList half derivation requires a "
                                    "distinct full list covering owned rows");
    ensure_at_least(result.ilist, static_cast<std::size_t>(n));
    ensure_at_least(result.numneigh, static_cast<std::size_t>(n));
    ensure_at_least(result.firstneigh, static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        result.ilist[i] = i;
        result.numneigh[i] = 0;
    }
    result.inum = owned;
    result.n_rows = n;
    const StorageReal* x = frame.span(kAtom_x);
    const StorageReal* y = frame.span(kAtom_y);
    const StorageReal* z = frame.span(kAtom_z);
    const TagInt* tags = frame.span(kAtom_tag);
    // NPairHalffull assembly: owned pairs use row order; orthogonal ghost
    // pairs use image coordinates. Triclinic rows reuse NPairBin's tag parity
    // ownership so direct half and full-derived half share the same rule.
    const auto enumerate_row = [&](int i, const auto& append) {
        const int end = full.firstneigh[i] + full.numneigh[i];
        for (int k = full.firstneigh[i]; k < end; ++k) {
            const int j = full.neighbors[k];
            if (j < owned) {
                if (j <= i)
                    continue;
            } else if (triclinic_) {
                const bool odd = ((tags[i] ^ tags[j]) & TagInt{1}) != 0;
                if ((tags[i] > tags[j] && !odd) || (tags[i] < tags[j] && odd))
                    continue;
                if (tags[i] == tags[j]) {
                    if (std::abs(z[j] - z[i]) > 0.01) {
                        if (z[j] < z[i])
                            continue;
                    } else if (std::abs(y[j] - y[i]) > 0.01) {
                        if (y[j] < y[i])
                            continue;
                    } else if (x[j] < x[i])
                        continue;
                }
            } else {
                if (z[j] < z[i])
                    continue;
                if (z[j] == z[i]) {
                    if (y[j] < y[i])
                        continue;
                    if (y[j] == y[i] && x[j] < x[i])
                        continue;
                }
            }
            append(k);
        }
    };
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (n_threads == 1 || owned < n_threads) {
        result.neighbors.clear();
        for (int i = 0; i < owned; ++i) {
            result.firstneigh[i] = static_cast<int>(result.neighbors.size());
            enumerate_row(i, [&](int k) {
                result.neighbors.push_back(full.neighbors[k]);
            });
            result.numneigh[i] = static_cast<int>(result.neighbors.size())
                                 - result.firstneigh[i];
        }
        result.total_pairs = static_cast<int>(result.neighbors.size());
    } else {
        if (static_cast<int>(thread_neighbors_.size()) < n_threads)
            thread_neighbors_.resize(static_cast<std::size_t>(n_threads));
        ensure_at_least(thread_offsets_,
                        static_cast<std::size_t>(n_threads + 1));
        for (auto& neighbors : thread_neighbors_)
            neighbors.clear();
#pragma omp parallel num_threads(n_threads)
        {
            int tid = 0, team_size = 1;
#ifdef _OPENMP
            tid = omp_get_thread_num();
            team_size = omp_get_num_threads();
#endif
            auto& neighbors = thread_neighbors_[tid];
            const int chunk = 1 + owned / team_size;
            const int begin = std::min(tid * chunk, owned);
            const int end = std::min(begin + chunk, owned);
            for (int i = begin; i < end; ++i) {
                result.firstneigh[i] = static_cast<int>(neighbors.size());
                enumerate_row(
                    i, [&](int k) { neighbors.push_back(full.neighbors[k]); });
                result.numneigh[i] =
                    static_cast<int>(neighbors.size()) - result.firstneigh[i];
            }
#pragma omp barrier
#pragma omp single
            {
                thread_offsets_[0] = 0;
                for (int t = 0; t < team_size; ++t)
                    thread_offsets_[t + 1] =
                        thread_offsets_[t]
                        + static_cast<int>(thread_neighbors_[t].size());
                result.total_pairs = thread_offsets_[team_size];
                result.neighbors.resize(
                    static_cast<std::size_t>(result.total_pairs));
            }
            const int offset = thread_offsets_[tid];
            std::copy(neighbors.begin(), neighbors.end(),
                      result.neighbors.begin() + offset);
            for (int i = begin; i < end; ++i)
                result.firstneigh[i] += offset;
        }
    }
    for (int i = owned; i < n; ++i)
        result.firstneigh[i] = result.total_pairs;
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
                    + xt_bytes(cell_count_) + xt_bytes(cell_offset_);
    // `size()` (live elements) ≈ resident pages; reserved slack is lazily
    // mapped and not resident, so this subtracts cleanly against peak RSS.
    n += (half_stencil_.size() + full_stencil_.size()) * sizeof(StencilOffset);
    n += thread_offsets_.size() * sizeof(int);
    n += bin_write_ptr_.size() * sizeof(int);
    // Multi-thread scratch: contiguous neighbor indices.
    for (const auto& tp : thread_neighbors_)
        n += tp.size() * sizeof(int);
    return n;
}

} // namespace md
} // namespace cpu
} // namespace ATV
