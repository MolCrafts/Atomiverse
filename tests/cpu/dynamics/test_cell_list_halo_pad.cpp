/// @file tests/cpu/dynamics/test_cell_list_halo_pad.cpp
/// @brief Unit tests for the halo-coupled cell-grid pad —
///        spec `.claude/specs/comm-topology-02-multihop.md`, coupling point
///        "cell-list 垫层联动".
///
/// `CellList::init_extended` today ignores its `cutoff_skin` parameter and
/// pads every decomposed axis by exactly `cutoff + skin`
/// (`src/cpu/neighbor/cell_list.cpp:47`, `:96-107`). Once the halo grows a
/// multi-hop ghost shell (`max_hops > 1`, or a bonded extension wider than
/// `cutoff + skin`), the ghosts land OUTSIDE that window and `bin_atoms`
/// clamps them into the edge cell (`:215-217`) — silently, with no missing
/// atom and no error, only wrong cell membership. `-02` therefore pads the
/// extended axes by `max(cutoff_skin, dd->ghost_shell_width)` while keeping
/// the cell edge derived from the GLOBAL box division (cross-rank lattice
/// alignment, the invariant argued at `:76-88`).
///
/// Surface pinned here (new in `-02`):
/// @code{.cpp}
///   double DomainDecomposition::ghost_shell_width;  // [Angstrom], 0 = none
///   int    CellList::cell_of(int row) const;        // flat bin of a row
/// @endcode
///
/// `cell_of` is the only observable of binning. Within a single process the
/// clamp is *pair-invariant*: it merges cells, and every merged candidate
/// is then rejected by the distance filter in `refresh_half/full`, so the
/// published `PairList` looks identical either way. What the clamp actually
/// destroys is the cross-rank symmetry of the home/forward cell
/// classification — an MPI-level effect covered by
/// `tests/mpi/test_halo_dual_image_ghosts.cpp`. The unit-level contract is
/// therefore cell membership itself; `n_at_last_build()` is the in-tree
/// precedent for this kind of read-only introspection on `CellList`.
///
/// Serial: the `DomainDecomposition` is a hand-filled stub (`dims = {4,1,1}`,
/// this rank at `coords_x = 1`), no MPI is initialized, and every expected
/// number below is a literal derived from the spec's geometry.
///
/// Geometry shared by every case (spec Testing strategy):
///   * cubic box `L = 15.78 A` (3^3 lattice at a0 = 5.26 A),
///   * `cutoff = 2.0 A`, `skin = 0.2 A` => `cutoff_skin = 2.2 A`,
///   * LAMMPS half-reach bins: global cell count per axis
///     `floor(L / (0.5 * 2.2)) = 14`, so the GLOBAL cell edge is
///     `cs = 15.78 / 14 = 1.127142857... A` on every axis,
///   * subdomain `[sublo, subhi) = [3.945, 7.89) A` (L/4, coords_x = 1),
///   * ghost shell `6.6 A` (pair cutoff 6.0 + auto skin 0.6 of the narrow
///     4-rank case), i.e. `max_hops = 2` along x.
///
/// Categories: basics (pad follows the shell), edge (`shell == 0`
/// degenerates to today's pad; lattice alignment survives the pad),
/// lifecycle (`grid_stale` re-derivation when the shell grows).

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/cell_list.h"

using namespace ATV;

namespace {

using DomainDecomposition = cpu::md::mpi::DomainDecomposition;

constexpr double kBoxL = 15.78;     // [Angstrom]
constexpr double kCutoff = 2.0;     // [Angstrom]
constexpr double kSkin = 0.2;       // [Angstrom]
constexpr double kCutoffSkin = 2.2; // [Angstrom]
constexpr double kShell = 6.6;      // ghost shell width [Angstrom]
constexpr int kGlobalCells = 14;    // floor(15.78 / 1.1)
constexpr double kSublo = 3.945;    // [Angstrom] = 1 * L / 4
constexpr double kSubhi = 7.89;     // [Angstrom] = 2 * L / 4

/// Global cell edge, identical on every axis and on every rank:
/// `L / floor(L / (0.5 * cutoff_skin))` = 15.78 / 14.
double cell_edge() { return kBoxL / kGlobalCells; }

/// Cells per undecomposed axis => the stride between two consecutive x
/// bins in the flat index `cx * ncy * ncz + cy * ncz + cz`. y and z carry
/// `dims == 1`, so they keep the global 14-cell grid: stride = 196.
constexpr int kXStride = kGlobalCells * kGlobalCells;

/// @brief `dims = {4,1,1}` stub over the cubic box, this rank at coords 1.
/// @param shell Ghost shell width [Angstrom] the halo would grow
///        (`CommBrick` writes `max(cutoff, bonded_extension)` here).
DomainDecomposition stub_dd(double shell) {
    DomainDecomposition dd;
    dd.dims[0] = 4;
    dd.dims[1] = 1;
    dd.dims[2] = 1;
    dd.coords[0] = 1;
    dd.coords[1] = 0;
    dd.coords[2] = 0;

    dd.sublo[0] = kSublo;
    dd.subhi[0] = kSubhi;
    dd.sublo[1] = 0.0;
    dd.subhi[1] = kBoxL;
    dd.sublo[2] = 0.0;
    dd.subhi[2] = kBoxL;
    dd.face_neighbors = std::array<int, 6>{0, 2, 1, 1, 1, 1};
    dd.ghost_shell_width = shell;
    return dd;
}

/// @brief Frame of probe rows at the given x, all sharing y = z = 7.89 A.
///
/// A shared (y, z) puts every row in the same `(cy, cz)` bin, so two rows'
/// flat cell indices differ by exactly `(cx_a - cx_b) * kXStride` and the
/// x bin is recoverable as `cell_of(row) / kXStride`.
cpu::Frame probe_frame(const std::vector<double>& xs) {
    const int n = static_cast<int>(xs.size());
    cpu::Frame f(n);
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] =
            static_cast<StorageReal>(xs[static_cast<std::size_t>(i)]);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(kSubhi);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(kSubhi);
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(kBoxL);
    return f;
}

/// Bin the probes and return each row's x cell index.
std::vector<int> x_bins(const cpu::Frame& frame, cpu::md::CellList& cl,
                        const DomainDecomposition& dd) {
    const int n = frame.nlocal();
    cl.init_extended(frame, kCutoffSkin, &dd, n);
    cl.build(frame, n);
    std::vector<int> bins(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int flat = cl.cell_of(i);
        EXPECT_EQ(flat % kXStride, cl.cell_of(0) % kXStride)
            << "row " << i << " left the shared (cy, cz) bin: the "
            << "undecomposed axes must keep the global " << kGlobalCells
            << "-cell grid";
        bins[static_cast<std::size_t>(i)] = flat / kXStride;
    }
    return bins;
}

} // namespace

// ── 1. compute_grid: pad = max(cutoff_skin, ghost_shell_width) ────────

/// The extended-axis window must cover `[sublo - 6.6, subhi + 6.6]`, not
/// `[sublo - 2.2, subhi + 2.2]`. Probed through binning: rows one tenth of
/// an Angstrom inside each end of the ghost shell, plus their `+cs`
/// neighbours, land in DISTINCT consecutive cells instead of collapsing
/// into the clamped edge cell.
///
/// Hard-coded expectations (cs = 15.78/14 = 1.1271429 A, window low edge
/// snapped down to the global lattice at `-3 * cs = -3.3814286 A`, 16 cells
/// along x):
///
///   x = sublo - 6.5   = -2.555   -> (x - lo)/cs = 0.7331 -> cx 0
///   x = sublo - 6.5 + cs         -> 1.7331               -> cx 1
///   x = sublo - 6.5 + 2cs        -> 2.7331               -> cx 2
///   x = sublo + 0.1   =  4.045   -> 6.5898               -> cx 6
///   x = subhi + 6.5 - cs = 13.2629 -> 14.768             -> cx 14
///   x = subhi + 6.5   = 14.39    -> 15.768               -> cx 15
///
/// `cx = 0` for the deepest low-side probe is not a clamp: the window's low
/// edge is lattice-snapped and therefore at most one cell below
/// `sublo - shell`, so a row 0.1 A inside the shell necessarily sits in the
/// first or second cell. What proves the absence of clamping is that the
/// three low-side probes (and the two high-side ones) occupy *consecutive*
/// cells — under today's 2.2 A pad they would all fold onto one edge cell,
/// which is exactly what `ZeroShellKeepsTodaysCutoffSkinPad` below pins.
TEST(TestCellListHaloPad, ExtendedAxisWindowCoversGhostShell) {
    const double cs = cell_edge();
    const DomainDecomposition dd = stub_dd(kShell);
    const cpu::Frame frame =
        probe_frame({kSublo - 6.5, kSublo - 6.5 + cs, kSublo - 6.5 + 2.0 * cs,
                     kSublo + 0.1, kSubhi + 6.5 - cs, kSubhi + 6.5});

    cpu::md::CellList cl(kCutoff, kSkin);
    const std::vector<int> bins = x_bins(frame, cl, dd);
    ASSERT_EQ(bins.size(), 6u);

    const int expect[6] = {0, 1, 2, 6, 14, 15};
    for (int i = 0; i < 6; ++i) {
        SCOPED_TRACE("probe row " + std::to_string(i));
        EXPECT_EQ(bins[static_cast<std::size_t>(i)], expect[i]);
    }
}

/// `ghost_shell_width == 0` — `CommBrick` has not built yet, or the run is
/// single-rank — degenerates to today's `cutoff_skin` pad: the window is
/// `[sublo - 2.2, subhi + 2.2]`, snapped to the lattice at `[cs, 9 * cs)`.
/// Rows deeper than that fold onto the edge cell, which is precisely the
/// silent failure a real 6.6 A shell must not hit.
TEST(TestCellListHaloPad, ZeroShellKeepsTodaysCutoffSkinPad) {
    const double cs = cell_edge();
    const DomainDecomposition dd = stub_dd(0.0);
    const cpu::Frame frame =
        probe_frame({kSublo - 6.5, kSublo - 6.5 + cs, kSublo + 0.1});

    cpu::md::CellList cl(kCutoff, kSkin);
    const std::vector<int> bins = x_bins(frame, cl, dd);
    ASSERT_EQ(bins.size(), 3u);

    // Both deep rows are outside the 2.2 A window and clamp to cell 0.
    EXPECT_EQ(bins[0], 0);
    EXPECT_EQ(bins[1], 0);
    // The owner is inside it: (4.045 - cs) / 1.1271429 = 2.589 -> cell 2.
    EXPECT_EQ(bins[2], 2);
}

// ── 2. compute_grid: the pad must not break lattice alignment ─────────

/// The cell edge stays `L / floor(L / cutoff_skin)` and the window origin
/// stays an integer multiple of it, so cell boundaries coincide across
/// ranks (the home/forward classification of `refresh_half` depends on it,
/// `cell_list.cpp:76-88`). Probed without reading the origin: two rows
/// straddling the GLOBAL lattice plane `2 * cs = 2.2542857 A` by 1e-3 A
/// must fall in different cells, while two rows straddling the mid-cell
/// plane `2.5 * cs = 2.8178571 A` by the same amount must share one.
TEST(TestCellListHaloPad, PadKeepsCellEdgeOnTheGlobalLattice) {
    const double cs = cell_edge();
    const double delta = 1.0e-3; // [Angstrom]
    const DomainDecomposition dd = stub_dd(kShell);
    const cpu::Frame frame = probe_frame({2.0 * cs - delta, 2.0 * cs + delta,
                                          2.5 * cs - delta, 2.5 * cs + delta});

    cpu::md::CellList cl(kCutoff, kSkin);
    const std::vector<int> bins = x_bins(frame, cl, dd);
    ASSERT_EQ(bins.size(), 4u);

    EXPECT_EQ(bins[1], bins[0] + 1)
        << "no cell boundary at the global lattice plane 2 * cs";
    EXPECT_EQ(bins[3], bins[2])
        << "a cell boundary appeared at the mid-cell plane 2.5 * cs";
}

// ── 3. grid_stale: a widening shell re-derives the grid ───────────────

/// `grid_stale()` compares the live box and subdomain against what
/// `compute_grid` captured; `-02` adds the captured shell width, so raising
/// `dd->ghost_shell_width` ALONE (box and slab unchanged, atoms frozen)
/// must stale the grid. Without it a run whose bonded extension or hop
/// count grows mid-flight keeps binning into a too-narrow window.
/// Reached through the public `needs_rebuild`, like
/// `tests/cpu/dynamics/test_cell_list_grid_stale.cpp`.
TEST(TestCellListHaloPad, RaisingGhostShellWidthStalesTheGrid) {
    DomainDecomposition dd = stub_dd(kShell);
    const cpu::Frame frame =
        probe_frame({kSublo + 0.1, kSublo + 1.0, kSublo + 2.0});
    const int n = frame.nlocal();

    cpu::md::CellList cl(kCutoff, kSkin);
    cl.init_extended(frame, kCutoffSkin, &dd, n);
    cl.build(frame, n);
    ASSERT_FALSE(cl.needs_rebuild(frame, n))
        << "nothing moved: the grid must be fresh right after build()";

    dd.ghost_shell_width = 9.0; // deeper shell, same box and same slab
    EXPECT_TRUE(cl.needs_rebuild(frame, n))
        << "a wider ghost shell needs a wider extended-axis window";

    // The comparison is against the CAPTURED width, not a one-way latch:
    // restoring it makes the grid fresh again (no rebuild ran in between).
    dd.ghost_shell_width = kShell;
    EXPECT_FALSE(cl.needs_rebuild(frame, n));
}
