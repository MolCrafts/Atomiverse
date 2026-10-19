/// Cartesian grid coverage and lattice alignment with explicit halo rows.
/// Geometry: L=15.78, reach=2.2, bin edge=L/14, fractional slab [.25,.5].
/// Every axis has a halo margin plus one safety bin, including dims==1.

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

    dd.sublo[0] = kSublo / kBoxL;
    dd.subhi[0] = kSubhi / kBoxL;
    dd.sublo[1] = 0.0;
    dd.subhi[1] = 1.0;
    dd.sublo[2] = 0.0;
    dd.subhi[2] = 1.0;
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
    // All axes carry halo cells plus one safety bin on either side.
    const int transverse_cells = dd.ghost_shell_width > kCutoffSkin ? 28 : 20;
    const int stride = transverse_cells * transverse_cells;
    cl.init_extended(frame, kCutoffSkin, &dd, n);
    cl.build(frame, n);
    std::vector<int> bins(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int flat = cl.cell_of(i);
        EXPECT_EQ(flat % stride, cl.cell_of(0) % stride)
            << "row " << i << " left the shared (cy, cz) bin: the "
            << "all rows share the same transverse coordinates";
        bins[static_cast<std::size_t>(i)] = flat / stride;
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
/// Low/high probes must occupy consecutive bins, including the safety margin.
TEST(TestCellListHaloPad, ExtendedAxisWindowCoversGhostShell) {
    const double cs = cell_edge();
    const DomainDecomposition dd = stub_dd(kShell);
    const cpu::Frame frame =
        probe_frame({kSublo - 6.5, kSublo - 6.5 + cs, kSublo - 6.5 + 2.0 * cs,
                     kSublo + 0.1, kSubhi + 6.5 - cs, kSubhi + 6.5});

    cpu::md::CellList cl(kCutoff, kSkin);
    const std::vector<int> bins = x_bins(frame, cl, dd);
    ASSERT_EQ(bins.size(), 6u);

    const int expect[6] = {1, 2, 3, 7, 15, 16};
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
    // The safety margin leaves the owner in x bin 3.
    EXPECT_EQ(bins[2], 3);
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
/// Reached through the spatial index geometry query, like
/// `tests/cpu/dynamics/test_cell_list_grid_stale.cpp`.
TEST(TestCellListHaloPad, RaisingGhostShellWidthStalesTheGrid) {
    DomainDecomposition dd = stub_dd(kShell);
    const cpu::Frame frame =
        probe_frame({kSublo + 0.1, kSublo + 1.0, kSublo + 2.0});
    const int n = frame.nlocal();

    cpu::md::CellList cl(kCutoff, kSkin);
    cl.init_extended(frame, kCutoffSkin, &dd, n);
    cl.build(frame, n);
    ASSERT_FALSE(cl.grid_stale(frame))
        << "nothing moved: the grid must be fresh right after build()";

    dd.ghost_shell_width = 9.0; // deeper shell, same box and same slab
    EXPECT_TRUE(cl.grid_stale(frame))
        << "a wider ghost shell needs a wider extended-axis window";

    // The comparison is against the CAPTURED width, not a one-way latch:
    // restoring it makes the grid fresh again (no rebuild ran in between).
    dd.ghost_shell_width = kShell;
    EXPECT_FALSE(cl.grid_stale(frame));
}
