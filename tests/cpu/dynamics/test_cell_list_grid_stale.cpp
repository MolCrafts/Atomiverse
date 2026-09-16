/// @file tests/cpu/md/test_cell_list_grid_stale.cpp
/// @brief Unit test for `CellList::grid_stale` under anisotropic box change
///        (ac-005 of mc-gcmc-npt): a change to ONLY the y or z edge must
///        trigger a grid rebuild. Before the fix `grid_stale` compared
///        `length(0)` alone, so an anisotropic volume move silently kept a
///        grid sized for the old box and dropped pair candidates (the NPT
///        blow-up documented above the `grid_stale` call in cell_list.cpp).

#include <gtest/gtest.h>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/cell_list.h"

using namespace ATV;

namespace {

cpu::Frame make_frame(double lx, double ly, double lz) {
    cpu::Frame frame(8);
    for (int i = 0; i < 8; ++i) {
        frame.span(ATV::kAtom_x)[i] = 0.5 + 2.0 * (i % 2);
        frame.span(ATV::kAtom_y)[i] = 0.5 + 2.0 * ((i / 2) % 2);
        frame.span(ATV::kAtom_z)[i] = 0.5 + 2.0 * (i / 4);
        frame.span(ATV::kAtom_mass)[i] = 1.0;
    }
    frame.box.set_cell(Mat3::diagonal(lx, ly, lz));
    return frame;
}

} // namespace

TEST(CellListGridStale, EachAxisAloneTriggersRebuild) {
    for (int axis = 0; axis < 3; ++axis) {
        cpu::Frame frame = make_frame(10.0, 10.0, 10.0);
        cpu::md::CellList cl(/*cutoff=*/3.0, /*skin=*/0.5);
        cl.init(frame, 3.5);
        cl.build(frame);
        ASSERT_FALSE(cl.needs_rebuild(frame)) << "axis " << axis;

        Mat3 cell = frame.box.cell();
        cell(axis, axis) *= 1.2; // stretch ONLY this edge
        frame.box.set_cell(cell);
        EXPECT_TRUE(cl.needs_rebuild(frame))
            << "stretching only axis " << axis << " must stale the grid";
    }
}
