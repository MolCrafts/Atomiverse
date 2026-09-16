/// @file tests/cpu/gpw/test_cpu_gpw_workspace.cpp
/// @brief Unit tests for private GpwWorkspace bag (default + assign).

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/gpw_workspace.h"

using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::internal::GpwWorkspace;

// Default-constructed workspace has empty grids and zero-sized bags.
TEST(GpwWorkspace, DefaultIsEmpty) {
    const GpwWorkspace ws;
    EXPECT_EQ(ws.grid.n[0], 0);
    EXPECT_EQ(ws.grid.n[1], 0);
    EXPECT_EQ(ws.grid.n[2], 0);
    EXPECT_TRUE(ws.mg.levels.empty());
    EXPECT_TRUE(ws.products.by_level.empty());
    EXPECT_TRUE(ws.products.prims.empty());
    EXPECT_TRUE(ws.cbg.gidx.empty());
    EXPECT_EQ(ws.cbg.npts, 0);
    EXPECT_TRUE(ws.kbg.gidx.empty());
    EXPECT_EQ(ws.kbg.nb, 0);
    EXPECT_TRUE(ws.last_veff.empty());
}

// Assigning grid + last_veff is a plain value write (no factory).
TEST(GpwWorkspace, AssignGridAndLastVeff) {
    GpwWorkspace ws;
    UniformGrid g;
    g.n[0] = 4;
    g.n[1] = 4;
    g.n[2] = 4;
    g.L[0] = g.L[1] = g.L[2] = 10.0;
    ws.grid = g;
    ws.last_veff.assign(static_cast<std::size_t>(g.npts()), -0.5);
    EXPECT_EQ(ws.grid.npts(), 64);
    ASSERT_EQ(ws.last_veff.size(), 64u);
    EXPECT_DOUBLE_EQ(ws.last_veff[0], -0.5);
}
