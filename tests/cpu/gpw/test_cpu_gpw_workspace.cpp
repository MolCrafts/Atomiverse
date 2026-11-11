/// @file tests/cpu/gpw/test_cpu_gpw_workspace.cpp
#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/gpw_workspace.h"

TEST(GpwWorkspace, StartsWithNoAcceptedPotentials) {
    const ATV::cpu::gpw::internal::GpwWorkspace workspace;
    EXPECT_TRUE(workspace.veff.empty());
    EXPECT_TRUE(workspace.veff_a.empty());
    EXPECT_TRUE(workspace.veff_b.empty());
    EXPECT_TRUE(workspace.v_es.empty());
}
