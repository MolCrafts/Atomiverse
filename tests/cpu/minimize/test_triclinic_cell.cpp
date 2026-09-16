#include "atomiverse/cpu/minimize/triclinic_cell_relax.h"
#include <gtest/gtest.h>
using namespace ATV;
TEST(TriclinicCellRelax, Name) {
    cpu::minimizer::TriclinicCellRelax t(0.02, 1.0);
    EXPECT_EQ(t.name(), "triclinic_cell_relax");
}
