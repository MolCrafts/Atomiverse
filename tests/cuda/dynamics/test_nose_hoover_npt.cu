/// @brief Constructor-only contract for the CUDA MTK barostat.
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include <gtest/gtest.h>

TEST(NoseHooverNPT, CtorMirrorsCpu) {
    ATV::cuda::md::NoseHooverNPT npt{300.0, 100.0, 0.0, 1000.0, 0.5};
    EXPECT_EQ(npt.name(), "nose_hoover_npt");
    EXPECT_DOUBLE_EQ(npt.T_target(), 300.0);
}
