/// @file cpu/pw/internal/kpoint.h
/// @brief Monkhorst-Pack k-point mesh generator.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/core/vec3.h"
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

struct KPointGrid {
    std::vector<Vec3> kpoints; // fractional coordinates
    std::vector<AccReal> weights;
};

KPointGrid generate_mp_mesh(int nk1, int nk2, int nk3, AccReal shift1 = 0,
                            AccReal shift2 = 0, AccReal shift3 = 0);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
