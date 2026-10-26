/// @file cpu/planewave/internal/convergence.cpp
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include <cmath>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

bool ConvergenceCheck::step(AccReal resid_l2) {
    ++iter_;
    if (iter_ <= warmup_)
        return false;
    return resid_l2 < tol_;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
