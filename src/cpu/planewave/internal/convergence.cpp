/// @file cpu/pw/internal/convergence.cpp
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include <cmath>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

bool ConvergenceCheck::step(const PWWorkspace& ws) {
    // Counting must happen even when the norm is not needed, so the warmup
    // gate lives in the AccReal overload and this one always computes.
    AccReal diff = 0;
    for (int i = 0; i < ws.n_g; ++i) {
        AccReal dr = std::abs(ws.rho_G[i] - ws.rho_G_old[i]);
        diff += dr * dr;
    }
    return step(std::sqrt(diff));
}

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
