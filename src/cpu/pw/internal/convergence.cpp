/// @file cpu/pw/internal/convergence.cpp
#include "atomiverse/cpu/pw/internal/convergence.h"
#include <cmath>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

bool ConvergenceCheck::step(const PWWorkspace& ws) {
    ++iter_;
    if (iter_ <= warmup_)
        return false;

    AccReal diff = 0;
    for (int i = 0; i < ws.n_g; ++i) {
        AccReal dr = std::abs(ws.rho_G[i] - ws.rho_G_old[i]);
        diff += dr * dr;
    }
    diff = std::sqrt(diff);
    return diff < tol_;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
