/// @file cpu/pw/internal/poisson.cpp
#include "atomiverse/cpu/pw/internal/poisson.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

void PoissonSolver::step(PWWorkspace& ws) {
    for (int i = 0; i < ws.n_g; ++i) {
        AccReal g2 = ws.gvec->g[i].norm_squared();
        if (g2 < AccReal(1e-16)) {
            ws.V_H_G[i] = std::complex<AccReal>(0, 0);
        } else {
            ws.V_H_G[i] = AccReal(4.0 * 3.141592653589793) * ws.rho_G[i] / g2;
        }
    }
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
