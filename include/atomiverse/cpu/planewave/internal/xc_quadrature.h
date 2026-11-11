#pragma once

namespace ATV::cpu::pw::internal {
// Shared CPU/CUDA private XC quadrature. The density/band FFT is unchanged.
// Even dimensions sample non-polynomial LSDA at >=1.5x and GGA at >=3x.
inline int xc_quadrature_dimension(int n, bool gga) {
    return 2 * (((gga ? 6 : 3) * n + 3) / 4);
}
} // namespace ATV::cpu::pw::internal
