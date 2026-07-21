/// @file cpu/gpw/internal/grid_fft.h
/// @brief FFT-based spectral gradient and divergence on the GPW uniform grid.
///
/// For periodic fields on an orthorhombic UniformGrid these compute exact
/// (spectral) derivatives: grad f = IFFT(i G FFT(f)), div v = IFFT(i G . FFT(v)),
/// with G_d = 2 pi m_d / L_d and the same signed-frequency / 1/N normalization
/// convention as `poisson_fft`. The Nyquist mode of each odd-derivative is zeroed
/// (standard) so the result stays real. Used by the GGA XC path for density
/// gradients and the divergence term of the XC potential.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h" // UniformGrid

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Spectral gradient of a real periodic field.
/// @return {df/dx, df/dy, df/dz}, each length grid.npts().
std::array<std::vector<double>, 3> gradient(const UniformGrid& grid,
                                            const std::vector<double>& f);

/// @brief Spectral divergence of a real periodic vector field.
/// @param v {v_x, v_y, v_z}, each length grid.npts().
/// @return div v, length grid.npts().
std::vector<double> divergence(const UniformGrid& grid,
                               const std::array<std::vector<double>, 3>& v);

} // namespace gpw
} // namespace cpu
} // namespace ATV
