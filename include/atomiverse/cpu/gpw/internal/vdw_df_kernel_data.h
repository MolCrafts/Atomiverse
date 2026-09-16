/// @file cpu/gpw/internal/vdw_df_kernel_data.h
/// @brief Accessors for RPS vdW-DF kernel tables (float32, inflate-on-first-use).
#pragma once

namespace ATV {
namespace cpu {
namespace gpw {
namespace vdw_df_data {

constexpr int kNAlpha = 20;
constexpr int kNRegions = 19;
constexpr int kNPCoef = 4;
constexpr int kNKpoints = 1025;
/// @brief Δk between kernel samples [1/Bohr] (2π/100).
constexpr double kDeltaK = 6.283185307179587e-02;
/// @brief q-mesh (length kNAlpha).
const double* qmesh();
/// @brief Kernel φ_αβ(k_i) as float32, layout [k][α][β], length kNKpoints*kNAlpha*kNAlpha.
const float* kernel_kaa();
/// @brief p_α(q) cubic spline coefs, layout [β][region][c0..c3], length kNAlpha*kNRegions*kNPCoef.
const float* splines_arc();

} // namespace vdw_df_data
} // namespace gpw
} // namespace cpu
} // namespace ATV
