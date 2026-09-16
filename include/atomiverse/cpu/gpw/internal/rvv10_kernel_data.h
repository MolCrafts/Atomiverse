/// @file cpu/gpw/internal/rvv10_kernel_data.h
/// @brief Accessors for rVV10 RPS kernel tables (CP2K rVV10_kernel_table.dat).
#pragma once

namespace ATV {
namespace cpu {
namespace gpw {
namespace rvv10_data {

constexpr int kNAlpha = 20;
constexpr int kNKpoints = 1025; // 0..1024 inclusive
/// @brief r_max [Bohr] from CP2K table header (dk = 2π/r_max).
constexpr double kRMax = 100.0;
/// @brief Δk = 2π/r_max [1/Bohr].
constexpr double kDeltaK = 6.28318530717958647692528676655900576e-02;
/// @brief q-mesh length kNAlpha (ascending).
const double* qmesh();
/// @brief Kernel φ_αβ(k_i) float32 layout [k][α][β], length kNKpoints*kNAlpha*kNAlpha.
const float* kernel_kaa();

} // namespace rvv10_data
} // namespace gpw
} // namespace cpu
} // namespace ATV
