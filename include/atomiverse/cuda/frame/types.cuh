/// @file cuda/frame/types.cuh
/// @brief CUDA backend precision contract: one type per physical role.
///
/// `ATV_CUDA_PREC_MODE` (double | mixed | single) selects the mode at configure
/// time; there is no other precision knob. Classical MD follows OpenMM's mixed
/// strategy (reference: openmm `platforms/common/src/kernels/verlet.cc`,
/// commit 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e); bit identity with OpenMM is
/// not a goal.
///
/// | role | type | double | mixed | single |
/// |------|------|--------|-------|--------|
/// | force math, splines, grids, FFT, charges | `force_t` | FP64 | FP32 | FP32 |
/// | stored coordinate (force-kernel input) | `pos_t` | FP64 | FP32 | FP32 |
/// | stored coordinate correction | `pos_t` | — | FP32 | — |
/// | reconstructed coordinate, velocity, integration mass, dt, thermostat/barostat state | `mixed_t` | FP64 | FP64 | FP32 |
/// | cell and derived geometry | `geom_t` | FP64 | FP64 | FP64 |
/// | energy, virial, kinetic energy, global reductions | `acc_t` | FP64 | FP64 | FP64 |
/// | classical force accumulation | `fixed_t` | 64-bit fixed point | 64-bit fixed point | 64-bit fixed point |
///
/// In mixed mode a coordinate is `x + x_corr`: integrators reconstruct it in
/// FP64, update it, and split it again (`cuda/frame/precision.cuh`); ordinary
/// force kernels read only the FP32 primary value. Every other mode stores the
/// coordinate exactly and has no correction column (`kPositionCorrection`).
///
/// Electronic-structure kernels (`cuda/scf/`) use `real_t / work_t / prec_t`,
/// which are FP64 in every mode. Classical MD code never uses them.
#pragma once

#include <type_traits>

namespace ATV {
namespace cuda {

/// @brief Configured CUDA precision mode.
enum class PrecisionMode { Double, Mixed, Single };

#if defined(ATV_CUDA_PRECISION_MIXED)
inline constexpr PrecisionMode kPrecisionMode = PrecisionMode::Mixed;
#elif defined(ATV_CUDA_PRECISION_SINGLE)
inline constexpr PrecisionMode kPrecisionMode = PrecisionMode::Single;
#else
inline constexpr PrecisionMode kPrecisionMode = PrecisionMode::Double;
#endif

/// Force math, splines, grids, FFT, charges and force-path coordinates.
using force_t =
    std::conditional_t<kPrecisionMode == PrecisionMode::Double, double, float>;
/// Stored coordinate primary value (and correction, when present).
using pos_t = force_t;
/// Integration state: reconstructed coordinates, velocities, masses, dt,
/// thermostat and barostat variables.
using mixed_t =
    std::conditional_t<kPrecisionMode == PrecisionMode::Single, float, double>;
/// Cell matrix and its derived geometry (volume, inverse, lengths).
using geom_t = double;
/// Energy, virial, kinetic energy and every global reduction.
using acc_t = double;
/// Classical force accumulation word (signed 64-bit fixed point).
using fixed_t = long long;

/// True when stored coordinates carry a separate correction column.
inline constexpr bool kPositionCorrection = !std::is_same_v<pos_t, mixed_t>;

static_assert(sizeof(force_t) <= sizeof(mixed_t),
              "integration state must not be narrower than force math");
static_assert(sizeof(fixed_t) == 8, "fixed-point forces are 64-bit");

/// @name Electronic-structure kernel tiers (Gaussian SCF only).
/// Gaussian SCF is FP64 in every mode: its CPU↔CUDA energy/force contract is
/// stated at the 1e-10 level, which no FP32 tier can meet.
/// @{
using real_t = double;
using work_t = double;
using prec_t = double;
/// @}

} // namespace cuda
} // namespace ATV
