/// @file cuda/frame/types.cuh
/// @brief CUDA force precision tiers and physical-state integration precision.
///
/// LAMMPS Kokkos (`KOKKOS_PREC`) exposes one compile-time mode and two types:
///
/// | mode   | `KK_FLOAT` (storage + force math) | `KK_ACC_FLOAT` (force/energy/virial acc) |
/// |--------|-----------------------------------|------------------------------------------|
/// | double | double                            | double                                   |
/// | mixed  | float                             | double                                   |
/// | single | float                             | float                                    |
///
/// Atomiverse uses those force precision tiers:
///
/// | ATV type | role                                      | LMP twin        |
/// |----------|-------------------------------------------|-----------------|
/// | `real_t` | main positions, forces and charges        | `KK_FLOAT`      |
/// | `work_t` | kernel temporaries / shared-mem params    | `KK_FLOAT`      |
/// | `prec_t` | energy / virial accumulation      | `KK_ACC_FLOAT`  |
///
/// Physical motion uses `motion_t = prec_t`: double in mixed/double, float
/// in single. Velocities and masses use motion_t; mixed positions keep a
/// real_t main value plus a real_t residual. Force kernels still read only
/// the main positions and use work_t arithmetic.
///
/// **`work_t` is not a free third precision axis.** It always equals `real_t`
/// (same as LAMMPS: force math uses the storage precision). Older docs that
/// advertised independent `-DATV_CUDA_WORK` are obsolete; pick the mode only:
///
///   cmake … -DATV_CUDA_PREC_MODE=double|mixed|single
///
/// Constraint: sizeof(real_t) <= sizeof(work_t) <= sizeof(prec_t) (always true
/// under the three modes above). Host-side Frame/CPU code uses
/// `ATV::StorageReal` / `WorkReal` / `AccReal` from `core/precision.h`
/// (CPU AccReal is always double).
#pragma once

#include <type_traits>

namespace ATV {
namespace cuda {

#ifndef ATV_CUDA_REAL_TYPE
#define ATV_CUDA_REAL_TYPE double
#endif

#ifndef ATV_CUDA_WORK_TYPE
#define ATV_CUDA_WORK_TYPE double
#endif

#ifndef ATV_CUDA_PREC_TYPE
#define ATV_CUDA_PREC_TYPE double
#endif

using real_t = ATV_CUDA_REAL_TYPE;
using work_t = ATV_CUDA_WORK_TYPE;
using prec_t = ATV_CUDA_PREC_TYPE;
using motion_t = prec_t;

/// @brief Whether positions need a residual to retain mixed integration state.
inline constexpr bool kMixedPrecision =
    std::is_same_v<real_t, float> && std::is_same_v<motion_t, double>;

static_assert(sizeof(real_t) <= sizeof(work_t),
              "real_t must not exceed work_t");
static_assert(sizeof(work_t) <= sizeof(prec_t),
              "work_t must not exceed prec_t");

// Mode invariants (LAMMPS KOKKOS_PREC mapping).
static_assert(sizeof(real_t) == sizeof(work_t),
              "work_t must equal real_t (LAMMPS KK_FLOAT); no free work axis");

} // namespace cuda
} // namespace ATV
