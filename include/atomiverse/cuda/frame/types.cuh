/// @file cuda/frame/types.cuh
/// @brief CUDA backend floating-point types — **aligned with LAMMPS Kokkos**.
///
/// LAMMPS Kokkos (`KOKKOS_PREC`) exposes one compile-time mode and two types:
///
/// | mode   | `KK_FLOAT` (storage + force math) | `KK_ACC_FLOAT` (force/energy/virial acc) |
/// |--------|-----------------------------------|------------------------------------------|
/// | double | double                            | double                                   |
/// | mixed  | float                             | double                                   |
/// | single | float                             | float                                    |
///
/// Atomiverse mirrors that exactly:
///
/// | ATV type | role                                      | LMP twin        |
/// |----------|-------------------------------------------|-----------------|
/// | `real_t` | per-atom storage (x, v, f, charges, …)    | `KK_FLOAT`      |
/// | `work_t` | kernel temporaries / shared-mem params    | `KK_FLOAT`      |
/// | `prec_t` | energy / virial / force accumulation      | `KK_ACC_FLOAT`  |
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

static_assert(sizeof(real_t) <= sizeof(work_t),
              "real_t must not exceed work_t");
static_assert(sizeof(work_t) <= sizeof(prec_t),
              "work_t must not exceed prec_t");

// Mode invariants (LAMMPS KOKKOS_PREC mapping).
static_assert(sizeof(real_t) == sizeof(work_t),
              "work_t must equal real_t (LAMMPS KK_FLOAT); no free work axis");

} // namespace cuda
} // namespace ATV
