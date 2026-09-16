/// @file core/precision.h
/// @brief LAMMPS-style tiered numerical precision for Atomiverse.
///
/// Precision is *not* a single global switch. Following LAMMPS, the wide/safe
/// type is the default everywhere, and a narrower type (single-precision float,
/// 32-bit integer) is spent only at deliberately chosen *roles* — never at the
/// accumulators or global counters where it would corrupt results.
///
/// ### Floating-point tiers
///
/// **CPU / shared** (`StorageReal` / `WorkReal` / `AccReal`):
/// only storage and work may be lowered to `float`; ::ATV::AccReal is
/// **always** `double` — the "key location" precision that is never economized.
///
/// **CUDA** (`real_t` / `work_t` / `prec_t` in `cuda/frame/types.cuh`) follows
/// **LAMMPS Kokkos** (`KOKKOS_PREC`), not a free three-axis split:
/// one mode knob `-DATV_CUDA_PREC_MODE=double|mixed|single` maps to
/// `(KK_FLOAT, KK_ACC_FLOAT)` with `work_t ≡ real_t ≡ KK_FLOAT` and
/// `prec_t ≡ KK_ACC_FLOAT`. There is no independent CUDA "work" precision.
///
/// | tier            | role                                   | CUDA twin | default | may lower? |
/// |-----------------|----------------------------------------|-----------|---------|-----------|
/// | `StorageReal`   | per-atom persistent state (x, v, f)    | `real_t`  | double  | yes (float) |
/// | `WorkReal`      | transient per-kernel compute scalars   | `work_t`  | double  | yes (float) |
/// | `AccReal`       | accumulators / reductions / energy / virial / SCF | `prec_t` | double | **no** (CPU); CUDA `single` mode may lower `prec_t` |
///
/// ### Integer tiers (ports LAMMPS `src/lmptype.h` roles)
/// Each role is sized independently so 64-bit width is paid only where a count
/// can actually overflow 32 bits.
///
/// | tier        | role                                   | LAMMPS twin | SMALLBIG | BIGBIG (default) |
/// |-------------|----------------------------------------|-------------|----------|------------------|
/// | `LocalInt`  | on-rank counts (nlocal, capacity, idx) | `smallint`  | int32    | int32            |
/// | `ImageInt`  | periodic image flags                   | `imageint`  | int32    | int64            |
/// | `TagInt`    | global atom / molecule IDs             | `tagint`    | int32    | int64            |
/// | `BigInt`    | global totals (natoms, ntimestep)      | `bigint`    | int64    | int64            |
///
/// Atomiverse defaults to BIGBIG (64-bit IDs) to preserve the legacy always-int64
/// tag width — a byte-identical default; SMALLBIG (32-bit IDs, the LAMMPS default)
/// is the memory-saving opt-in.
///
/// Configure at build time with `-DATV_REAL_STORAGE=float|double`,
/// `-DATV_REAL_WORK=float|double`, and `-DATV_ID_SIZE=smallbig|bigbig`
/// (see the root CMakeLists.txt).
#pragma once

#include <cstdint>

namespace ATV {

// ─────────────────────────── Floating-point tiers ───────────────────────────

#ifndef ATV_REAL_STORAGE_TYPE
/// @brief Storage-tier real type when CMake does not override it.
#define ATV_REAL_STORAGE_TYPE double
#endif

#ifndef ATV_REAL_WORK_TYPE
/// @brief Work-tier real type when CMake does not override it.
#define ATV_REAL_WORK_TYPE double
#endif

/// @brief Precision of per-atom *stored* fields (positions, velocities, forces
/// as persisted in a Frame). Lower to `float` for memory-bound large systems.
using StorageReal = ATV_REAL_STORAGE_TYPE;

/// @brief Precision of transient per-kernel compute scalars (local distances,
/// pair terms). Independent of storage so kernels can compute wide over narrow
/// storage.
using WorkReal = ATV_REAL_WORK_TYPE;

/// @brief Precision of accumulators and reductions — energy sums, force
/// accumulation, virial, SCF matrices. **Always `double`**: this is the LAMMPS
/// "double accumulation" rule, the one place precision is never economized.
using AccReal = double;

// ───────────────────────────── Integer tiers ────────────────────────────────

/// @brief On-rank count / index type (LAMMPS `smallint`). Always 32-bit signed.
using LocalInt = std::int32_t;

/// @brief Global system-total count (LAMMPS `bigint`): total atom count across
/// ranks, timestep index. **Always 64-bit** so a global count never overflows.
using BigInt = std::int64_t;

#ifdef ATV_ID_BIGBIG
/// @brief Periodic image-flag type (LAMMPS `imageint`), 64-bit under BIGBIG.
using ImageInt = std::int64_t;
/// @brief Global atom / molecule ID type (LAMMPS `tagint`), 64-bit under BIGBIG.
using TagInt = std::int64_t;
#else // ATV_ID_SMALLBIG (default)
/// @brief Periodic image-flag type (LAMMPS `imageint`), 32-bit under SMALLBIG.
using ImageInt = std::int32_t;
/// @brief Global atom / molecule ID type (LAMMPS `tagint`), 32-bit under SMALLBIG.
using TagInt = std::int32_t;
#endif

} // namespace ATV
