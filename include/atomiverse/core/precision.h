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
/// **CUDA** uses one type per physical role (`cuda/frame/types.cuh`), selected
/// by the single knob `-DATV_CUDA_PREC_MODE=double|mixed|single` and modelled
/// on OpenMM's mixed precision: force math `force_t`, stored coordinate
/// `pos_t` (+ correction in mixed), integration state `mixed_t`, geometry
/// `geom_t`, reductions `acc_t` (always double) and 64-bit fixed-point force
/// accumulation `fixed_t`.
///
/// | tier            | role                                   | default | may lower? |
/// |-----------------|----------------------------------------|---------|-----------|
/// | `StorageReal`   | per-atom persistent state (x, v, f)    | double  | yes (float) |
/// | `WorkReal`      | transient per-kernel compute scalars   | double  | yes (float) |
/// | `AccReal`       | accumulators / reductions / energy / virial / SCF | double | **no** |
///
/// ### Integer tiers (ports LAMMPS `src/lmptype.h` roles)
/// Each role is sized independently so 64-bit width is paid only where a count
/// can actually overflow 32 bits.
///
/// | tier        | role                                   | LAMMPS twin | SMALLBIG | BIGBIG (default) |
/// |-------------|----------------------------------------|-------------|----------|------------------|
/// | `LocalInt`  | on-rank counts (nlocal, capacity, idx) | `smallint`  | int32    | int32            |
/// | `ImageInt`  | one periodic image flag (per component)| `imageint`  | int32    | int32            |
/// | `TagInt`    | global atom / molecule IDs             | `tagint`    | int32    | int64            |
/// | `BigInt`    | global totals (natoms, ntimestep)      | `bigint`    | int64    | int64            |
///
/// `ImageInt` is fixed at int32 regardless of `ATV_ID_SIZE`. LAMMPS packs all
/// three image counters into one `imageint` (10 bits each in an int32 under
/// SMALLBIG, 21 bits each in an int64 under `-DLAMMPS_BIGBIG`;
/// `src/lmptype.h`), so its reach is ±512 or ±2^20 crossings. Atomiverse
/// stores the three counters unpacked (`kAtom_image`, stride 3); one signed
/// 32-bit counter per component reaches ±2^31 crossings, beyond the BIGBIG
/// packed range, with no pack/unpack step. A stable run moves an atom by less
/// than one cell per step, so ±2^31 exceeds any step budget a run can afford;
/// int64 would double the column for no reachable gain. (The former int8
/// counter overflowed after ±127 crossings — reached by long diffusive,
/// transport and NEMD-drift runs.) int32 is also molpy's `ix/iy/iz` dtype, so
/// the molpy boundary copies images with no narrowing.
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

/// @brief One periodic image counter (LAMMPS `imageint` role, unpacked):
///        how many times an atom crossed the cell along one lattice vector.
///        Always 32-bit signed in every ID build (see the tier table).
using ImageInt = std::int32_t;

#ifdef ATV_ID_BIGBIG
/// @brief Global atom / molecule ID type (LAMMPS `tagint`), 64-bit under BIGBIG.
using TagInt = std::int64_t;
#else // ATV_ID_SMALLBIG (default)
/// @brief Global atom / molecule ID type (LAMMPS `tagint`), 32-bit under SMALLBIG.
using TagInt = std::int32_t;
#endif

} // namespace ATV
