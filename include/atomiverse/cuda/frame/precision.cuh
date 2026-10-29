/// @file cuda/frame/precision.cuh
/// @brief Device/host codecs of the CUDA precision contract: split coordinates
///        and 64-bit fixed-point forces.
///
/// Coordinates: `load_position` reconstructs the integration value from the
/// stored primary + correction pair, `store_position` splits it again
/// (`hi = round(v)`, `lo = round(v - hi)`), exactly OpenMM's mixed-precision
/// position update.
///
/// Forces: OpenMM accumulates `(long long)(f * 2^32)` with f in kJ/mol/nm.
/// ATV keeps eV/Angstrom and folds the unit conversion into the scale, so the
/// resolution (2.4e-13 eV/Angstrom) and the range (|f| < 2.2e6 eV/Angstrom)
/// are OpenMM's. A contribution is scaled in its own precision and truncated
/// toward zero. Encodes that leave the signed 64-bit range (or are NaN/Inf)
/// and wrapping sums raise the Frame's `ForceFault` instead of silently
/// corrupting the force; integrators refuse to consume a faulted force.
///
/// Requires nvcc (atomics); host-only conversions are `__host__ __device__`.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

/// @brief kJ/mol/nm per eV/Angstrom (CODATA 2018 eV in kJ/mol, times 10).
inline constexpr double kKjPerMolNmPerEvPerAngstrom = 964.8533212331002;

/// @brief Fixed-point force scale [1 / (eV/Angstrom)]: OpenMM's 2^32 per
///        kJ/mol/nm expressed in ATV force units.
inline constexpr double kForceFixedScale =
    4294967296.0 * kKjPerMolNmPerEvPerAngstrom;

/// @brief Inverse fixed-point force scale [eV/Angstrom].
inline constexpr double kForceFixedResolution = 1.0 / kForceFixedScale;

/// @brief Exclusive bound of a scaled contribution: 2^63.
inline constexpr double kFixedLimit = 9223372036854775808.0;

// ── Coordinates ───────────────────────────────────────────────────────

/// @brief Reconstruct a coordinate for integration.
/// @param primary Stored coordinate column [Angstrom].
/// @param correction Correction column, or null without correction.
/// @param i Row.
/// @return Coordinate [Angstrom] in integration precision.
__host__ __device__ __forceinline__ mixed_t
load_position(const pos_t* primary, const pos_t* correction, int i) {
    if constexpr (kPositionCorrection)
        return mixed_t(primary[i]) + mixed_t(correction[i]);
    else
        return mixed_t(primary[i]);
}

/// @brief Split an integration coordinate into its stored representation.
/// @param value Coordinate [Angstrom].
/// @param primary Receives the force-path value.
/// @param correction Receives the remainder (unchanged without correction).
__host__ __device__ __forceinline__ void
split_position(mixed_t value, pos_t& primary, pos_t& correction) {
    primary = pos_t(value);
    if constexpr (kPositionCorrection)
        correction = pos_t(value - mixed_t(primary));
}

/// @brief Store an integration coordinate into the stored columns.
/// @param primary Stored coordinate column [Angstrom].
/// @param correction Correction column, or null without correction.
/// @param i Row.
/// @param value Coordinate [Angstrom].
__host__ __device__ __forceinline__ void
store_position(pos_t* primary, pos_t* correction, int i, mixed_t value) {
    pos_t hi, lo = pos_t(0);
    split_position(value, hi, lo);
    primary[i] = hi;
    if constexpr (kPositionCorrection)
        correction[i] = lo;
}

// ── Fixed-point forces ────────────────────────────────────────────────

/// @brief Encode one force value; false when it cannot be represented.
/// @tparam T Precision the contribution was computed in.
/// @param value Force [eV/Angstrom].
/// @param encoded Fixed-point word, truncated toward zero.
/// @return False for NaN, Inf or |value * scale| >= 2^63.
template <typename T>
__host__ __device__ __forceinline__ bool encode_force(T value,
                                                      fixed_t& encoded) {
    const T scaled = value * T(kForceFixedScale);
    if (!(scaled < T(kFixedLimit) && scaled > -T(kFixedLimit)))
        return false;
    encoded = static_cast<fixed_t>(scaled);
    return true;
}

/// @brief Decode a fixed-point force.
/// @tparam T Consumer precision.
/// @param encoded Fixed-point word.
/// @return Force [eV/Angstrom].
template <typename T>
__host__ __device__ __forceinline__ T decode_force(fixed_t encoded) {
    return T(encoded) * T(kForceFixedResolution);
}

/// @brief Record a fault; the first recorded row wins.
/// @param fault Frame fault record (null: no record available).
/// @param kind `ForceFault::kConversion` or `kAccumulation`.
/// @param row Local row of the offending force.
__device__ __forceinline__ void raise_force_fault(ForceFault* fault,
                                                  unsigned kind, int row) {
    if (fault == nullptr)
        return;
    atomicCAS(&fault->row, -1, row);
    atomicOr(&fault->kinds, kind);
}

/// @brief Record a device reserve that a rebuild outgrew.
/// @param fault Frame fault record (null: ignored).
/// @param required Entries the rebuild needed.
__device__ __forceinline__ void raise_capacity_fault(ForceFault* fault,
                                                     long long required) {
    if (fault == nullptr)
        return;
    atomicMax(&fault->required, required);
    atomicOr(&fault->kinds, ForceFault::kCapacity);
}

/// @brief State-writer gate: true once any fault froze the device state.
///
/// Thermostats, barostats, box writers and momentum removal return early so
/// a run that faulted stays at its faulting step while the host has not yet
/// observed the fault. Uniform across the grid: faults are raised by earlier
/// kernels in stream order.
/// @param fault Frame fault record (null: never frozen).
__device__ __forceinline__ bool state_frozen(const ForceFault* fault) {
    return fault != nullptr && fault->kinds != 0;
}

/// @brief Check one returned fixed-point accumulation for wraparound.
///
/// Split from the atomic so a kernel can issue several atomics, do other
/// work, and only then consume the returned words.
/// @param old Word returned by the atomic add.
/// @param encoded Fixed-point contribution that was added.
/// @param fault Frame fault record.
/// @param i Row the contribution was added to.
__device__ __forceinline__ void check_fixed_sum(fixed_t old, fixed_t encoded,
                                                ForceFault* fault, int i) {
    using U = unsigned long long;
    const U o = static_cast<U>(old), add = static_cast<U>(encoded);
    const U sum = o + add;
    if (((o ^ sum) & (add ^ sum)) >> 63)
        raise_force_fault(fault, ForceFault::kAccumulation, i);
}

/// @brief Encode and atomically add a contribution; returns the previous
///        word for a later `check_fixed_sum`.
/// @tparam T Precision the contribution was computed in.
/// @param force Force column.
/// @param i Row.
/// @param value Contribution [eV/Angstrom].
/// @param fault Frame fault record (conversion faults are raised here).
/// @param encoded Receives the encoded contribution (0 on conversion fault).
/// @return Word held by the slot before the add.
template <typename T>
__device__ __forceinline__ fixed_t add_force_deferred(fixed_t* force, int i,
                                                      T value,
                                                      ForceFault* fault,
                                                      fixed_t& encoded) {
    using U = unsigned long long;
    if (!encode_force(value, encoded)) {
        raise_force_fault(fault, ForceFault::kConversion, i);
        encoded = 0;
    }
    return static_cast<fixed_t>(
        atomicAdd(reinterpret_cast<U*>(force + i), static_cast<U>(encoded)));
}

/// @brief A thread's fixed-point force adds whose wraparound check is
///        deferred until the kernel has done other work (typically its
///        block energy/virial reduction), hiding the atomic return latency.
///
/// Slots are addressed by compile-time constant indices so the record stays
/// in registers; a slot whose row is negative is skipped.
/// @tparam N Number of adds per thread.
template <int N>
struct DeferredForceAdds {
    fixed_t old[N];   ///< Words returned by the atomics.
    fixed_t added[N]; ///< Encoded contributions.
    int row[N];       ///< Destination rows, -1 for an unused slot.

    __device__ __forceinline__ DeferredForceAdds() {
#pragma unroll
        for (int k = 0; k < N; ++k)
            row[k] = -1;
    }

    /// @brief Encode and add a contribution into slot k.
    /// @tparam T Precision the contribution was computed in.
    /// @param k Slot (compile-time constant at the call site).
    /// @param force Force column.
    /// @param i Row (negative: skip).
    /// @param value Contribution [eV/Angstrom].
    /// @param fault Frame fault record.
    template <typename T>
    __device__ __forceinline__ void add(int k, fixed_t* force, int i, T value,
                                        ForceFault* fault) {
        if (i < 0)
            return;
        old[k] = add_force_deferred(force, i, value, fault, added[k]);
        row[k] = i;
    }

    /// @brief Check every recorded add for wraparound.
    /// @param fault Frame fault record.
    __device__ __forceinline__ void check(ForceFault* fault) const {
#pragma unroll
        for (int k = 0; k < N; ++k)
            if (row[k] >= 0)
                check_fixed_sum(old[k], added[k], fault, row[k]);
    }
};

/// @brief Add an encoded word to a force slot, detecting wraparound.
/// @param force Force column.
/// @param i Row.
/// @param encoded Fixed-point contribution.
/// @param fault Frame fault record.
__device__ __forceinline__ void add_fixed(fixed_t* force, int i,
                                          fixed_t encoded, ForceFault* fault) {
    using U = unsigned long long;
    const U old =
        atomicAdd(reinterpret_cast<U*>(force + i), static_cast<U>(encoded));
    check_fixed_sum(static_cast<fixed_t>(old), encoded, fault, i);
}

/// @brief Accumulate a force contribution into a fixed-point column.
/// @tparam T Precision the contribution was computed in.
/// @param force Force column.
/// @param i Row.
/// @param value Contribution [eV/Angstrom].
/// @param fault Frame fault record.
template <typename T>
__device__ __forceinline__ void add_force(fixed_t* force, int i, T value,
                                          ForceFault* fault) {
    fixed_t encoded;
    if (!encode_force(value, encoded)) {
        raise_force_fault(fault, ForceFault::kConversion, i);
        return;
    }
    add_fixed(force, i, encoded, fault);
}

/// @brief Add a force contribution to a slot owned by the calling thread.
///
/// Same encoding and fault semantics as `add_force`, without the atomic: for
/// per-atom fixes where exactly one thread touches row i.
/// @tparam T Precision the contribution was computed in.
/// @param force Force column.
/// @param i Row.
/// @param value Contribution [eV/Angstrom].
/// @param fault Frame fault record.
template <typename T>
__device__ __forceinline__ void add_force_owned(fixed_t* force, int i, T value,
                                                ForceFault* fault) {
    using U = unsigned long long;
    fixed_t encoded;
    if (!encode_force(value, encoded)) {
        raise_force_fault(fault, ForceFault::kConversion, i);
        return;
    }
    const U old = static_cast<U>(force[i]), add = static_cast<U>(encoded);
    const U sum = old + add;
    if (((old ^ sum) & (add ^ sum)) >> 63)
        raise_force_fault(fault, ForceFault::kAccumulation, i);
    force[i] = static_cast<fixed_t>(sum);
}

/// @brief Overwrite a force slot.
/// @tparam T Precision the value was computed in.
/// @param force Force column.
/// @param i Row.
/// @param value Force [eV/Angstrom].
/// @param fault Frame fault record.
template <typename T>
__device__ __forceinline__ void store_force(fixed_t* force, int i, T value,
                                            ForceFault* fault) {
    fixed_t encoded = 0;
    if (!encode_force(value, encoded))
        raise_force_fault(fault, ForceFault::kConversion, i);
    force[i] = encoded;
}

/// @brief Integrator gate: true when the Frame forces may be consumed.
///
/// On the first refusal, one thread stamps the step and the offending row's
/// global tag for the run-return report.
/// @param fault Frame fault record (null: always valid).
/// @param tag Global tag column of the same Frame.
/// @param step Current step index.
/// @return False when any fault kind is set.
__device__ __forceinline__ bool
forces_consumable(ForceFault* fault, const std::int64_t* tag, long long step) {
    if (fault == nullptr || fault->kinds == 0)
        return true;
    if (blockIdx.x == 0 && threadIdx.x == 0 && fault->step < 0) {
        fault->step = step;
        fault->tag = (fault->row >= 0 && tag != nullptr) ? tag[fault->row] : -1;
    }
    return false;
}

/// @brief Integrator gate reading the step index from device memory, so a
///        replayed graph stamps the step it actually ran (null: `step`).
__device__ __forceinline__ bool forces_consumable(ForceFault* fault,
                                                  const std::int64_t* tag,
                                                  const long long* step_ptr,
                                                  long long step) {
    if (fault == nullptr || fault->kinds == 0)
        return true;
    return forces_consumable(fault, tag, step_ptr ? *step_ptr : step);
}

} // namespace cuda
} // namespace ATV
