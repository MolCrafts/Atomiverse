/// @file core/capture_buffer.h
/// @brief The shared asynchronous frame-capture buffer: compile-time byte
///        budget, column set, capture kinds and accounting. CUDA-safe,
///        header-only.
///
/// Every asynchronous consumer of a driver — trajectory and thermo
/// recorders, online analysis observers — reads its steps from ONE buffer
/// per driver (`cpu::CaptureBuffer` / `cuda::CaptureBuffer`). A step any
/// consumer needs is captured once, as an independent copy, and every
/// consumer due on that step reads the same immutable copy; the slot is
/// freed when the last of them releases it. Capture is lossless: when the
/// buffer is full the stepping thread waits for space, nothing is dropped.
///
/// The budget bounds the bytes of every capture in flight across all
/// consumers. Its default is the CMake cache variable
/// `ATV_CAPTURE_BUFFER_BYTES` (256 MiB); a driver overrides it at run time
/// (`set_capture_buffer_bytes`, Python `MD(capture_buffer_bytes=…)`).
#pragma once

#include <cstdint>

#ifndef ATV_CAPTURE_BUFFER_BYTES
#error "ATV_CAPTURE_BUFFER_BYTES is defined by CMake (cache variable of the " \
       "same name); build against the atomiverse target"
#endif

namespace ATV {

/// @brief Default capture buffer budget [bytes] (`ATV_CAPTURE_BUFFER_BYTES`).
inline constexpr std::uint64_t kCaptureBufferBytes =
    static_cast<std::uint64_t>(ATV_CAPTURE_BUFFER_BYTES);

/// @brief Per-atom columns a consumer reads from a capture. The buffer
///        captures the union over its consumers; the cell, the system
///        scalars and the step label are always captured.
///
/// Positions, velocities, forces, charges and masses are captured in FP64
/// (a CUDA position is primary + correction, a force is decoded from fixed
/// point). On CUDA the identity columns (`types`, species) are static for a
/// run and copied once when the buffer is allocated, not per capture.
struct CaptureColumns {
    bool positions = false;  ///< x, y, z [Å].
    bool velocities = false; ///< vx, vy, vz [Å/fs].
    bool forces = false;     ///< fx, fy, fz [eV/Å].
    bool charges = false;    ///< q [e].
    bool masses = false;     ///< m [amu].
    bool types = false;      ///< Force-field type index.
    bool tags = false;       ///< Global atom tag (int64).
    bool images = false;     ///< Periodic image flags (3 × int32).

    /// @brief Union of two column sets.
    constexpr CaptureColumns operator|(const CaptureColumns& o) const {
        return {positions || o.positions, velocities || o.velocities,
                forces || o.forces,       charges || o.charges,
                masses || o.masses,       types || o.types,
                tags || o.tags,           images || o.images};
    }
    constexpr bool operator==(const CaptureColumns&) const = default;
};

/// @brief Why a capture was taken.
enum class CaptureKind : std::uint8_t {
    setup, ///< Start of a run (LAMMPS `Output::setup`): the current state.
    step,  ///< End of a step on a consumer's cadence (LAMMPS `Output::write`).
    final_state, ///< A minimization stopped early (LAMMPS `Min::run`).
};

/// @brief Accounting of one driver's capture buffer.
struct CaptureStats {
    std::uint64_t budget_bytes = 0;     ///< Buffer budget [bytes].
    std::uint64_t frame_bytes = 0;      ///< One captured frame [bytes].
    std::uint64_t capacity = 0;         ///< Frames the budget holds [count].
    std::uint64_t captures = 0;         ///< Frames captured [count].
    std::uint64_t peak_bytes = 0;       ///< Peak occupancy [bytes].
    std::uint64_t blocked_episodes = 0; ///< Times the run waited for space.
    double blocked_seconds = 0.0;       ///< Total time the run waited [s].
};

} // namespace ATV
