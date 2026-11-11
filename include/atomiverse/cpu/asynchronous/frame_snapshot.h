/// @file cpu/asynchronous/frame_snapshot.h
/// @brief FrameSnapshot -- a consumer's read-only view of one captured frame
///        in the driver's shared `cpu::CaptureBuffer`.
///
/// The spans point into the buffer slot, an independent copy of the frame
/// taken once for every consumer due on that step; they stay valid until the
/// consumer's `on_snapshot` returns. A column no consumer of the buffer asked
/// for is empty.
#pragma once

#include <cstdint>
#include <span>

#include "atomiverse/core/capture_buffer.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/types.h"
#include "atomiverse/cpu/frame/box.h"

namespace ATV {
namespace cpu {

/// @brief Read-only view of one captured frame (locally owned atoms).
struct FrameSnapshot {
    /// Step label [completed steps / iterations] (LAMMPS `ntimestep`).
    BigInt step = 0;
    CaptureKind kind = CaptureKind::step; ///< Why it was captured.
    int n_owned = 0;                      ///< Atoms captured.
    cpu::Box box{}; ///< Simulation cell at the captured step [Å].

    std::span<const double> x, y, z;    ///< Positions [Å].
    std::span<const double> vx, vy, vz; ///< Velocities [Å/fs].
    std::span<const double> fx, fy, fz; ///< Forces [eV/Å].
    std::span<const double> charge;     ///< Charges [e].
    std::span<const double> mass;       ///< Masses [amu].
    std::span<const int> type;          ///< Force-field type indices.
    std::span<const std::int64_t> tag;  ///< Global atom tags.
    std::span<const ImageInt> image;    ///< Image flags, row-major 3 × n.
};

} // namespace cpu
} // namespace ATV
