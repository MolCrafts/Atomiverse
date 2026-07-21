/// @file cpu/async/frame_snapshot.h
/// @brief POD snapshot of a `cpu::Frame` for off-hot-path analysis consumers.
///
/// The snapshot is a contiguous host-allocated copy of a subset of frame
/// fields, selected via a `CaptureMask`. Snapshots live in a triple-buffered
/// ring held by `AnalysisExecutor`; each slot is populated by
/// `Publish` at `end_of_step @ +10` and consumed by
/// `AsyncComponentBase::on_snapshot` on a worker thread.
///
/// @see .claude/specs/parallel-md-engine.md  § B.1
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/box.h"

namespace ATV {
namespace cpu {

/// @brief Selects which fields of `cpu::Frame::Atoms` a snapshot should carry.
///
/// `Publish` copies only the fields flagged true, and clears
/// (i.e. resizes to zero) the rest. Consumers reading unflagged fields see
/// an empty vector — a simple invariant easy to assert in tests.
struct CaptureMask {
    bool positions = true;   ///< x, y, z.
    bool velocities = false; ///< vx, vy, vz.
    bool forces = false;     ///< fx, fy, fz.
    bool types = false;      ///< atom type.
    bool tags = false;       ///< global tag.
};

/// @brief Contiguous host-side copy of the locally-owned portion of a `cpu::Frame`.
///
/// Buffers are `std::vector<>`-backed for trivial copy/move semantics and
/// bounded heap growth. All vectors are sized at `build()` time to the
/// frame's initial capacity; `Publish` only writes, never reallocates.
struct FrameSnapshot {
    int step = 0;    ///< Simulation step at which this snapshot was published.
    int n_owned = 0; ///< Number of owned atoms captured.
    cpu::Box box{};  ///< Simulation cell at the time of capture [Angstrom].

    std::vector<double> x, y, z;    ///< Positions [Angstrom].
    std::vector<double> vx, vy, vz; ///< Velocities [Angstrom/fs].
    std::vector<double> fx, fy, fz; ///< Forces [eV/Angstrom].
    std::vector<int> type;          ///< Atom type indices.
    std::vector<std::int64_t> tag;  ///< Global atom tags.

    /// @brief Optional opaque payload for domain-specific captures (e.g.
    /// density matrix for Mulliken, orbital coefficients for cube writes).
    std::vector<std::byte> custom_payload;
};

} // namespace cpu
} // namespace ATV
