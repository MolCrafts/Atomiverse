/// @file cpu/communication/gather_for_io.h
/// @brief `gather_for_root_io` -- resolve the frame a rank-0 recorder writes.
///
/// Shared preamble of the root-writing recorders (TrajectoryWriter,
/// FinalGeometry): under multi-rank every rank joins the collective owner
/// gather (`gather_to_root`, tag-ordered) and only rank 0 materializes the
/// gathered result; single-rank runs write the local frame directly.
///
/// @see cpu/comm/gather.h
#pragma once

namespace ATV {
namespace cpu {

class Frame;
struct Context;

namespace md {
namespace mpi {

/// @brief Resolve which frame a root-writing recorder should persist.
///
/// Collective under multi-rank: every rank must call this in lockstep
/// (it enters `gather_to_root`) *before* any rank gate.
///
/// @param frame    Local host-resident frame (owner rows first) [Angstrom].
/// @param runtime  CPU execution context (MPI rank/size + decomposition).
/// @param gathered Caller-owned scratch frame; overwritten with the
///                 tag-ordered gathered geometry on rank 0 under multi-rank,
///                 untouched otherwise. Must outlive the returned pointer.
/// @return Frame to write: `&frame` on single-rank runs, `&gathered` on
///         rank 0 under multi-rank, `nullptr` on non-root ranks (skip I/O).
const cpu::Frame* gather_for_root_io(const cpu::Frame& frame,
                                     const cpu::Context& runtime,
                                     cpu::Frame& gathered);

} // namespace mpi
} // namespace md
} // namespace cpu
} // namespace ATV
