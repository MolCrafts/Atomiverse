/// @file cuda/frame/transfer.cuh
/// @brief cuda::to_cpu() / cuda::destroy() -- CUDA device transfer functions.
///
/// CUDA-free declarations (implementations in src/cuda/frame/transfer.cu).
///
/// @see cuda::Frame for the CUDA device data
/// @see cpu::Frame for the CPU device data
/// @see cpu::to_cuda() for the reverse direction
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/frame/frame.cuh"

namespace ATV {
namespace cuda {

/// @brief Transfer CUDA Frame back to CPU Frame (D2H copy, no deallocation).
///
/// Copies positions, velocities, forces, energy, kinetic energy, stress,
/// and box from device to host.
///
/// @param src CUDA Frame to read from.
/// @param dst CPU Frame to write into (must be pre-sized).
void to_cpu(const Frame& src, cpu::Frame& dst);

/// @brief Free all device memory in a CUDA Frame.
///
/// Frees all per-atom, system-scalar, box, and topology device pointers.
/// The SCF workspace and neighbor pair list are no longer frame-owned --
/// their owning components free them. cudaFree(nullptr) is safe, so
/// unallocated fields are handled. Resets the Frame to zero after freeing.
///
/// @param frame CUDA Frame to free (zeroed after this call).
void destroy(Frame& frame);

/// @brief Allocate + fill the multi-GPU per-atom bonded SoA on `d` from the
/// owner-filtered per-atom topology in `src` (partners stored as GLOBAL TAGS).
///
/// Used only by the multi-GPU `MultiGpu::partition` path: the single-GPU /
/// standard `cpu::to_cuda` leaves these arrays null (the single-GPU bonded
/// kernels read the flat BondList/AngleList/DihedralList instead). Arrays are
/// sized to `d.capacity_` (ghost headroom) and zeroed; the first
/// `src.nlocal()` rows are copied from `src`. The device frame must already
/// be materialized (so `d.capacity_` is set). Caps mirror `cpu::Atom`.
/// `destroy()` frees these (null-safe), so no separate teardown is needed.
void materialize_per_atom_bonded(Frame& d, const cpu::Frame& src);

} // namespace cuda
} // namespace ATV
