/// @file cuda/frame/molrs_exchange.cuh
/// @brief cuda::from_molrs / cuda::to_molrs -- molrs.Frame <-> cuda::Frame.
///
/// `molrs.Frame` is the canonical exchange layer; `cuda::Frame` is the CUDA
/// backend's device-resident execution frame. The materializer is shared
/// infrastructure (sibling of `cuda/transfer.cuh` at the backend root, per
/// the CLAUDE.md backend-layering rule) and is meant to run once per
/// `run()` -- never inside a step loop.
///
/// Structurally symmetric with `cpu/materialize`: `from_molrs()` reads the
/// fixed molrs field set into a transient host frame and performs an
/// explicit H2D transfer onto the device; `to_molrs()` performs an
/// explicit D2H transfer and pushes the results back into the molrs.Frame.
/// The molrs column reading/writing is reused from `cpu/materialize` so the
/// block/column-name contract stays in one place.
///
/// @see cpu::from_molrs / cpu::to_molrs for the host-side bridge
/// @see cuda::Frame for the CUDA execution frame
/// @see molrs.hpp for the molrs CXX FFI bridge
//
// Contract inheritance: cuda::to_molrs == cuda::to_cpu into a transient
// host frame + cpu::to_molrs, so it inherits the cpu round-trip field
// contract (see cpu/frame/molrs_exchange.h). to_cpu's returned field set
// (x/v/f/tag/charge/image/scalars/box) covers every contract-written
// field; the transient host frame's type/mass are zero-valued, but the
// contract does not write those back, so no garbage reaches molrs.
#pragma once

#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cuda {

/// @brief Build a .Frame into a new device-resident cuda::Frame.
///
/// Reads the molrs.Frame into a transient `cpu::Frame` (via
/// `cpu::from_molrs`, which fixes the required/optional field contract),
/// then performs an explicit H2D transfer (via `cpu::to_cuda`). Required
/// fields (atom positions, `type`) fail loud with a named exception;
/// optional fields default to zero / empty.
///
/// @param src molrs.Frame to read (introspected by block/column name).
/// @return Newly allocated cuda::Frame with all available fields on device.
/// @throws std::runtime_error if a required field is missing.
Frame from_molrs(const molrs::FrameRef& src);

/// @brief Export a cuda::Frame into an existing molrs.Frame.
///
/// Performs an explicit D2H transfer (via `cuda::to_cpu`) into a transient
/// `cpu::Frame`, then delegates to `cpu::to_molrs`. Overwrites the
/// per-atom result columns (positions, velocities, forces, charge) and the
/// system-scalar blocks; topology and box are left untouched.
///
/// @param src cuda::Frame whose device state is exported.
/// @param dst molrs.Frame to update in place (compatible atom count).
void to_molrs(const Frame& src, molrs::FrameRef& dst);

} // namespace cuda
} // namespace ATV
