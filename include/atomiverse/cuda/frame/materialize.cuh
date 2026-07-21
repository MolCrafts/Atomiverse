/// @file cuda/frame/materialize.cuh
/// @brief cuda::materialize / cuda::write_back -- molrs.Frame <-> cuda::Frame.
///
/// `molrs.Frame` is the canonical exchange layer; `cuda::Frame` is the CUDA
/// backend's device-resident execution frame. The materializer is shared
/// infrastructure (sibling of `cuda/transfer.cuh` at the backend root, per
/// the CLAUDE.md backend-layering rule) and is meant to run once per
/// `run()` -- never inside a step loop.
///
/// Structurally symmetric with `cpu/materialize`: `materialize()` reads the
/// fixed molrs field set into a transient host frame and performs an
/// explicit H2D transfer onto the device; `write_back()` performs an
/// explicit D2H transfer and pushes the results back into the molrs.Frame.
/// The molrs column reading/writing is reused from `cpu/materialize` so the
/// block/column-name contract stays in one place.
///
/// @see cpu::materialize / cpu::write_back for the host-side bridge
/// @see cuda::Frame for the CUDA execution frame
/// @see molrs.hpp for the molrs CXX FFI bridge
//
// Contract inheritance: cuda::write_back == cuda::to_cpu into a transient
// host frame + cpu::write_back, so it inherits the cpu round-trip field
// contract (see cpu/frame/materialize.h). to_cpu's returned field set
// (x/v/f/tag/charge/image/scalars/box) covers every contract-written
// field; the transient host frame's type/mass are zero-valued, but the
// contract does not write those back, so no garbage reaches molrs.
#pragma once

#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cuda {

/// @brief Materialize a molrs.Frame into a new device-resident cuda::Frame.
///
/// Reads the molrs.Frame into a transient `cpu::Frame` (via
/// `cpu::materialize`, which fixes the required/optional field contract),
/// then performs an explicit H2D transfer (via `cpu::to_cuda`). Required
/// fields (atom positions, `type`) fail loud with a named exception;
/// optional fields default to zero / empty.
///
/// @param src molrs.Frame to read (introspected by block/column name).
/// @return Newly allocated cuda::Frame with all available fields on device.
/// @throws std::runtime_error if a required field is missing.
Frame materialize(const molrs::FrameRef& src);

/// @brief Write a cuda::Frame's results back into an existing molrs.Frame.
///
/// Performs an explicit D2H transfer (via `cuda::to_cpu`) into a transient
/// `cpu::Frame`, then delegates to `cpu::write_back`. Overwrites the
/// per-atom result columns (positions, velocities, forces, charge) and the
/// system-scalar blocks; topology and box are left untouched.
///
/// @param src cuda::Frame whose device state is exported.
/// @param dst molrs.Frame to update in place (compatible atom count).
void write_back(const Frame& src, molrs::FrameRef& dst);

} // namespace cuda
} // namespace ATV
