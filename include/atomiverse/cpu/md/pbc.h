/// @file cpu/md/pbc.h
/// @brief Orthorhombic periodic-boundary wrap for owned atoms.
///
/// Integrators (VelocityVerlet, NVELimit, ...) call `wrap_orthorhombic`
/// after advancing positions so atoms stay in `[0, L_d)` per axis. Ghost
/// rows are intentionally untouched -- the next `HaloSend`
/// rebuild will overwrite them with freshly packed neighbor data.
#pragma once

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Wrap owned-atom positions into the primary periodic image.
///
/// Orthorhombic only: uses the diagonal `box(d, d)` entries. Atoms at
/// exactly `x == L` are wrapped to `0`. Ghost rows `[n_owned, n)` are
/// left alone; they are owned by the halo exchange.
///
/// @param frame Frame whose `atoms.x/y/z` are shifted in place.
/// @param n_owned Number of owned atoms to wrap (typically `owned_count(ctx)`).
inline void wrap_orthorhombic(cpu::Frame& frame, int n_owned) {
    const double lx = frame.simbox.length(0);
    const double ly = frame.simbox.length(1);
    const double lz = frame.simbox.length(2);
    StorageReal* __restrict sp_x = frame.span(kAtom_x);
    StorageReal* __restrict sp_y = frame.span(kAtom_y);
    StorageReal* __restrict sp_z = frame.span(kAtom_z);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_owned; ++i) {
        sp_x[i] -= lx * std::floor(sp_x[i] / lx);
        sp_y[i] -= ly * std::floor(sp_y[i] / ly);
        sp_z[i] -= lz * std::floor(sp_z[i] / lz);
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
