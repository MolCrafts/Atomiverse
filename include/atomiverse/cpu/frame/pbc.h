/// @file cpu/frame/pbc.h
/// @brief Periodic-boundary wrap for owned atoms.
///
/// Runtime calls `wrap_owned` before neighbor/communication rebuilds
/// so owned atoms lie in the primary fractional cell. Ghost rows are untouched;
/// `CommBrick` rebuilds them with the selected periodic image shifts.
#pragma once

#include <cstddef>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Wrap owned Cartesian positions into the primary periodic cell and
///        count each lattice translation into the image flags.
///
/// LAMMPS `Domain::pbc`: a row moved by \(-H\mathbf n\) gains \(\mathbf n\)
/// in `kAtom_image`, so the unwrapped position \(\mathbf r + H\,\mathrm{image}\)
/// is unchanged by the wrap (unwrapped-coordinate consumers: `FixEfield`
/// energy, transport). The bookkeeping is `Box::wrap_into`'s, exact for any
/// cell shape. Ghost rows retain their halo-selected images. Runtime invokes
/// this only at a communication/neighbor rebuild, never during ordinary
/// integration.
/// @param frame   Frame whose owned rows are wrapped.
/// @param n_owned Owned row count.
inline void wrap_owned(cpu::Frame& frame, int n_owned) {
    StorageReal* __restrict sp_x = frame.span(kAtom_x);
    StorageReal* __restrict sp_y = frame.span(kAtom_y);
    StorageReal* __restrict sp_z = frame.span(kAtom_z);
    ImageInt* __restrict image = frame.span(kAtom_image).data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_owned; ++i) {
        double x = sp_x[i], y = sp_y[i], z = sp_z[i];
        frame.box.wrap_into(x, y, z, image + 3 * static_cast<std::size_t>(i));
        sp_x[i] = x;
        sp_y[i] = y;
        sp_z[i] = z;
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
