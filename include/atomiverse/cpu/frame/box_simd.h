/// @file cpu/frame/box_simd.h
/// @brief xsimd batch overload for `cpu::Box::min_image_batch`.
///
/// Split out from `cpu/frame/box.h` so that xsimd-unaware consumers can use
/// Box without pulling in `<xsimd/xsimd.hpp>`.
///
/// Uses the same nearest-integer reduction as scalar Box::min_image,
/// including displacements spanning multiple cell lengths.
#pragma once

#include <xsimd/xsimd.hpp>

#include "atomiverse/cpu/frame/box.h"

namespace ATV {
namespace cpu {

template <class B>
inline void Box::min_image_batch(B& bdx, B& bdy, B& bdz) const noexcept {
    assert(ortho_ && "min_image_batch is orthorhombic-only");
    const B v_lx(cell_(0, 0)), v_ly(cell_(1, 1)), v_lz(cell_(2, 2));
    if (cell_(0, 0) > 0)
        bdx -= v_lx * xsimd::nearbyint(bdx / v_lx);
    if (cell_(1, 1) > 0)
        bdy -= v_ly * xsimd::nearbyint(bdy / v_ly);
    if (cell_(2, 2) > 0)
        bdz -= v_lz * xsimd::nearbyint(bdz / v_lz);
}

} // namespace cpu
} // namespace ATV
