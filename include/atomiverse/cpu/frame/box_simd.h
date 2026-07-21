/// @file cpu/frame/box_simd.h
/// @brief xsimd batch overload for `cpu::Box::min_image_batch`.
///
/// Split out from `cpu/box.h` so that CUDA translation units and other
/// xsimd-unaware consumers can include `cpu/box.h` without pulling in
/// `<xsimd/xsimd.hpp>`. Include this header from CPU TUs that need the
/// batched min-image (currently only `src/cpu/md/lj.cpp`).
///
/// The body uses `xsimd::select` to mirror the scalar branchy form
/// exactly, preserving ULP-equivalent results between the SIMD ket-loop
/// and the scalar tail in `LJ::launch`.
#pragma once

#include <xsimd/xsimd.hpp>

#include "atomiverse/cpu/frame/box.h"

namespace ATV {
namespace cpu {

template <class B>
inline void Box::min_image_batch(B& bdx, B& bdy, B& bdz) const noexcept {
    assert(ortho_ && "min_image_batch is orthorhombic-only");
    const B v_lx(cell_(0, 0)), v_ly(cell_(1, 1)), v_lz(cell_(2, 2));
    const B v_hlx(half_lx_), v_hly(half_ly_), v_hlz(half_lz_);
    const B v_nhlx(-half_lx_), v_nhly(-half_ly_), v_nhlz(-half_lz_);

    bdx = xsimd::select(bdx > v_hlx, bdx - v_lx, bdx);
    bdx = xsimd::select(bdx < v_nhlx, bdx + v_lx, bdx);
    bdy = xsimd::select(bdy > v_hly, bdy - v_ly, bdy);
    bdy = xsimd::select(bdy < v_nhly, bdy + v_ly, bdy);
    bdz = xsimd::select(bdz > v_hlz, bdz - v_lz, bdz);
    bdz = xsimd::select(bdz < v_nhlz, bdz + v_lz, bdz);
}

} // namespace cpu
} // namespace ATV
