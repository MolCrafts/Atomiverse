/// @file cpu/scf/dense_matrix.h
/// @brief The shared dense matrix types (`cpu::linalg::RealMatrix`/`RealVector`)
///        made available as `cpu::scf::RealMatrix`/`RealVector`.
///
/// Single-purpose type-alias header: SCF code uses these matrix types pervasively
/// and unqualified. The operations on them live in `cpu/linalg/eigensolve.h` and
/// `cpu/linalg/density.h` (call those `linalg::` functions qualified).
#pragma once

#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace scf {

using linalg::RealMatrix;
using linalg::RealVector;

} // namespace scf
} // namespace cpu
} // namespace ATV
