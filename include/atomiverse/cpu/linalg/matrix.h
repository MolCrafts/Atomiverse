/// @file cpu/linalg/matrix.h
/// @brief Shared dense matrix/vector types for AO-basis electronic structure.
///
/// Column-major for LAPACK compatibility via xtensor-blas. Used by both
/// `cpu::scf` and `cpu::gpw`; the operations on these types live in the sibling
/// `eigensolve.h` / `density.h`.
#pragma once

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace linalg {

using RealMatrix = xt::xtensor<double, 2, xt::layout_type::column_major>;
using RealVector = xt::xtensor<double, 1>;

} // namespace linalg
} // namespace cpu
} // namespace ATV
