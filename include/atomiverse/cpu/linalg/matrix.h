/// @file cpu/linalg/matrix.h
/// @brief Shared dense matrix/vector types for AO-basis electronic structure.
///
/// Column-major for LAPACK compatibility via xtensor-blas. Used by both
/// `cpu::scf` and `cpu::gpw`; the operations on these types live in the sibling
/// `eigensolve.h` / `density.h`.
#pragma once

#include <complex>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace linalg {

using RealMatrix = xt::xtensor<double, 2, xt::layout_type::column_major>;
using RealVector = xt::xtensor<double, 1>;
/// @brief Complex dense matrix (default row-major) — GPW k-point Bloch MOs /
///        k-space Hamiltonians. Layout matches `gpw::internal::CplxMatrix`.
using CplxMatrix = xt::xtensor<std::complex<double>, 2>;

} // namespace linalg
} // namespace cpu
} // namespace ATV
