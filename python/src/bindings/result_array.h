/// @file python/src/bindings/result_array.h
/// @brief Owned numpy arrays for dense double result data (1-D and 2-D).
///
/// A driver result object (`cpu::vib::HessianResult`, …) publishes dense
/// `xt::xtensor<double, N>` members that Python wants as numpy arrays. The
/// engine's storage cannot be aliased: the result the binding hands back is a
/// copy the caller may outlive the driver with, and the driver's own cached
/// copy is overwritten by the next `run()`. Both functions are therefore thin
/// shape wrappers over the shared `to_owned_array()` primitive
/// (`owned_buffer.h`), which allocates, copies and attaches an `nb::capsule`
/// deleter — a capsule is nanobind's owner object, a raw pointer plus the
/// function that frees it, which numpy keeps alive as the array's base. The
/// buffer's lifetime therefore follows the numpy array and nothing in the
/// engine has to stay alive.
///
/// Sibling of `force_array.h`, not a replacement for it: that header is the
/// one *typed* `(n_atoms, 3)` [eV/Å] force currency, with the mis-sized-buffer
/// check that only makes sense for forces. These two functions carry no
/// physical meaning at all — they are the shape-preserving copy every other
/// dense double result needs.
#pragma once

#include "owned_buffer.h"

#include <nanobind/ndarray.h>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

/// @brief A fresh, numpy-owned 1-D double array.
using VectorArray = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

/// @brief A fresh, numpy-owned 2-D (row-major) double array.
using MatrixArray = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

/// @brief Copy a contiguous double buffer into a fresh numpy-owned 1-D array.
///
/// @param values Contiguous source values; an empty span yields an empty
///        `(0,)` array.
/// @return A 1-D numpy array of length `values.size()` owning its buffer.
inline VectorArray to_vector_array(std::span<const double> values) {
    return to_owned_array<VectorArray>(values, {values.size()});
}

/// @brief Copy a contiguous row-major double buffer into a fresh numpy-owned
///        2-D array.
///
/// @param values Contiguous source values in row-major order (one whole row
///        after another; the layout of an `xt::xtensor<double, 2>` and of a
///        default numpy array) of length exactly `rows * cols`.
/// @param rows Number of rows of the produced array.
/// @param cols Number of columns of the produced array.
/// @return A `(rows, cols)` numpy array owning its buffer.
/// @throws std::invalid_argument (Python `ValueError`) when the span length is
///         not `rows * cols` — a mis-sized matrix is never a shape to trim to.
inline MatrixArray to_matrix_array(std::span<const double> values,
                                   std::size_t rows, std::size_t cols) {
    if (values.size() != rows * cols)
        throw std::invalid_argument(
            "matrix buffer length " + std::to_string(values.size())
            + " does not match the requested shape (" + std::to_string(rows)
            + ", " + std::to_string(cols) + ")");
    return to_owned_array<MatrixArray>(values, {rows, cols});
}
