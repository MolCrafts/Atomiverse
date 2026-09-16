/// @file python/src/bindings/owned_buffer.h
/// @brief The one allocate → copy → capsule → release primitive every numpy
///        array these bindings hand back is built from.
///
/// Nothing the engine holds may be aliased into Python: an engine buffer is
/// either reused by the next solve (`AM1Result::forces_ev_a`), overwritten by
/// the next `run()` (`HessianResult`), a temporary (`GradientResult::forces`),
/// or a live working frame that keeps evolving (`cpu::mc` columns). So every
/// array crossing the boundary is a *copy* whose buffer numpy owns: allocate,
/// copy element by element, attach an `nb::capsule` — nanobind's owner object,
/// a raw pointer plus the function that frees it, which numpy keeps alive as
/// the array's base — and release the temporary owner to it. The buffer's
/// lifetime then follows the array and nothing in the engine has to outlive
/// the call.
///
/// That four-step sequence is what lives here, once. It is deliberately below
/// `force_array.h` (the typed `(n_atoms, 3)` [eV/Å] force currency) and
/// `result_array.h` (dense 1-D/2-D double results): those headers carry the
/// semantics — element type, shape, units, size checks — and this one carries
/// none. It is also the only one of the three usable for a non-`double`
/// element, which is what `bind_mc.cpp`'s `uint32` type column needs.
#pragma once

#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <type_traits>

namespace nb = nanobind;

/// @brief Copy a contiguous source into a fresh buffer owned by a new numpy
///        array of the requested shape.
///
/// Elements are converted one by one with `static_cast`, so the source type
/// may differ from the array's element type (an engine `TagInt` column into a
/// `uint32` numpy column, say). A same-type copy is the identity conversion.
///
/// @tparam Array Concrete `nb::ndarray<nb::numpy, …>` type to produce; its
///         `Scalar` is the element type allocated and cast to.
/// @tparam Source Element type of the contiguous source being copied.
/// @param values Contiguous source elements, in the produced array's own
///        (row-major) order. An empty span yields an empty array.
/// @param shape Extents of the produced array; their product must equal
///        `values.size()`, which is the caller's job to guarantee — the
///        semantically typed wrappers check it with a message that names the
///        quantity.
/// @return An `Array` of `shape` owning its buffer; the buffer is released by
///         the array's capsule deleter, not by the caller.
template <typename Array, typename Source>
Array to_owned_array(std::span<const Source> values,
                     std::initializer_list<std::size_t> shape) {
    using Element = std::remove_const_t<typename Array::Scalar>;
    auto buffer = std::make_unique<Element[]>(values.size());
    std::transform(
        values.begin(), values.end(), buffer.get(),
        [](const Source& value) { return static_cast<Element>(value); });
    nb::capsule owner(buffer.get(), [](void* p) noexcept {
        delete[] static_cast<Element*>(p);
    });
    return Array(buffer.release(), shape, owner);
}
