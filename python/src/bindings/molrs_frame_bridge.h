/// @file python/molrs_frame_bridge.h
/// @brief Cross-extension bridge: borrow a molpy.Frame from a Python object.
///
/// Atomiverse does not bind a Frame data type. The **user-facing** exchange
/// layer is `molpy.Frame`. molpy re-exports the molrs schema object, so the
/// capsule name stays `"molrs.FrameRef"` — that is an engine-private FFI
/// detail, not a user import. Users never `import molrs`.
///
/// `borrow_molrs_frame()` extracts that capsule, hands the raw address to
/// the molrs-cxxapi CXX bridge (`molrs::frame_clone_from_addr`), and returns
/// a `rust::Box<molrs::FrameRef>` borrowing the *same* shared frame. Writes
/// through that handle are visible in the originating Python `molpy.Frame`.
#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <stdexcept>

#include "atomiverse/molrs.hpp"

namespace nb = nanobind;

/// @brief Borrow the molpy.Frame behind a Python object as a cxx handle.
///
/// @param frame_obj A Python `molpy.Frame` instance.
/// @return A `rust::Box<molrs::FrameRef>` sharing the same molrs Store as
///         the Python frame — reads and writes round-trip to it.
/// @throws std::runtime_error if `frame_obj` is not a molpy.Frame or its
///         capsule is malformed.
inline ::rust::Box<molrs::FrameRef> borrow_molrs_frame(nb::object frame_obj) {
    if (!nb::hasattr(frame_obj, "_ffi_frameref_capsule")) {
        throw std::runtime_error(
            "expected a molpy.Frame: object has no _ffi_frameref_capsule(). "
            "Pass a molpy.Frame (from the molpy package). Never import molrs "
            "from user code.");
    }
    nb::object cap_obj = frame_obj.attr("_ffi_frameref_capsule")();
    PyObject* cap = cap_obj.ptr();
    if (!PyCapsule_CheckExact(cap)) {
        throw std::runtime_error(
            "molrs.Frame._ffi_frameref_capsule() did not return a PyCapsule");
    }
    void* ptr = PyCapsule_GetPointer(cap, "molrs.FrameRef");
    if (ptr == nullptr) {
        // PyCapsule_GetPointer set a Python error (wrong name / NULL).
        throw std::runtime_error(
            "molrs.Frame capsule is not named \"molrs.FrameRef\" or is NULL");
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    return molrs::frame_clone_from_addr(static_cast<std::size_t>(addr));
}
