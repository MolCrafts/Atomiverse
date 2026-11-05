/// @file python/molpy_frame_bridge.h
/// @brief Cross-extension bridge: borrow a molpy.Frame from a Python object.
///
/// Atomiverse does not bind a Frame data type. The **user-facing** exchange
/// layer is `molpy.Frame`. The capsule name is the one the linked molcxx
/// bridge accepts (`molrs::frameref_capsule_name()`), so a frame from a molpy
/// whose handle layout differs fails the name check before any dereference.
///
/// `borrow_molpy_frame()` extracts that capsule, hands the raw address to
/// the molcxx CXX bridge (`molrs::frame_clone_from_addr`), and returns
/// a `rust::Box<molrs::FrameRef>` borrowing the *same* shared frame. Writes
/// through that handle are visible in the originating Python `molpy.Frame`.
#pragma once

#include <string>

#include <nanobind/nanobind.h>

#include <cstdint>
#include <stdexcept>

#include "atomiverse/molpy.hpp"

namespace nb = nanobind;

/// @brief The capsule name the linked molcxx bridge accepts.
///
/// Cached: the answer cannot change inside one process.
/// @return The exact capsule name to hand `PyCapsule_GetPointer`.
inline const std::string& molpy_frameref_capsule_name() {
    static const std::string name(molrs::frameref_capsule_name());
    return name;
}

/// @brief Borrow a molpy.Frame handle after checking its capsule name.
/// @param frame_obj A Python molpy.Frame instance.
/// @return A handle sharing the originating frame's store.
/// @throws std::runtime_error on a malformed or foreign frame capsule.
inline ::rust::Box<molrs::FrameRef> borrow_molpy_frame(nb::object frame_obj) {
    if (!nb::hasattr(frame_obj, "_ffi_frameref_capsule")) {
        throw std::runtime_error(
            "expected a molpy.Frame: object has no _ffi_frameref_capsule(). "
            "Pass a molpy.Frame from the molpy package.");
    }
    const std::string& expected = molpy_frameref_capsule_name();
    nb::object cap_obj = frame_obj.attr("_ffi_frameref_capsule")();
    PyObject* cap = cap_obj.ptr();
    if (!PyCapsule_CheckExact(cap)) {
        throw std::runtime_error(
            "molpy.Frame._ffi_frameref_capsule() did not return a PyCapsule");
    }
    void* ptr = PyCapsule_GetPointer(cap, expected.c_str());
    if (ptr == nullptr) {
        // PyCapsule_GetPointer set a Python error (wrong name / NULL). The
        // name carries the producer's handle-layout fingerprint, so the usual
        // cause is a molpy built separately from Atomiverse's linked molcxx.
        PyErr_Clear();
        const char* actual = PyCapsule_GetName(cap);
        throw std::runtime_error(
            std::string("molpy.Frame capsule is named \"")
            + (actual ? actual : "(null)") + "\" but this build expects \""
            + expected
            + "\". Rebuild and install both wheels from the same local "
              "molpy checkout and Rust toolchain.");
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    return molrs::frame_clone_from_addr(static_cast<std::size_t>(addr));
}
