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

#include <string>

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
/// @brief The `FrameRef` PyCapsule name this interpreter's molrs produces.
///
/// molrs 0.14 put the ABI line in the capsule name —
/// `"molrs.FrameRef/<major.minor>"` — so a cross-minor handle exchange fails
/// the name check instead of dereferencing a drifted layout. Builds on 0.13
/// and earlier used the bare `"molrs.FrameRef"`, and hardcoding that is what
/// broke every Python compute entry against a 0.14 molpy.
///
/// molrs's own contract says to query the name rather than spell it
/// (`molrs-cxxapi/src/lib.rs`, `frame_clone_from_addr` safety note), so that
/// is what this does: `molrs._ffi_abi_token()[2]`. Importing molrs here is
/// engine-internal — the "never import molrs from user code" rule in the
/// error message below is guidance for callers, not for this bridge.
///
/// Cached: the answer cannot change inside one interpreter.
/// @return The exact capsule name to hand `PyCapsule_GetPointer`.
/// @throws std::runtime_error when molrs is absent or the token is malformed.
inline const std::string& molrs_frameref_capsule_name() {
    static const std::string name = [] {
        nb::object token =
            nb::module_::import_("molrs").attr("_ffi_abi_token")();
        if (nb::len(token) < 3) {
            throw std::runtime_error(
                "molrs._ffi_abi_token() returned fewer than 3 fields; this "
                "molrs is too old for the Atomiverse Python bridge");
        }
        return nb::cast<std::string>(token[2]);
    }();
    return name;
}

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
    const std::string& expected = molrs_frameref_capsule_name();
    void* ptr = PyCapsule_GetPointer(cap, expected.c_str());
    if (ptr == nullptr) {
        // PyCapsule_GetPointer set a Python error (wrong name / NULL). The
        // name carries molrs's ABI line, so the usual cause is a frame
        // produced by a different molrs minor than the one in this process.
        PyErr_Clear();
        const char* actual = PyCapsule_GetName(cap);
        throw std::runtime_error(
            std::string("molrs.Frame capsule is named \"")
            + (actual ? actual : "(null)") + "\" but this build expects \""
            + expected
            + "\". That is a molrs ABI-line mismatch between the frame's "
              "producer and this interpreter's molrs — rebuild them against "
              "one molrs.");
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    return molrs::frame_clone_from_addr(static_cast<std::size_t>(addr));
}
