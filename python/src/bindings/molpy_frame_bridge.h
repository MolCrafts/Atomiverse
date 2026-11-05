/// @file python/molpy_frame_bridge.h
/// @brief Cross-extension bridge: borrow a molpy.Frame from a Python object.
///
/// Atomiverse does not bind a Frame data type. The **user-facing** exchange
/// layer is `molpy.Frame`. Capsule names are obtained from the merged
/// molpy native extension and validated by its molcxx bridge.
///
/// `borrow_molpy_frame()` extracts that capsule, hands the raw address to
/// the molcxx CXX bridge (`molrs::frame_clone_from_addr`), and returns
/// a `rust::Box<molrs::FrameRef>` borrowing the *same* shared frame. Writes
/// through that handle are visible in the originating Python `molpy.Frame`.
#pragma once

#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <stdexcept>

#include "atomiverse/molpy.hpp"

namespace nb = nanobind;

/// @brief The runtime capsule name, after checking the linked molpy ABI.
///
/// Query molpy's own token for the name and compare its ABI line against the
/// molrs-ffi source selected by ATV_MOLPY_PATH at build time. Comparing only
/// the capsule and runtime package would accept a different embedded layout.
///
/// Cached: the answer cannot change inside one interpreter.
/// @return The exact capsule name to hand `PyCapsule_GetPointer`.
/// @throws std::runtime_error when the token is malformed or its ABI differs.
inline const std::string& molpy_frameref_capsule_name() {
    static const std::string name = [] {
        nb::object token =
            nb::module_::import_("molpy._core").attr("_ffi_abi_token")();
        if (nb::len(token) < 3) {
            throw std::runtime_error(
                "molpy._core._ffi_abi_token() returned fewer than 3 fields; "
                "this "
                "molpy is too old for the Atomiverse Python bridge");
        }
        const std::string runtime_abi = nb::cast<std::string>(token[0]);
        if (runtime_abi != ATV_MOLPY_ABI_LINE) {
            throw std::runtime_error(
                "molpy ABI-line mismatch: runtime molpy is " + runtime_abi
                + " but Atomiverse is linked against " + ATV_MOLPY_ABI_LINE
                + ". Rebuild and install both wheels from one molpy source.");
        }
        const std::string runtime_name = nb::cast<std::string>(token[2]);
        const std::string linked_name(molrs::frameref_capsule_name());
        if (runtime_name != linked_name) {
            throw std::runtime_error(
                "molpy native build mismatch: the Frame capsule differs from "
                "Atomiverse's linked molcxx bridge. Rebuild both wheels from "
                "the same local molpy checkout and Rust toolchain.");
        }
        return linked_name;
    }();
    return name;
}

/// @brief Borrow a molpy.Frame handle after validating the molpy ABI.
/// @param frame_obj A Python molpy.Frame instance.
/// @return A handle sharing the originating frame's store.
/// @throws std::runtime_error on an ABI mismatch or malformed frame capsule.
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
        // name carries molpy's ABI line, so the usual cause is a frame
        // produced by a different molpy minor than the one in this process.
        PyErr_Clear();
        const char* actual = PyCapsule_GetName(cap);
        throw std::runtime_error(
            std::string("molpy.Frame capsule is named \"")
            + (actual ? actual : "(null)") + "\" but this build expects \""
            + expected
            + "\". That is a molpy ABI-line mismatch between the frame's "
              "producer and this interpreter's molpy — rebuild them against "
              "one molpy.");
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    return molrs::frame_clone_from_addr(static_cast<std::size_t>(addr));
}
