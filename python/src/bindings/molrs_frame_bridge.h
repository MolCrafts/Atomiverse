/// @file python/molrs_frame_bridge.h
/// @brief Cross-extension bridge: borrow a molpy.Frame from a Python object.
///
/// Atomiverse does not bind a Frame data type. The **user-facing** exchange
/// layer is `molpy.Frame`. molpy re-exports the molrs schema object, so the
/// capsule name is `"molrs.FrameRef/<major.minor>"` — an engine-private FFI
/// detail, not a user import. Users never `import molrs`.
///
/// `borrow_molrs_frame()` extracts that capsule, hands the raw address to
/// the molrs-cxxapi CXX bridge (`molrs::frame_clone_from_addr`), and returns
/// a `rust::Box<molrs::FrameRef>` borrowing the *same* shared frame. Writes
/// through that handle are visible in the originating Python `molpy.Frame`.
#pragma once

#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <stdexcept>

#include "atomiverse/molrs.hpp"

namespace nb = nanobind;

/// @brief The runtime capsule name, after checking the linked molrs ABI.
///
/// Query molrs's own token for the name and compare its ABI line against the
/// molrs-ffi source selected by ATV_MOLRS_PATH at build time. Comparing only
/// the capsule and runtime package would accept a different embedded layout.
///
/// Cached: the answer cannot change inside one interpreter.
/// @return The exact capsule name to hand `PyCapsule_GetPointer`.
/// @throws std::runtime_error when the token is malformed or its ABI differs.
inline const std::string& molrs_frameref_capsule_name() {
    static const std::string name = [] {
        nb::object token =
            nb::module_::import_("molrs").attr("_ffi_abi_token")();
        if (nb::len(token) < 3) {
            throw std::runtime_error(
                "molrs._ffi_abi_token() returned fewer than 3 fields; this "
                "molrs is too old for the Atomiverse Python bridge");
        }
        const std::string runtime_abi = nb::cast<std::string>(token[0]);
        if (runtime_abi != ATV_MOLRS_ABI_LINE) {
            throw std::runtime_error(
                "molrs ABI-line mismatch: runtime molrs is " + runtime_abi
                + " but Atomiverse is linked against " + ATV_MOLRS_ABI_LINE
                + ". Rebuild and install both wheels from one molrs source.");
        }
        return nb::cast<std::string>(token[2]);
    }();
    return name;
}

/// @brief Borrow a molpy.Frame handle after validating the molrs ABI.
/// @param frame_obj A Python molpy.Frame instance.
/// @return A handle sharing the originating frame's store.
/// @throws std::runtime_error on an ABI mismatch or malformed frame capsule.
inline ::rust::Box<molrs::FrameRef> borrow_molrs_frame(nb::object frame_obj) {
    if (!nb::hasattr(frame_obj, "_ffi_frameref_capsule")) {
        throw std::runtime_error(
            "expected a molpy.Frame: object has no _ffi_frameref_capsule(). "
            "Pass a molpy.Frame (from the molpy package). Never import molrs "
            "from user code.");
    }
    const std::string& expected = molrs_frameref_capsule_name();
    nb::object cap_obj = frame_obj.attr("_ffi_frameref_capsule")();
    PyObject* cap = cap_obj.ptr();
    if (!PyCapsule_CheckExact(cap)) {
        throw std::runtime_error(
            "molrs.Frame._ffi_frameref_capsule() did not return a PyCapsule");
    }
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
