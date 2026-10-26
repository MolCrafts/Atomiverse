/// @file src/cpu/io/extxyz.cpp
/// @brief Implementation of cpu::load_extxyz (ext)XYZ -> cpu::Frame via molrs.

#include "atomiverse/cpu/io/extxyz.h"

#include <stdexcept>

#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/molpy.hpp"
#include "readable_path.h"

namespace ATV {
namespace cpu {

Frame load_extxyz(const std::string& path) {
    // Pre-flight the common "wrong filename" case so the message is ours.
    internal::require_readable_file("cpu::load_extxyz", path);

    // molrs parses the atom table + Lattice header and writes Z into
    // atoms.atomic_number. A readable-but-corrupt file raises `rust::Error`;
    // the wrapper attaches the path, which the bridge message omits.
    auto fref = internal::with_molrs_context("cpu::load_extxyz", path, [&] {
        return molrs::read_xyz({path.data(), path.size()});
    });

    // The periodic box is required: a missing `Lattice=` header leaves no
    // box, and materialize treats the box as optional. Fail loud rather
    // than silently substituting a degenerate/identity box (no fallback).
    if (!molrs::frame_has_box(*fref)
        || molrs::frame_box(*fref).h.size() != 9u) {
        throw std::runtime_error("cpu::load_extxyz(" + path
                                 + "): ExtXYZ is missing the required "
                                   "Lattice=\"...\" header (no box)");
    }

    return from_molpy(*fref);
}

} // namespace cpu
} // namespace ATV
