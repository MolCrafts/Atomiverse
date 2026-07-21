/// @file src/cpu/io/extxyz.cpp
/// @brief Implementation of cpu::load_extxyz (ext)XYZ -> cpu::Frame via molrs.

#include "atomiverse/cpu/io/extxyz.h"

#include <stdexcept>

#include "atomiverse/cpu/frame/materialize.h"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {

Frame load_extxyz(const std::string& path) {
    // molrs parses the atom table + Lattice header and tags atoms.type with Z.
    auto fref = molrs::xyz_read_first_frame({path.data(), path.size()});

    // The periodic box is required: a missing `Lattice=` header leaves no
    // simbox, and materialize treats the box as optional. Fail loud rather
    // than silently substituting a degenerate/identity box (no fallback).
    if (molrs::frame_simbox(*fref).size() != 9u) {
        throw std::runtime_error("cpu::load_extxyz(" + path
                                 + "): ExtXYZ is missing the required "
                                   "Lattice=\"...\" header (no box)");
    }

    return materialize(*fref);
}

} // namespace cpu
} // namespace ATV
