/// @file src/cpu/io/extxyz.cpp
/// @brief Implementation of cpu::load_extxyz (ext)XYZ -> cpu::Frame via molpy.

#include "atomiverse/cpu/io/extxyz.h"
#include "atomiverse/cpu/units/frame.h"

#include <stdexcept>

#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/molpy.hpp"
#include "readable_path.h"

namespace ATV {
namespace cpu {

static Frame read_extxyz(const std::string& path, std::string_view declared) {
    // Pre-flight the common "wrong filename" case so the message is ours.
    internal::require_readable_file("cpu::load_extxyz", path);

    // molpy parses the atom table + Lattice header and writes Z into
    // atoms.atomic_number. A readable-but-corrupt file raises `rust::Error`;
    // the wrapper attaches the path, which the bridge message omits.
    auto fref = internal::with_molpy_context("cpu::load_extxyz", path, [&] {
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

    if (!declared.empty()) {
        const auto keys = molrs::frame_meta_keys(*fref);
        bool present = false;
        for (const auto& key : keys)
            if (key == "units_style")
                present = true;
        if (!present) {
            molrs::KeyedMetaValue entry{};
            entry.key = "units_style";
            entry.dtype = molrs::MetaType::String;
            entry.string_value = rust::String(std::string(declared));
            molrs::frame_set_meta(*fref, std::move(entry));
        }
    }
    return from_molpy(*fref);
}

Frame load_extxyz(const std::string& path) { return read_extxyz(path, {}); }

Frame load_extxyz(const std::string& path, std::string_view source_profile) {
    if (source_profile != "electronic")
        (void)units::parse_style(source_profile);
    auto frame = read_extxyz(path, source_profile);
    if (ATV::units::has_frame_units(frame)
        && ATV::units::frame_profile(frame) != source_profile)
        throw std::invalid_argument(
            "load_extxyz: source units conflict with file metadata");
    ATV::units::annotate(frame, source_profile);
    return frame;
}

} // namespace cpu
} // namespace ATV
