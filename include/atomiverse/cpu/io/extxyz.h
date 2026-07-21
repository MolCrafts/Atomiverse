/// @file cpu/io/extxyz.h
/// @brief cpu::load_extxyz -- load an (ext)XYZ file into a cpu::Frame via molrs.
///
/// All XYZ/ExtXYZ parsing lives in molrs (`molrs::io::data::xyz`); this is a
/// thin C++ wrapper that bridges the parsed `molrs.Frame` into a `cpu::Frame`
/// through `cpu::materialize`. There is no hand-rolled C++ parser -- per the
/// project rule that all I/O goes through molrs.
///
/// @see cpu::materialize (cpu/frame/materialize.h)
/// @see molrs.hpp for the molrs CXX FFI bridge
#pragma once

#include <string>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

/// @brief Load the first frame of an (ext)XYZ file into a new cpu::Frame.
///
/// Reads the atom table (positions and the required ExtXYZ `species` column,
/// consumed into per-atom `type` = atomic number Z) and the simulation box from
/// the extended-XYZ `Lattice="..."` header. Only the first frame of a
/// multi-frame file is read.
///
/// @param path Filesystem path to the (ext)XYZ file.
/// @return Newly constructed cpu::Frame with atom positions, `type` (Z), and
///         the periodic box filled.
/// @throws std::runtime_error if the file cannot be read, contains an unknown
///         element symbol, or lacks the required `Lattice=` header (no fallback
///         box is substituted).
Frame load_extxyz(const std::string& path);

} // namespace cpu
} // namespace ATV
