/// @file cpu/io/extxyz.h
/// @brief cpu::load_extxyz -- load an (ext)XYZ file into a cpu::Frame via molpy.
///
/// All XYZ/ExtXYZ parsing lives in molpy (`molrs::io::data::xyz`); this is a
/// thin C++ wrapper that bridges the parsed `molpy.Frame` into a `cpu::Frame`
/// through `cpu::from_molpy`. There is no hand-rolled C++ parser -- per the
/// project rule that all I/O goes through molpy.
///
/// @see cpu::from_molpy (cpu/frame/molpy_exchange.h)
/// @see molpy_cxx/bridge.h (molpy molcxx) for the molpy CXX FFI bridge
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
/// @throws std::runtime_error if `path` does not exist, is a directory, or
///         cannot be opened for reading; or if the file lacks the required
///         `Lattice="..."` header (no fallback box is substituted). The message
///         names the function and the offending path.
///          Parse failures *inside* the file (unknown element symbol, missing
///          `species` column, malformed atom table) also throw: molpy
///          `read_xyz` returns `Result`, cxx maps that to a
///          `rust::Error`, and this function rethrows it as a
///          `std::runtime_error` naming the path.
Frame load_extxyz(const std::string& path);

} // namespace cpu
} // namespace ATV
