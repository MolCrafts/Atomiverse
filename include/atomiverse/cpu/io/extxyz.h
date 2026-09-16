/// @file cpu/io/extxyz.h
/// @brief cpu::load_extxyz -- load an (ext)XYZ file into a cpu::Frame via molrs.
///
/// All XYZ/ExtXYZ parsing lives in molrs (`molrs::io::data::xyz`); this is a
/// thin C++ wrapper that bridges the parsed `molrs.Frame` into a `cpu::Frame`
/// through `cpu::from_molrs`. There is no hand-rolled C++ parser -- per the
/// project rule that all I/O goes through molrs.
///
/// @see cpu::from_molrs (cpu/frame/molrs_exchange.h)
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
/// @throws std::runtime_error if `path` does not exist, is a directory, or
///         cannot be opened for reading; or if the file lacks the required
///         `Lattice="..."` header (no fallback box is substituted). The message
///         names the function and the offending path.
///          Parse failures *inside* the file (unknown element symbol, missing
///          `species` column, malformed atom table) also throw as of molrs
///          0.13, which made `xyz_read_first_frame` return `Result`; cxx maps
///          that to a `rust::Error`, and this function rethrows it as a
///          `std::runtime_error` naming the path. Earlier molrs releases
///          aborted the process on those inputs and could not be caught.
Frame load_extxyz(const std::string& path);

} // namespace cpu
} // namespace ATV
