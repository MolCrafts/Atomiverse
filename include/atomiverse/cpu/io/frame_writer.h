/// @file cpu/io/frame_writer.h
/// @brief Per-frame trajectory writers — raw arrays in, delegating to molrs.
///
/// The engine hands molrs one frame's worth of raw per-atom arrays; molrs builds
/// the `Frame` and writes it (XYZ or Zarr). No C++ record/accumulator type.
///
/// CUDA-safe: only `std::vector`/`std::array`/`std::string` (no `rust/cxx.h`).
#pragma once

#include <array>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {

/// @brief Write one frame to a standard-XYZ file (element + coordinates).
///
/// Delegates to molrs `write_frame_xyz`. @param type_id per-atom atomic number Z
/// (element symbol derived in molrs). @param x,y,z per-atom coordinates [Å].
/// @param box_mat row-major 3×3 cell [Å]. @param append `false` truncates/creates
/// the file, `true` appends the frame.
void write_frame_xyz(const std::string& path, const std::vector<int>& type_id,
                     const std::vector<double>& x, const std::vector<double>& y,
                     const std::vector<double>& z,
                     const std::array<double, 9>& box_mat, bool append);

/// @brief Write one frame plus named per-atom fields to a single-frame Zarr store.
///
/// Delegates to molrs `write_frame_zarr`. @param field_names one name per extra
/// per-atom column. @param field_data row-major `[n_fields, n_atoms]`, matching
/// `field_names` order. @param box_mat row-major 3×3 cell [Å].
void write_frame_zarr(const std::string& path, const std::vector<int>& type_id,
                      const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<double>& z,
                      const std::array<double, 9>& box_mat,
                      const std::vector<std::string>& field_names,
                      const std::vector<double>& field_data);

} // namespace cpu
} // namespace ATV
