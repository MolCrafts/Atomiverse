/// @file cpu/io/frame_writer.h
/// @brief Per-frame trajectory writers — raw arrays in, delegating to molpy.
///
/// The engine's raw per-atom arrays are put into one molpy exchange frame
/// (`atoms.{atomic_number, x, y, z}` + cell), which the molpy writer takes
/// whole (XYZ or `*.mrec`). No C++ record/accumulator type.
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
/// Delegates to molpy `write_xyz`. @param atomic_number per-atom atomic
/// number Z, the engine's `kAtom_atomic_number` (0..118; 0 is the wildcard `*`); molpy
/// spells the symbols explicitly (`Element::fill_symbols`). @param x,y,z
/// per-atom coordinates [Å]. @param box_h row-major 3×3 cell matrix H [Å];
/// all zeros means no cell, any other singular H is an error. @param append
/// `false` truncates/creates the file, `true` appends the frame.
/// @throws std::invalid_argument on a negative atomic number;
/// std::runtime_error (molpy, path attached) on Z > 118, a length mismatch
/// or a singular H.
void write_xyz_frame(const std::string& path,
                     const std::vector<int>& atomic_number,
                     const std::vector<double>& x, const std::vector<double>& y,
                     const std::vector<double>& z,
                     const std::array<double, 9>& box_h, bool append);

/// @brief Write one frame plus named per-atom fields as a `*.mrec` record
///        whose `frame` section is that frame (molpy' Zarr-v3 scientific
///        record).
///
/// Delegates to molpy `write_mrec_frame`; read it back with molpy
/// `read_mrec_frame`. The checkpoint door: one frame, whole, replaced on every
/// call — for a streaming trajectory use cpu::MrecRecorder. The record stores
/// species as `atoms.atomic_number` (no `element` symbols).
/// @param path store path; must not use the retired `.zarr` suffix (molpy
/// refuses it), conventionally `*.mrec`. @param atomic_number per-atom atomic
/// number Z, the engine's `kAtom_atomic_number` (0..118; 0 is the wildcard).
/// @param x,y,z per-atom coordinates [Å]. @param field_names one name per
/// extra per-atom column; unique, and none of `atomic_number`/`x`/`y`/`z`.
/// @param field_data row-major `[n_fields, n_atoms]`, matching `field_names`
/// order. @param box_h row-major 3×3 cell matrix H [Å]; all zeros means no
/// cell, any other singular H is an error.
/// @throws std::invalid_argument on a negative atomic number, a field_data
/// length other than n_fields × n_atoms, or a reserved / repeated field name;
/// std::runtime_error (molpy, path attached) on Z > 118, a coordinate count
/// other than the atom count or a singular H.
void write_mrec_frame(const std::string& path,
                      const std::vector<int>& atomic_number,
                      const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<double>& z,
                      const std::array<double, 9>& box_h,
                      const std::vector<std::string>& field_names,
                      const std::vector<double>& field_data);

} // namespace cpu
} // namespace ATV
