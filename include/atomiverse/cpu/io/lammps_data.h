/// @file cpu/io/lammps_data.h
/// @brief LAMMPS data-file loader for atom_style full and molecular.
///
/// Self-contained C++ reader (does not go through molpy auto-detection).
/// Emits a fully populated `cpu::Frame`: positions, masses, types, optional
/// partial charges and velocities, bonds/angles/dihedrals/impropers,
/// orthogonal box, and `build_special_list()` 1-2/1-3/1-4 partners.
///
/// Supported Atoms layouts (auto-detected from the first Atoms data row,
/// optional trailing image flags accepted):
/// - **full**: `id mol-id type q x y z` — writes `kAtom_charge` [e]
/// - **molecular**: `id mol-id type x y z` — charges left at zero
///
/// `* Type Labels` and `* Coeffs` sections are skipped on purpose: numeric
/// type ids in the rows are authoritative, and parameters come from the
/// force-field components (e.g. `load_lammps_coeffs`), never from the data
/// file. Any other unknown section, a triclinic tilt, or a section with more
/// or fewer rows than the header declares throws.
///
/// Indices in the returned Frame are 0-indexed (LAMMPS files are 1-indexed).
#pragma once

#include <string>

namespace ATV {
namespace cpu {

class Frame;

/// @brief Read a LAMMPS data file (`atom_style full` or `molecular`).
///
/// @param path Filesystem path to the LAMMPS data file.
/// @return A populated `cpu::Frame` with topology and (for full) charges.
/// @throws std::runtime_error if the file cannot be opened, the header is
///         inconsistent, a section is unsupported, or a row is malformed.
Frame load_lammps_data(const std::string& path);

} // namespace cpu
} // namespace ATV
