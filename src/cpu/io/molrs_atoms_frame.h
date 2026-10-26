/// @file src/cpu/io/molrs_atoms_frame.h
/// @brief The one frame a molrs single-frame writer is handed: a fresh molrs
/// exchange frame holding `atoms.{atomic_number, x, y, z}` and the cell.
///
/// The molrs bridge has one frame currency — `write_xyz(path, frame, append)`
/// and `write_mrec_frame(path, frame)` both take a `molrs::FrameRef` — so the
/// engine's raw per-atom arrays are put into one here, and a caller adds what
/// else it writes (typed meta, extra per-atom columns) before handing it over.
/// Internal to `src/cpu/` (same status as `molrs_species.h`) — not a public
/// header.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "atomiverse/molpy.hpp"
#include "rust/cxx.h"

namespace ATV::cpu::internal {

/// @brief Build the exchange frame of one frame of atoms.
///
/// Every call crosses the bridge and may throw `rust::Error` — a column whose
/// length differs from the first one (molrs refuses a ragged block), a cell
/// that is not 9 values or is singular — so run it inside
/// `with_molrs_context` to name the operation and path.
///
/// @param atomic_number per-atom Z as molrs' `atomic_number` (see
/// `molrs_atomic_numbers`). @param x,y,z per-atom coordinates [Å].
/// @param h row-major 3×3 cell matrix H [Å], or empty for no cell.
/// @return the frame, owned by the caller.
inline ::rust::Box<molrs::FrameRef>
molrs_atoms_frame(const std::vector<std::uint64_t>& atomic_number,
                  const std::vector<double>& x, const std::vector<double>& y,
                  const std::vector<double>& z, ::rust::Slice<const double> h) {
    auto frame = molrs::frame_new();
    molrs::frame_set_column_u64(*frame, "atoms", "atomic_number",
                                {atomic_number.data(), atomic_number.size()});
    molrs::frame_set_column_f64(*frame, "atoms", "x", {x.data(), x.size()});
    molrs::frame_set_column_f64(*frame, "atoms", "y", {y.data(), y.size()});
    molrs::frame_set_column_f64(*frame, "atoms", "z", {z.data(), z.size()});
    if (h.size() != 0)
        molrs::frame_set_box(
            *frame, h,
            ::rust::Slice<const double>(
                std::array<double, 3>{0.0, 0.0, 0.0}.data(), 3),
            ::rust::Slice<const bool>(
                std::array<bool, 3>{true, true, true}.data(), 3));
    return frame;
}

} // namespace ATV::cpu::internal
