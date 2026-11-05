/// @file cpu/record/xyz.cpp
/// @brief XYZ writer — delegating to molpy FFI, one frame at a time.

#include "atomiverse/cpu/record/xyz.h"
#include "../io/molpy_atoms_frame.h"
#include "../io/molpy_meta.h"
#include "../io/molpy_species.h"
#include "../io/readable_path.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/molpy.hpp"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace ATV {
namespace cpu {

namespace {
/// @brief Stopgap launch-boundary span resolution for a `const cpu::Frame&`.
///
/// Extract one frame's raw arrays (atomic number, x, y, z, cell matrix H) for
/// the molpy writer. `kAtom_type` carries Z (the engine keys species on the
/// atomic number; `from_molpy` fills it from `atoms.atomic_number`), so it
/// crosses as molpy' `atomic_number` unchanged.
void extract(const Frame& frame, std::vector<std::uint64_t>& atomic_number,
             std::vector<double>& x, std::vector<double>& y,
             std::vector<double>& z, std::array<double, 9>& box) {
    const int n = static_cast<int>(frame.nlocal());
    // Hoisted once before the loop -- the 13.7x rule.
    const int* __restrict sp_type = frame.span(kAtom_type);
    internal::molpy_atomic_numbers(sp_type, static_cast<std::size_t>(n),
                                   atomic_number, "cpu::XYZ");
    x.resize(n);
    y.resize(n);
    z.resize(n);
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>(sp_x[i]);
        y[i] = static_cast<double>(sp_y[i]);
        z[i] = static_cast<double>(sp_z[i]);
    }
    const auto& cell = frame.box.cell();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            box[static_cast<std::size_t>(i * 3 + j)] =
                static_cast<double>(cell(i, j));
}
} // namespace

// ── XYZ ──────────────────────────────────────────────────────────────────────

XYZ::XYZ(const std::string& filename) : filename_(filename) {}

void XYZ::write(const Frame& frame) { write_impl(frame, /*append=*/false); }
void XYZ::append(const Frame& frame) { write_impl(frame, /*append=*/true); }

void XYZ::write_impl(const Frame& frame, bool append) {
    std::array<double, 9> box{};
    extract(frame, atomic_number_, x_, y_, z_, box);
    // The bridge message does not name the file; with_molpy_context does.
    internal::with_molpy_context("cpu::XYZ", filename_, [&] {
        // No periodic cell (det == 0) crosses as no box, as in to_molpy.
        auto out = internal::molpy_atoms_frame(
            atomic_number_, x_, y_, z_,
            {box.data(), frame.box.volume() > 0.0 ? box.size() : 0});
        for (auto& entry : internal::encode_molpy_meta(frame.meta))
            molrs::frame_set_meta(*out, std::move(entry));
        molrs::write_xyz({filename_.data(), filename_.size()}, *out, append);
    });
}

} // namespace cpu
} // namespace ATV
