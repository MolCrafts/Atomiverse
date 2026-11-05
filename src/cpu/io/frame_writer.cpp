/// @file cpu/io/frame_writer.cpp
/// @brief Per-frame trajectory writers — the engine's arrays become one molpy
/// exchange frame, which the molpy writer takes whole.

#include "atomiverse/cpu/io/frame_writer.h"
#include "atomiverse/molpy.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "molpy_atoms_frame.h"
#include "molpy_species.h"
#include "readable_path.h"

namespace ATV {
namespace cpu {

namespace {
/// The cell matrix H as molpy takes it: all nine values, or none when the
/// engine has no cell (an all-zero H, a default-constructed Box). A non-zero
/// singular H is still sent — molpy refuses it rather than dropping the box.
::rust::Slice<const double> molpy_h(const std::array<double, 9>& box_h) {
    for (const double v : box_h)
        if (v != 0.0)
            return {box_h.data(), box_h.size()};
    return {box_h.data(), 0};
}
} // namespace

void write_xyz_frame(const std::string& path,
                     const std::vector<int>& atomic_number,
                     const std::vector<double>& x, const std::vector<double>& y,
                     const std::vector<double>& z,
                     const std::array<double, 9>& box_h, bool append) {
    std::vector<std::uint64_t> z_number;
    internal::molpy_atomic_numbers(atomic_number.data(), atomic_number.size(),
                                   z_number, "cpu::write_xyz_frame");
    // The bridge message does not name the file; with_molpy_context does.
    internal::with_molpy_context("cpu::write_xyz_frame", path, [&] {
        const auto frame =
            internal::molpy_atoms_frame(z_number, x, y, z, molpy_h(box_h));
        molrs::write_xyz({path.data(), path.size()}, *frame, append);
    });
}

void write_mrec_frame(const std::string& path,
                      const std::vector<int>& atomic_number,
                      const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<double>& z,
                      const std::array<double, 9>& box_h,
                      const std::vector<std::string>& field_names,
                      const std::vector<double>& field_data) {
    constexpr const char* who = "cpu::write_mrec_frame";
    const std::size_t n = atomic_number.size();
    if (field_data.size() != field_names.size() * n)
        throw std::invalid_argument(
            std::string(who) + ": " + std::to_string(field_data.size())
            + " field values for " + std::to_string(field_names.size())
            + " fields of " + std::to_string(n)
            + " atoms; field_data is [n_fields, n_atoms]");
    for (std::size_t f = 0; f < field_names.size(); ++f) {
        const std::string& name = field_names[f];
        if (name == "atomic_number" || name == "x" || name == "y"
            || name == "z")
            throw std::invalid_argument(std::string(who) + ": field '" + name
                                        + "' would overwrite the frame's own "
                                          "atomic_number/x/y/z column");
        for (std::size_t g = 0; g < f; ++g)
            if (field_names[g] == name)
                throw std::invalid_argument(std::string(who) + ": field '"
                                            + name + "' is named twice");
    }
    std::vector<std::uint64_t> z_number;
    internal::molpy_atomic_numbers(atomic_number.data(), n, z_number, who);
    internal::with_molpy_context(who, path, [&] {
        auto frame =
            internal::molpy_atoms_frame(z_number, x, y, z, molpy_h(box_h));
        for (std::size_t f = 0; f < field_names.size(); ++f) {
            const std::string& name = field_names[f];
            molrs::frame_set_column_f64(*frame, "atoms",
                                        {name.data(), name.size()},
                                        {field_data.data() + f * n, n});
        }
        molrs::write_mrec_frame({path.data(), path.size()}, *frame);
    });
}

} // namespace cpu
} // namespace ATV
