/// @file cpu/io/frame_writer.cpp
/// @brief Per-frame trajectory writers — thin marshaling to molrs FFI.

#include "atomiverse/cpu/io/frame_writer.h"
#include "atomiverse/molrs.hpp"

#include <cstdint>

namespace ATV {
namespace cpu {

void write_frame_xyz(const std::string& path, const std::vector<int>& type_id,
                     const std::vector<double>& x, const std::vector<double>& y,
                     const std::vector<double>& z,
                     const std::array<double, 9>& box_mat, bool append) {
    molrs::write_frame_xyz(
        {path.data(), path.size()},
        {reinterpret_cast<const std::int32_t*>(type_id.data()), type_id.size()},
        {x.data(), x.size()}, {y.data(), y.size()}, {z.data(), z.size()},
        {box_mat.data(), box_mat.size()}, append);
}

void write_frame_zarr(const std::string& path, const std::vector<int>& type_id,
                      const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<double>& z,
                      const std::array<double, 9>& box_mat,
                      const std::vector<std::string>& field_names,
                      const std::vector<double>& field_data) {
    rust::Vec<rust::String> names;
    for (const auto& n : field_names)
        names.push_back(rust::String(n));
    molrs::write_frame_zarr(
        {path.data(), path.size()},
        {reinterpret_cast<const std::int32_t*>(type_id.data()), type_id.size()},
        {x.data(), x.size()}, {y.data(), y.size()}, {z.data(), z.size()},
        {box_mat.data(), box_mat.size()}, std::move(names),
        {field_data.data(), field_data.size()});
}

} // namespace cpu
} // namespace ATV
