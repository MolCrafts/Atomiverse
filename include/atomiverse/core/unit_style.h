/// @file core/unit_style.h
/// Run unit-system identities and their input spelling. No conversion data.
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
namespace ATV::units {
/// @brief LAMMPS `units` style. The byte value is the task-wire encoding.
enum class Style : std::uint8_t {
    lj = 0,
    real = 1,
    metal = 2,
    si = 3,
    cgs = 4,
    electron = 5,
    micro = 6,
    nano = 7,
};

/// Number of styles (wire range check).
inline constexpr int kStyleCount = 8;

/// LAMMPS spelling of each style, indexed by `Style`.
inline constexpr std::array<const char*, kStyleCount> kStyleNames = {
    "lj", "real", "metal", "si", "cgs", "electron", "micro", "nano"};

/// @brief LAMMPS name of `s` (e.g. `"real"`).
[[nodiscard]] constexpr const char* style_name(Style s) {
    return kStyleNames[static_cast<std::uint8_t>(s)];
}

/// @brief Style named `name`, spelled as in the LAMMPS `units` command.
/// @throws std::invalid_argument on an unknown name.
[[nodiscard]] inline constexpr Style parse_style(std::string_view name) {
    for (int i = 0; i < kStyleCount; ++i)
        if (name == kStyleNames[i])
            return static_cast<Style>(i);
    throw std::invalid_argument(
        "units: unknown LAMMPS style '" + std::string(name)
        + "' (lj, real, metal, si, cgs, electron, micro, nano)");
}

} // namespace ATV::units
