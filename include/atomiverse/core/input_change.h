/// @file core/input_change.h
/// @brief Changes to caller-owned inputs between continuable run segments.
#pragma once

namespace ATV {
enum class InputChange : unsigned {
    None = 0,
    Geometry = 1u << 0,
    Velocity = 1u << 1,
    Charge = 1u << 2,
    Cell = 1u << 3,
    Identity =
        1u << 4, ///< Population, species, tags, masses, topology or schema.
};
constexpr InputChange operator|(InputChange a, InputChange b) noexcept {
    return static_cast<InputChange>(static_cast<unsigned>(a)
                                    | static_cast<unsigned>(b));
}
constexpr InputChange& operator|=(InputChange& a, InputChange b) noexcept {
    return a = a | b;
}
constexpr bool has_input_change(InputChange changes, InputChange bit) noexcept {
    return (static_cast<unsigned>(changes) & static_cast<unsigned>(bit)) != 0;
}
} // namespace ATV
