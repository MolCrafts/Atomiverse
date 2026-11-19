/// @file core/quantity.h
/// Unit algebra at preparation boundaries. No registry or strings in kernels.
#pragma once
#include "atomiverse/core/unit_style.h"
#include <array>
#include <stdexcept>

namespace ATV::units {
// L, M, T, charge, temperature. Charge is an independent base dimension;
// Gaussian electrical units are mapped explicitly by the run profile.
struct Dimension {
    std::array<int, 5> powers{};
    constexpr bool operator==(const Dimension&) const = default;
};
constexpr Dimension operator*(Dimension a, Dimension b) {
    for (int i = 0; i < 5; ++i)
        a.powers[i] += b.powers[i];
    return a;
}
constexpr Dimension operator/(Dimension a, Dimension b) {
    for (int i = 0; i < 5; ++i)
        a.powers[i] -= b.powers[i];
    return a;
}
inline constexpr Dimension length_dimension{{1, 0, 0, 0, 0}};
inline constexpr Dimension mass_dimension{{0, 1, 0, 0, 0}};
inline constexpr Dimension time_dimension{{0, 0, 1, 0, 0}};
inline constexpr Dimension charge_dimension{{0, 0, 0, 1, 0}};
inline constexpr Dimension temperature_dimension{{0, 0, 0, 0, 1}};

/// Resolved unit, relative to a run's coherent basis. The domain prevents
/// accidentally bridging reduced LJ quantities to physical quantities.
struct Unit {
    Dimension dimension;
    double scale = 1;
    int domain = 0;
};
constexpr Unit operator*(Unit a, Unit b) {
    if (a.domain != b.domain)
        throw std::invalid_argument("units: incompatible domains");
    return {a.dimension * b.dimension, a.scale * b.scale, a.domain};
}
constexpr Unit operator/(Unit a, Unit b) {
    if (a.domain != b.domain)
        throw std::invalid_argument("units: incompatible domains");
    return {a.dimension / b.dimension, a.scale / b.scale, a.domain};
}
constexpr double convert(double value, const Unit& from, const Unit& to) {
    if (from.dimension != to.dimension || from.domain != to.domain)
        throw std::invalid_argument("units: incompatible quantity conversion");
    return value * (from.scale / to.scale);
}
/// Static units are checked at compile time. No dimension descriptors or
/// exception paths exist in the generated numerical code.
template <Unit From, Unit To>
constexpr double convert(double value) noexcept {
    static_assert(From.dimension == To.dimension,
                  "units: incompatible dimensions");
    static_assert(From.domain == To.domain, "units: incompatible domains");
    return value * (From.scale / To.scale);
}
/// A prepared linear conversion: exactly one scalar, no runtime unit tags.
class Conversion {
    double scale_ = 1.;

  public:
    constexpr Conversion() = default;
    constexpr Conversion(const Unit& source, const Unit& target)
        : scale_(convert(1., source, target)) {}
    constexpr double operator()(double value) const noexcept {
        return value * scale_;
    }
    constexpr double inverse(double value) const noexcept {
        return value / scale_;
    }
    /// For already-compiled lower-level numeric kernel argument blocks.
    constexpr double scale() const noexcept { return scale_; }
};
static_assert(sizeof(Conversion) == sizeof(double));
struct Quantity {
    double value;
    Unit unit;
    constexpr double numerical_value_in(Unit target) const {
        return convert(value, unit, target);
    }
    constexpr Quantity to(Unit target) const {
        return {numerical_value_in(target), target};
    }
};
constexpr Quantity operator*(Quantity a, Quantity b) {
    return {a.value * b.value, a.unit * b.unit};
}
constexpr Quantity operator/(Quantity a, Quantity b) {
    return {a.value / b.value, a.unit / b.unit};
}
constexpr Quantity operator+(Quantity a, Quantity b) {
    return {a.value + b.numerical_value_in(a.unit), a.unit};
}
constexpr Quantity operator-(Quantity a, Quantity b) {
    return {a.value - b.numerical_value_in(a.unit), a.unit};
}

} // namespace ATV::units
