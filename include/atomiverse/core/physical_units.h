/// @file core/physical_units.h
/// Physical boundary profiles. SI scales are independent of the historical
/// equation conversions resolved once for classical kernels.
#pragma once
#include "atomiverse/core/unit_constants.h"
#include <cmath>
#include <optional>
#include <string>
#include <string_view>

namespace ATV::units {
struct ReducedReference {
    std::string identity;
    double mass_kg;
    double length_m;
    double energy_j;
    void validate() const {
        if (identity.empty() || !std::isfinite(mass_kg) || mass_kg <= 0
            || !std::isfinite(length_m) || length_m <= 0
            || !std::isfinite(energy_j) || energy_j <= 0)
            throw std::invalid_argument(
                "units: LJ reference needs an identity and positive finite "
                "mass, length and energy");
    }
    bool operator==(const ReducedReference&) const = default;
};

/// All resolved physical units share the SI domain. Energy and pressure are
/// explicitly specified: e.g. real energy is per particle after removing mol,
/// and electron velocity uses fs, not the atomic unit of time.
class PhysicalUnits {
    double length_, mass_, time_, energy_, charge_, temperature_, pressure_,
        field_, density_, potential_;
    constexpr Unit make(Dimension d, double scale) const {
        return {d, scale, 0};
    }
    constexpr PhysicalUnits(double l, double m, double t, double e, double q,
                            double temperature, double p, double field,
                            double density = 0., double potential = 0.)
        : length_(l), mass_(m), time_(t), energy_(e), charge_(q),
          temperature_(temperature), pressure_(p), field_(field),
          density_(density > 0. ? density : m / (l * l * l)),
          potential_(potential > 0. ? potential : field * l) {}

  public:
    // Typed factory for setup/compile-time use: no string parsing.
    static constexpr PhysicalUnits
    from_style(Style style, const ReducedReference* reference = nullptr) {
        const double amu = 1.e-3 / avogadro;
        const double ev = elementary_charge_si;
        const double bohr = 1.e-10 * kBohrInAngstrom;
        switch (style) {
        case Style::real:
            return {1.e-10,  amu,   1.e-15, 4184. / avogadro, ev, 1.,
                    101325., 1.e10, 1000.};
        case Style::metal:
            return {1.e-10, amu, 1.e-12, ev, ev, 1., 1.e5, 1.e10, 1000.};
        case Style::si:
            return {1., 1., 1., 1., 1., 1., 1., 1.};
        case Style::cgs: {
            const double esu = ev / 4.803204712570263e-10;
            return {1.e-2, 1.e-3, 1., 1.e-7, esu, 1., .1, 1.e-5 / esu};
        }
        case Style::electron:
            return {bohr, amu, 1.e-15, kHartreeInEv * ev, ev, 1., 1.,
                    100., 0.,  1.};
        case Style::micro:
            return {1.e-6, 1.e-15, 1.e-6, 1.e-15, 1.e-12, 1., 1.e3, 1.e6};
        case Style::nano:
            return {1.e-9, 1.e-21, 1.e-9, 1.e-21, ev, 1., 1.e6, 1.e9};
        case Style::lj: {
            if (!reference)
                throw std::invalid_argument("units: physical conversion of LJ "
                                            "requires an explicit reference");
            reference->validate();
            const auto& r = *reference;
            const double t = std::sqrt(r.mass_kg / r.energy_j) * r.length_m;
            const double q = std::sqrt(r.energy_j * r.length_m / coulomb_si);
            const double p =
                r.energy_j / (r.length_m * r.length_m * r.length_m);
            const double field = r.energy_j / (r.length_m * q);
            if (!std::isfinite(t) || t <= 0 || !std::isfinite(q) || q <= 0
                || !std::isfinite(p) || p <= 0 || !std::isfinite(field)
                || field <= 0)
                throw std::invalid_argument(
                    "units: LJ reference produces invalid derived scales");
            return {r.length_m, r.mass_kg, t,
                    r.energy_j, q,         r.energy_j / boltzmann_si,
                    p,          field};
        }
        }
        throw std::invalid_argument("units: invalid physical profile");
    }
    // Strings belong to external input parsing only.
    static constexpr PhysicalUnits
    from_style(std::string_view profile,
               const ReducedReference* reference = nullptr) {
        if (profile == "electronic")
            return {1.e-10,
                    1.e-3 / avogadro,
                    1.e-15,
                    elementary_charge_si,
                    elementary_charge_si,
                    1.,
                    1.602176634e11,
                    1.e10};
        return from_style(parse_style(profile), reference);
    }
    static constexpr PhysicalUnits
    from_style(Style style, const std::optional<ReducedReference>& reference) {
        return from_style(style, reference ? &*reference : nullptr);
    }
    static constexpr PhysicalUnits
    from_style(std::string_view style,
               const std::optional<ReducedReference>& reference) {
        return from_style(style, reference ? &*reference : nullptr);
    }
    static PhysicalUnits from_style(Style style,
                                    const ReducedReference& reference) {
        return from_style(style, &reference);
    }
    constexpr Quantity quantity(double value, Unit unit) const {
        return {value, unit};
    }
    constexpr Unit length() const { return make(length_dimension, length_); }
    constexpr Unit mass() const { return make(mass_dimension, mass_); }
    constexpr Unit time() const { return make(time_dimension, time_); }
    constexpr Unit charge() const { return make(charge_dimension, charge_); }
    constexpr Unit temperature() const {
        return make(temperature_dimension, temperature_);
    }
    constexpr Unit velocity() const { return length() / time(); }
    constexpr Unit energy() const {
        return make(mass_dimension * length_dimension * length_dimension
                        / time_dimension / time_dimension,
                    energy_);
    }
    constexpr Unit force() const { return energy() / length(); }
    constexpr Unit pressure() const {
        return make((energy() / length() / length() / length()).dimension,
                    pressure_);
    }
    constexpr Unit density() const {
        return make((mass() / (length() * length() * length())).dimension,
                    density_);
    }
    constexpr Unit electric_field() const {
        return make((force() / charge()).dimension, field_);
    }
    constexpr Unit potential() const {
        // LAMMPS electrical potential uses the electric-field unit times length.
        return make((energy() / charge()).dimension, potential_);
    }
};
using UnitContext = PhysicalUnits;
} // namespace ATV::units
