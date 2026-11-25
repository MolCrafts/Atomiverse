/// @file core/unit_system.h
/// Prepared run-unit values. All conversions derive from PhysicalUnits;
/// there is no second hand-maintained table of equation factors.
#pragma once
#include "atomiverse/core/physical_units.h"
#include <limits>
// nvcc does not support constexpr destruction of the host-only owning
// reference metadata (libstdc++ string/optional). Prepared numeric Conversion
// and PhysicalUnits remain constexpr on both compilers.
#if defined(__CUDACC__)
#define ATV_UNIT_SYSTEM_CONSTEXPR inline
#else
#define ATV_UNIT_SYSTEM_CONSTEXPR constexpr
#endif
namespace ATV::units {
struct UnitSystem {
    Style style = Style::lj;
    Conversion kinetic_energy;
    Conversion velocity_impulse;
    Conversion density;
    Conversion pressure;
    Conversion electric_field;
    Conversion electric_potential;
    Conversion length_from_angstrom;
    Conversion time_from_femtoseconds;
    Conversion charge_from_electron;

  private:
    double boltzmann_ = 1.;
    double planck_ = 1.;
    double coulomb_ = 1.;
    std::optional<PhysicalUnits> physical_;
    std::optional<ReducedReference> reference_;
    friend ATV_UNIT_SYSTEM_CONSTEXPR UnitSystem from_style(Style);
    static ATV_UNIT_SYSTEM_CONSTEXPR UnitSystem prepare(const PhysicalUnits&);

  public:
    const PhysicalUnits& physical() const {
        if (!physical_)
            throw std::invalid_argument("units: LJ needs a physical reference "
                                        "for coupling to physical models");
        return *physical_;
    }
    const std::optional<ReducedReference>& reference() const {
        return reference_;
    }
    void bind_reference(const ReducedReference& reference) {
        if (style != Style::lj)
            throw std::invalid_argument(
                "units: only LJ has reduced reference scales");
        reference.validate();
        physical_ = PhysicalUnits::from_style(Style::lj, reference);
        reference_ = reference;
        constexpr auto si = PhysicalUnits::from_style(Style::si);
        length_from_angstrom =
            Conversion(Unit{length_dimension, 1.e-10, 0}, physical_->length());
        time_from_femtoseconds =
            Conversion(Unit{time_dimension, 1.e-15, 0}, physical_->time());
        charge_from_electron =
            Conversion(Unit{charge_dimension, elementary_charge_si, 0},
                       physical_->charge());
        planck_ = convert(planck_si, si.energy() * si.time(),
                          physical_->energy() * physical_->time());
    }
    double dielectric = 1.;
    double dt = .005;
    double skin = .3;
    constexpr const char* name() const {
        // Native electronic execution differs from metal in its clock.
        // Keep serialized results faithful to the actual prepared basis.
        return style == Style::metal && physical_
                       && physical_->time().scale
                              == PhysicalUnits::from_style("electronic")
                                     .time()
                                     .scale
                   ? "electronic"
                   : style_name(style);
    }
    constexpr double boltzmann_constant() const { return boltzmann_; }
    constexpr double planck_constant() const { return planck_; }
    constexpr double thermal_energy(double temperature) const {
        return boltzmann_ * temperature;
    }
    constexpr double coulomb_constant() const { return coulomb_ / dielectric; }
    constexpr double two_charge_force() const {
        return coulomb_ * charge_from_electron(1.) * charge_from_electron(1.)
               / (length_from_angstrom(1.) * length_from_angstrom(1.));
    }
    /// 1 eV/Å³ (160.2176634 GPa) in the run's pressure unit; 1 for LJ
    /// without a physical reference.
    double ev_per_cubic_angstrom() const {
        if (!physical_)
            return 1.;
        return 1.602176634e11 / physical_->pressure().scale;
    }
    static ATV_UNIT_SYSTEM_CONSTEXPR UnitSystem lj() { return {}; }
    static ATV_UNIT_SYSTEM_CONSTEXPR UnitSystem electronic();
};
ATV_UNIT_SYSTEM_CONSTEXPR UnitSystem
UnitSystem::prepare(const PhysicalUnits& u) {
    UnitSystem out;
    out.physical_ = u;
    constexpr auto si = PhysicalUnits::from_style(Style::si);
    out.boltzmann_ = convert(boltzmann_si, si.energy() / si.temperature(),
                             u.energy() / u.temperature());
    out.planck_ =
        convert(planck_si, si.energy() * si.time(), u.energy() * u.time());
    out.kinetic_energy =
        Conversion(u.mass() * u.velocity() * u.velocity(), u.energy());
    out.velocity_impulse =
        Conversion(u.force() / u.mass() * u.time(), u.velocity());
    out.density = Conversion(u.mass() / (u.length() * u.length() * u.length()),
                             u.density());
    out.pressure = Conversion(
        u.energy() / (u.length() * u.length() * u.length()), u.pressure());
    out.coulomb_ = convert(
        coulomb_si, si.energy() * si.length() / (si.charge() * si.charge()),
        u.energy() * u.length() / (u.charge() * u.charge()));
    out.electric_field = Conversion(u.electric_field(), u.force() / u.charge());
    out.electric_potential = Conversion(u.potential(), u.energy() / u.charge());
    out.length_from_angstrom =
        Conversion(Unit{length_dimension, 1.e-10, 0}, u.length());
    out.time_from_femtoseconds =
        Conversion(Unit{time_dimension, 1.e-15, 0}, u.time());
    out.charge_from_electron =
        Conversion(Unit{charge_dimension, elementary_charge_si, 0}, u.charge());
    return out;
}
ATV_UNIT_SYSTEM_CONSTEXPR UnitSystem UnitSystem::electronic() {
    auto out = prepare(PhysicalUnits::from_style("electronic"));
    out.style = Style::metal;
    return out;
}
[[nodiscard]] ATV_UNIT_SYSTEM_CONSTEXPR UnitSystem from_style(Style style) {
    if (style == Style::lj)
        return {};
    auto out = UnitSystem::prepare(PhysicalUnits::from_style(style));
    out.style = style;
    constexpr double timesteps[] = {.005,  1.,   .001, 1.e-8,
                                    1.e-8, .001, 2.,   .00045};
    constexpr double skins[] = {.3, 2., 2., .001, .1, 2., .1, .1};
    out.dt = timesteps[static_cast<unsigned>(style)];
    out.skin = skins[static_cast<unsigned>(style)];
    return out;
}
[[nodiscard]] inline UnitSystem from_style(std::string_view style) {
    return from_style(parse_style(style));
}

/// @brief Sentinel for a pressure-valued default that is resolved against
///        the run's units at build/launch time (e.g. a cell-relax bulk
///        modulus of 1 eV/Å³), because a bare number would mean a different
///        pressure in every style.
inline constexpr double kStyleDefault = -std::numeric_limits<double>::max();

/// @brief `value`, or `style_value` when `value` is `kStyleDefault`.
[[nodiscard]] constexpr double resolve_style_default(double value,
                                                     double style_value) {
    return value == kStyleDefault ? style_value : value;
}
} // namespace ATV::units

namespace ATV::units {

/// @brief de Broglie thermal wavelength in a LAMMPS `units` style.
///
/// `Λ = hplanck / sqrt(2 π m boltz T mvv2e)` (LAMMPS `fix gcmc`), with mass,
/// temperature and the result in the style's own units.
/// @param mass Particle mass in the style's mass unit.
/// @param temperature Temperature in the style's temperature unit.
/// @param u The run's unit system.
/// @return Λ in the style's length unit.
[[nodiscard]] inline double lambda_thermal(double mass, double temperature,
                                           const UnitSystem& u) {
    return u.planck_constant()
           / std::sqrt(2.0 * pi * mass * u.thermal_energy(temperature)
                       * u.kinetic_energy.scale());
}

} // namespace ATV::units

#undef ATV_UNIT_SYSTEM_CONSTEXPR
