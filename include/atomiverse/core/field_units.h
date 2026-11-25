/// @file core/field_units.h
/// Physical interpretation of canonical atom columns. Dynamic columns must
/// declare their own units before a boundary attempts to convert them.
#pragma once
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/quantity.h"
#include <optional>
#include <string_view>
namespace ATV::units {
template <class Context>
constexpr Unit field_unit(AtomFieldId field, const Context& context) {
    switch (field) {
    case AtomFieldId::x:
    case AtomFieldId::y:
    case AtomFieldId::z:
        return context.length();
    case AtomFieldId::vx:
    case AtomFieldId::vy:
    case AtomFieldId::vz:
        return context.velocity();
    case AtomFieldId::fx:
    case AtomFieldId::fy:
    case AtomFieldId::fz:
        return context.force();
    case AtomFieldId::mass:
        return context.mass();
    case AtomFieldId::charge:
        return context.charge();
    // Identity, counts, image flags and topology carry no physical unit.
    case AtomFieldId::type:
    case AtomFieldId::atomic_number:
    case AtomFieldId::tag:
    case AtomFieldId::image:
    case AtomFieldId::num_bond:
    case AtomFieldId::bond_atom:
    case AtomFieldId::bond_type:
    case AtomFieldId::num_angle:
    case AtomFieldId::angle_atom:
    case AtomFieldId::angle_type:
    case AtomFieldId::num_dihedral:
    case AtomFieldId::dihedral_atom:
    case AtomFieldId::dihedral_type:
    case AtomFieldId::num_improper:
    case AtomFieldId::improper_atom:
    case AtomFieldId::improper_type:
    case AtomFieldId::nspecial:
    case AtomFieldId::special:
        return context.length() / context.length();
    case AtomFieldId::COUNT:
        break;
    }
    throw std::invalid_argument("units: atom field has no unit declaration");
}
/// Exact schema declarations for floating metadata; an absent declaration
/// must fail at a conversion boundary. Extensive stress has energy units.
template <class Context>
std::optional<Unit> metadata_unit(std::string_view key,
                                  const Context& context) {
    if (key == kEnergy.key || key == kEnergyPair.key || key == kEnergyBond.key
        || key == kEnergyAngle.key || key == kEnergyDihedral.key
        || key == kEnergyImproper.key || key == kEnergyChargingWork.key
        || key == kEnergyElectronic.key || key == kEnergyElectronKinetic.key
        || key == kEnergyElectronThermostat.key || key == kKinetic.key
        || key == kConserved.key || key == kThermostatEnergy.key
        || key == kStress.key || key == kStressKinetic.key)
        return context.energy();
    if (key == kDipole.key)
        return context.charge() * context.length();
    if (key == kQuadrupole.key)
        return context.charge() * context.length() * context.length();
    if (key == kMultipoleOrigin.key)
        return context.length();
    if (key == kElectrodeCharge.key)
        return context.charge();
    if (key == kPotentialDrop.key)
        return context.potential();
    if (key == kCapacitanceDiff.key || key == kCapacitanceIntegral.key)
        return context.charge() / context.potential();
    // The KMC clock is physical time in ps in every run style
    // (cpu/montecarlo/kinetic.h); its displacement is a Frame length.
    if (key == "kmc_time_ps" || key == "kmc_waiting_time_ps")
        return Unit{time_dimension, 1.e-12, 0};
    if (key == "kmc_displacement")
        return context.length();
    // Reference-scale metadata itself is always in SI, not reduced units.
    if (key == "units_lj_mass_kg")
        return Unit{mass_dimension, 1., 0};
    if (key == "units_lj_length_m")
        return Unit{length_dimension, 1., 0};
    if (key == "units_lj_energy_j")
        return Unit{mass_dimension * length_dimension * length_dimension
                        / time_dimension / time_dimension,
                    1., 0};
    return {};
}
} // namespace ATV::units
