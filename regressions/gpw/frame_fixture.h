// Public Frame helpers for archived GPW comparisons. Not a solver API.
#pragma once

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"

#include <array>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace ATV::regression::gpw_fixture {

// Original archive fixtures use Bohr. Convert only at the public Frame boundary.
inline cpu::Frame frame_bohr(std::initializer_list<const double*> positions,
                             std::initializer_list<int> atomic_numbers,
                             const std::array<double, 3>& cell_bohr) {
    if (positions.size() != atomic_numbers.size())
        throw std::invalid_argument(
            "GPW fixture species/position count mismatch");
    cpu::Frame frame(static_cast<int>(positions.size()));
    frame.box = cpu::Box::orthorhombic(cell_bohr[0] * units::kBohrInAngstrom,
                                       cell_bohr[1] * units::kBohrInAngstrom,
                                       cell_bohr[2] * units::kBohrInAngstrom);
    auto species = atomic_numbers.begin();
    int i = 0;
    for (const double* position : positions) {
        frame.span(kAtom_type)[i] = *species++;
        frame.span(kAtom_x)[i] = position[0] * units::kBohrInAngstrom;
        frame.span(kAtom_y)[i] = position[1] * units::kBohrInAngstrom;
        frame.span(kAtom_z)[i] = position[2] * units::kBohrInAngstrom;
        ++i;
    }
    return frame;
}

// Preserve each archive's original Ha/Bohr gates while methods publish eV/Å.
inline std::vector<std::array<double, 3>>
forces_ha_bohr(const cpu::Frame& frame) {
    std::vector<std::array<double, 3>> forces(frame.nlocal());
    for (int i = 0; i < frame.nlocal(); ++i) {
        forces[i] = {
            frame.span(kAtom_fx)[i] / units::kHartreePerBohrInEvPerAngstrom,
            frame.span(kAtom_fy)[i] / units::kHartreePerBohrInEvPerAngstrom,
            frame.span(kAtom_fz)[i] / units::kHartreePerBohrInEvPerAngstrom};
    }
    return forces;
}

inline std::array<double, 6>
stress_ha_bohr3(std::array<double, 6> stress_eva3) {
    for (double& value : stress_eva3)
        value /= units::kHartreePerBohr3InEvPerAngstrom3;
    return stress_eva3;
}

} // namespace ATV::regression::gpw_fixture
