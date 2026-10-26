#pragma once

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/vibration/supercell.h"

namespace atv_dfpt_finite_q_checks {

// Supplemental runtime curvature of the SAME Re C_00(X) observable. A
// (+h,-h) displacement of primitive atom0x across the two cells has squared
// pattern norm N_cells=2, so divide the supercell energy curvature by 2.
// This is not an external oracle; the sections retain their external spectra.
inline double x_pattern_curvature(ATV::cpu::pw::PW& method,
                                  const ATV::cpu::Frame& unit) {
    constexpr double h = 0.03; // original finite-displacement step [Å]
    std::array<double, 3> energies{};
    for (int step = -1; step <= 1; ++step) {
        auto supercell = ATV::cpu::vib::expand_supercell(unit, 2, 1, 1);
        supercell.span(ATV::kAtom_x)[0] += step * h;
        supercell.span(ATV::kAtom_x)[unit.nlocal()] -= step * h;
        const auto& result = method.run(supercell);
        if (!result.converged || !std::isfinite(result.energy_ha))
            throw std::runtime_error(
                "finite-q curvature: displaced SCF did not converge");
        energies[static_cast<std::size_t>(step + 1)] =
            result.energy_ha * ATV::units::kHartreeInEv;
    }
    const double curvature =
        (energies[2] - 2 * energies[1] + energies[0]) / (2 * h * h);
    if (!std::isfinite(curvature))
        throw std::runtime_error("finite-q curvature: non-finite curvature");
    std::printf("X-pattern supplemental curvature: E-=% .12f E0=% .12f E+=% "
                ".12f eV h=%.3f Ang Ncells=2 C00=% .8f eV/Ang^2\n",
                energies[0], energies[1], energies[2], h, curvature);
    return curvature;
}
} // namespace atv_dfpt_finite_q_checks
