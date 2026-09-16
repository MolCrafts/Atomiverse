/// @file cpu/scf/internal/dispersion.cpp
/// @brief The flag → dispersion-evaluator switch shared by every gradient a
///        mean-field method publishes (see the header).

#include "atomiverse/cpu/scf/internal/dispersion.h"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/dftd3.h"
#include "atomiverse/cpu/scf/dftd4.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

double dispersion_energy(const Frame& frame, bool d3bj, bool d4) {
    if (d4)
        return D4{}.energy(frame);
    if (d3bj)
        return D3BJ{}.energy(frame);
    return 0.0;
}

std::vector<std::array<double, 3>> dispersion_forces(const Frame& frame,
                                                     bool d3bj, bool d4) {
    if (d4)
        return D4{}.forces(frame);
    if (d3bj)
        return D3BJ{}.forces(frame);
    return {};
}

void add_dispersion_forces(const Frame& frame, bool d3bj, bool d4,
                           GradientResult& gr) {
    if (!d3bj && !d4)
        return;
    const std::vector<std::array<double, 3>> f_disp =
        dispersion_forces(frame, d3bj, d4);
    const int natm = static_cast<int>(gr.forces.shape()[0]);
    if (static_cast<int>(f_disp.size()) != natm)
        throw std::runtime_error(
            "add_dispersion_forces: the dispersion evaluator returned "
            + std::to_string(f_disp.size()) + " force triples for "
            + std::to_string(natm) + " atoms");
    for (int A = 0; A < natm; ++A)
        for (int d = 0; d < 3; ++d)
            gr.forces(A, d) +=
                units::convert(f_disp[static_cast<std::size_t>(A)]
                                     [static_cast<std::size_t>(d)],
                               units::Force::HartreePerBohr,
                               units::Force::eVPerAngstrom);
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
