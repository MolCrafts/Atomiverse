/// @file examples/cpp/molecules.h
/// @brief Small isolated molecules shared by the numbered C++ examples.
///
/// A `cpu::Frame` is the only input every method reads: positions [Å],
/// atomic number in `kAtom_atomic_number`, masses [amu] and a box. The examples are
/// isolated molecules, so the box is a large cubic vacuum cell.
#pragma once

#include <array>
#include <cstddef>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"

namespace examples {

/// One atom of a molecule: atomic number, mass [amu], position [Å].
struct Atom {
    int number;
    double mass;
    double x, y, z;
};

/// Build a frame from atoms in a cubic vacuum box of side `box` [Å].
template <std::size_t N>
ATV::cpu::Frame molecule(const std::array<Atom, N>& atoms, double box = 20.0) {
    ATV::cpu::Frame f(static_cast<int>(N));
    for (std::size_t i = 0; i < N; ++i) {
        f.span(ATV::kAtom_atomic_number)[i] = atoms[i].number;
        f.span(ATV::kAtom_mass)[i] = atoms[i].mass;
        f.span(ATV::kAtom_x)[i] = atoms[i].x;
        f.span(ATV::kAtom_y)[i] = atoms[i].y;
        f.span(ATV::kAtom_z)[i] = atoms[i].z;
    }
    f.box = ATV::cpu::Box::cubic(box);
    return f;
}

/// H2 along x with bond length `bond` [Å].
inline ATV::cpu::Frame h2(double bond = 0.74) {
    return molecule<2>(
        {{{1, 1.008, 0.0, 0.0, 0.0}, {1, 1.008, bond, 0.0, 0.0}}});
}

/// A single hydrogen atom (doublet).
inline ATV::cpu::Frame hydrogen_atom() {
    return molecule<1>({{{1, 1.008, 0.0, 0.0, 0.0}}});
}

/// Water at the experimental geometry (O–H 0.9572 Å, H–O–H 104.5°).
inline ATV::cpu::Frame water() {
    return molecule<3>({{{8, 15.999, 0.0, 0.0, 0.0},
                         {1, 1.008, 0.7572, 0.0, -0.5860},
                         {1, 1.008, -0.7572, 0.0, -0.5860}}});
}

} // namespace examples
