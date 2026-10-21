/// @file cpu/potentials/lj_type_coeff.h
/// @brief Data contract for Lennard-Jones coefficient tables and parsers.
#pragma once

namespace ATV::cpu {

/// @brief One Lennard-Jones type-pair coefficient entry in public units.
struct LJTypeCoeff {
    int type_i;     ///< First atom type (Frame kAtom_type value).
    int type_j;     ///< Second atom type.
    double epsilon; ///< Well depth [eV].
    double sigma;   ///< Length scale [Angstrom].
};

} // namespace ATV::cpu
