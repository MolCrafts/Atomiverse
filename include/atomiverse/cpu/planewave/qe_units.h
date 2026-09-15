/// @file cpu/planewave/qe_units.h
/// @brief Quantum ESPRESSO-rounded unit factors owned by the PW method.
#pragma once

namespace ATV::cpu::pw::qe_units {

/// Quantum ESPRESSO convention: 1 Bohr = 0.5291772083 Angstrom.
inline constexpr double angstrom_to_bohr = 1.8897259886;

/// Exact Rydberg-to-Hartree factor.
inline constexpr double rydberg_to_hartree = 0.5;

/// Quantum ESPRESSO-rounded Rydberg-to-eV factor.
inline constexpr double rydberg_to_ev = 13.605693;

} // namespace ATV::cpu::pw::qe_units
