/// @file cpu/vib/atomic_masses.h
/// @brief IUPAC-2013 standard atomic weights for vibrational analysis.
///
/// Mirrors `pyscf/data/elements.py::MASSES` (IUPAC-2013, isotope-averaged
/// standard atomic weights). A mass mismatch shifts every frequency
/// (`nu ~ m^(-1/2)`) — using 12.000 for carbon costs ~0.7 cm^-1 at
/// 1500 cm^-1 — so the exact IUPAC-2013 values are mandatory.
///
/// The table is keyed by atomic number `Z`. `mass_amu(Z)` throws on an
/// unsupported element rather than returning a fallback, per the
/// project's no-fallback convention.
#pragma once

#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Standard atomic weight of element `Z` in amu (IUPAC-2013).
///
/// Values copied verbatim from PySCF `elements.MASSES`. Covers `Z = 1..18`
/// (H..Ar) — the range required by the vibrational test molecules and
/// typical organic chemistry. Extend the switch when heavier elements
/// are needed.
///
/// @param Z Atomic number (1 = H, 6 = C, ...).
/// @return Isotope-averaged standard atomic weight [amu].
/// @throws std::out_of_range if `Z` is outside the supported range.
inline double mass_amu(int Z) {
    switch (Z) {
    case 1:
        return 1.008; // H
    case 2:
        return 4.002602; // He
    case 3:
        return 6.94; // Li
    case 4:
        return 9.0121831; // Be
    case 5:
        return 10.81; // B
    case 6:
        return 12.011; // C
    case 7:
        return 14.007; // N
    case 8:
        return 15.999; // O
    case 9:
        return 18.998403163; // F
    case 10:
        return 20.1797; // Ne
    case 11:
        return 22.98976928; // Na
    case 12:
        return 24.305; // Mg
    case 13:
        return 26.9815385; // Al
    case 14:
        return 28.085; // Si
    case 15:
        return 30.973761998; // P
    case 16:
        return 32.06; // S
    case 17:
        return 35.45; // Cl
    case 18:
        return 39.948; // Ar
    default:
        throw std::out_of_range(
            "cpu::vib::mass_amu: unsupported atomic number Z="
            + std::to_string(Z) + " (IUPAC-2013 table covers Z=1..18)");
    }
}

/// @brief IUPAC element symbol of element `Z`.
///
/// Covers `Z = 1..18` (H..Ar) — the range required by the vibrational
/// test molecules. Used by `cpu::vib::NormalModeWriter` to format atom
/// symbols in the per-mode XYZ animations. Throws on an unsupported
/// element rather than returning a fallback, per the project's
/// no-fallback convention.
///
/// @param Z Atomic number (1 = H, 6 = C, ...).
/// @return IUPAC element symbol (e.g. `"C"`).
/// @throws std::out_of_range if `Z` is outside the supported range.
inline std::string element_symbol(int Z) {
    switch (Z) {
    case 1:
        return "H";
    case 2:
        return "He";
    case 3:
        return "Li";
    case 4:
        return "Be";
    case 5:
        return "B";
    case 6:
        return "C";
    case 7:
        return "N";
    case 8:
        return "O";
    case 9:
        return "F";
    case 10:
        return "Ne";
    case 11:
        return "Na";
    case 12:
        return "Mg";
    case 13:
        return "Al";
    case 14:
        return "Si";
    case 15:
        return "P";
    case 16:
        return "S";
    case 17:
        return "Cl";
    case 18:
        return "Ar";
    default:
        throw std::out_of_range(
            "cpu::vib::element_symbol: unsupported atomic number Z="
            + std::to_string(Z) + " (table covers Z=1..18)");
    }
}

} // namespace vib
} // namespace cpu
} // namespace ATV
