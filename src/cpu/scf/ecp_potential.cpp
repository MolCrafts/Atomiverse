/// @file src/cpu/scf/ecp_potential.cpp
/// @brief EcpPotential lookup constructor over the in-tree def2-ECP table.

#include "atomiverse/cpu/scf/ecp_potential.h"

#include <stdexcept>
#include <string>

#include "atomiverse/cpu/scf/basis_data/ecp_def2.h"

namespace ATV {
namespace cpu {
namespace scf {

namespace {

/// IUPAC symbols indexed by Z, covering the basis_data tables (Z <= 86).
constexpr const char* kSymbols[] = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc",
    "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"};
constexpr int kMaxSymbolZ = sizeof(kSymbols) / sizeof(kSymbols[0]) - 1;

} // namespace

std::string EcpPotential::element_symbol(int Z) {
    if (Z < 1 || Z > kMaxSymbolZ)
        throw std::runtime_error(
            "EcpPotential::element_symbol: unsupported atomic number Z="
            + std::to_string(Z) + " (table covers Z=1.."
            + std::to_string(kMaxSymbolZ) + ")");
    return kSymbols[Z];
}

const std::vector<basis_data::AtomEcp>&
EcpPotential::family_table(std::string_view name) {
    if (name == "def2-ECP")
        return basis_data::ecp_def2();
    throw std::runtime_error("EcpPotential: unknown ECP family '"
                             + std::string(name) + "'. Supported: def2-ECP");
}

EcpPotential::EcpPotential(molrs::Element element, std::string_view name)
    : element(element), name(name) {
    const auto& table = family_table(name);
    const int z = static_cast<int>(element);
    if (z >= 1 && z < static_cast<int>(table.size())
        && !table[static_cast<std::size_t>(z)].channels.empty()) {
        const auto& row = table[static_cast<std::size_t>(z)];
        n_core = row.n_core;
        channels = row.channels;
        return;
    }
    std::string supported;
    for (int Z = 1; Z < static_cast<int>(table.size()); ++Z) {
        if (!table[static_cast<std::size_t>(Z)].channels.empty())
            supported += " " + element_symbol(Z);
    }
    throw std::runtime_error(
        "EcpPotential: no in-tree '" + std::string(name)
        + "' entry for Z=" + std::to_string(z)
        + " (all-electron element in this family). Supported:" + supported);
}

} // namespace scf
} // namespace cpu
} // namespace ATV
