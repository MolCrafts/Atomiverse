/// @file cpu/scf/ecp_potential.h
/// @brief EcpPotential — semi-local effective core potential for one element,
///        constructed by (`molrs::Element`, name) lookup into the in-tree
///        static tables (basis_data/ecp_def2.h). No file or string parsing
///        anywhere; unsupported (element, name) throws — no fallback.
#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "atomiverse/cpu/scf/basis_data/basis_table.h"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Semi-local effective core potential of one element/parameterization:
///        U = U_L + Σ_{l<L} Σ_m |lm⟩ (U_l − U_L) ⟨lm|, each channel a Gaussian
///        expansion Σ_k d_k r^{n_k−2} exp(−ζ_k r²) (Kahn form).
///
/// Mirrors the `GthPotential` lookup-constructor precedent: an explicit
/// parameter constructor plus a table-lookup constructor over the checked-in
/// def2-ECP data. `channels` is sorted by ascending l with the local part
/// U_L last (see `basis_data::AtomEcp`).
struct EcpPotential {
    molrs::Element element; ///< Chemical element (molrs enum, Z = value).
    std::string name;       ///< Parameterization family, e.g. "def2-ECP".
    int n_core = 0;         ///< Core electrons replaced by the ECP.
    std::vector<basis_data::EcpChannel>
        channels; ///< Projector channels (U_l − U_L), then local U_L last.

    /// @brief Construct an explicitly supplied potential without a lookup.
    /// @param element  Chemical element owning the potential.
    /// @param name     Identifier for the supplied potential.
    /// @param n_core   Number of core electrons replaced [dimensionless].
    /// @param channels Complete channel payload, local U_L last (exponents
    ///                 [Bohr^-2], coefficients [Hartree]).
    EcpPotential(molrs::Element element, std::string name, int n_core,
                 std::vector<basis_data::EcpChannel> channels)
        : element(element), name(std::move(name)), n_core(n_core),
          channels(std::move(channels)) {}

    /// @brief Look up a published ECP parameterization from the in-tree table.
    ///
    /// Provenance: def2-ECP — Weigend, Ahlrichs, PCCP 7, 3297 (2005);
    /// I ECP28MWB — Bergner et al., Mol. Phys. 80, 1431 (1993); Au ECP60MWB —
    /// Andrae et al., Theor. Chim. Acta 77, 123 (1990). Throws
    /// std::runtime_error naming the supported entries when (element, name)
    /// is not in the table — no fallback.
    /// @param element Chemical element (e.g. `molrs::Element::I`).
    /// @param name    Family name (currently "def2-ECP").
    EcpPotential(molrs::Element element, std::string_view name);

    /// @brief Resolve a supported ECP family name to its Z-indexed table.
    ///
    /// Shared by the lookup constructor, `internal::IntegralWorkspace` (core-Hamiltonian
    /// wiring), and the method-level x2c double-counting gate. Empty rows
    /// mark all-electron elements (def2 convention: Z ≤ 36).
    /// @param name Family name (currently "def2-ECP").
    /// @return The Z-indexed static table.
    /// @throws std::runtime_error naming the supported families for an
    ///         unknown name — no fallback.
    static const std::vector<basis_data::AtomEcp>&
    family_table(std::string_view name);

    /// @brief IUPAC element symbol for atomic number `Z` (Z = 1..86).
    ///
    /// Used in lookup/gate diagnostics (supported-entry listings and the
    /// ECP↔x2c double-counting error). Throws std::runtime_error outside
    /// the covered range — no fallback.
    /// @param Z Atomic number [dimensionless].
    /// @return Element symbol, e.g. "Au".
    static std::string element_symbol(int Z);
};

} // namespace scf
} // namespace cpu
} // namespace ATV
