/// @file cpu/gpw/gth_potential.h
/// @brief GthPotential — analytic Goedecker-Teter-Hutter pseudopotential,
///        constructed by (`molrs::Element`, name) lookup into the in-tree static tables
///        (basis_data/gpw_tables.h). No file or string parsing anywhere.
#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief One nonlocal projector angular-momentum channel of a GTH potential.
struct GthNonlocalChannel {
    int l = 0;      ///< angular momentum (0=s, 1=p, ...), by channel order.
    double r = 0.0; ///< projector radius r_l [Bohr].
    int nproj = 0;  ///< number of projectors in this channel.
    std::vector<double>
        h; ///< nproj*nproj symmetric coupling matrix, row-major.

    /// @brief h(i,j) accessor into the row-major dense matrix.
    double hij(int i, int j) const {
        return h[static_cast<std::size_t>(i) * nproj + j];
    }
};

/// @brief Analytic GTH pseudopotential for one element/parameterization.
///
/// Local part: V_loc(r) = -Zion/r erf(r/(sqrt(2) rloc))
///   + exp(-(r/rloc)^2/2) * sum_i C[i] (r/rloc)^(2(i-1)).
/// Nonlocal part: separable Gaussian projectors with couplings `channels[*].h`.
struct GthPotential {
    molrs::Element element;   ///< chemical element (molrs enum, Z = value).
    std::string name;         ///< parameterization, e.g. "GTH-PBE-q6".
    int zion = 0;             ///< number of valence electrons.
    double rloc = 0.0;        ///< local Gaussian radius [Bohr].
    std::vector<double> cloc; ///< local polynomial coefficients C_1..C_n.
    std::vector<GthNonlocalChannel> channels; ///< nonlocal projector channels.

    /// @brief Construct an explicitly supplied potential without a lookup.
    /// @param element  Chemical element owning the potential.
    /// @param name     Identifier for the supplied potential.
    /// @param zion     Number of valence electrons.
    /// @param rloc     Local Gaussian radius [Bohr].
    /// @param cloc     Local polynomial coefficients.
    /// @param channels Complete nonlocal-projector payload (possibly empty).
    GthPotential(molrs::Element element, std::string name, int zion,
                 double rloc, std::vector<double> cloc,
                 std::vector<GthNonlocalChannel> channels)
        : element(element), name(std::move(name)), zion(zion), rloc(rloc),
          cloc(std::move(cloc)), channels(std::move(channels)) {}

    /// @brief Look up a published GTH parameterization from the in-tree table.
    ///
    /// Provenance: GTH-PBE — Krack, TCA 114, 145 (2005); GTH-PADE —
    /// Goedecker-Teter-Hutter, PRB 54, 1703 (1996). Throws std::runtime_error
    /// naming the supported entries when (element, name) is not in the table —
    /// no fallback.
    /// @param element Chemical element (e.g. `molrs::Element::O`).
    /// @param name    Parameterization name (e.g. "GTH-PBE-q6").
    GthPotential(molrs::Element element, std::string_view name);
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
