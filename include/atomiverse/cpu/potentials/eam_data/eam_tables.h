/// @file cpu/potentials/eam_data/eam_tables.h
/// @brief Static parameter tables for the analytic EAM (Sutton-Chen) potential.
///
/// Transcribed once from the published table and committed as plain C++ data.
/// **Nothing is parsed at build, test, or run time** — no setfl / .eam.alloy /
/// funcfl reader exists anywhere in the tree, per the binding precedent set by
/// the GPW data tables (CLAUDE.md, "GPW data (in-tree static .h tables, no
/// parsing)"). Consumed only by the `EAMPotential(molrs::Element, name)` lookup
/// constructor, which throws on an unsupported element or name.
///
/// Provenance: A. P. Sutton and J. Chen, "Long-range Finnis-Sinclair potentials",
/// *Philos. Mag. Lett.* **61**, 139 (1990). DOI 10.1080/09500839008206493.
/// Parameters as published in that paper's table (n, m dimensionless;
/// eps [eV]; c dimensionless; a [Angstrom], the fcc lattice constant).
///
/// Unlike the GPW tables there is no generator under tools/: these are five
/// numbers per element read off a printed table, so a generator would be a
/// program with no second caller (CLAUDE.md: do not extract single-use code).
#pragma once

namespace ATV {
namespace cpu {
namespace eam_data {

/// @brief One analytic-EAM (Sutton-Chen) entry.
///
/// The Sutton-Chen energy, in the paper's notation:
///
///     E = eps * [ 1/2 * sum_i sum_{j!=i} (a/r_ij)^n  -  c * sum_i sqrt(rho_i) ]
///     rho_i = sum_{j!=i} (a/r_ij)^m
///
/// i.e. an EAM with pair term phi(r) = eps*(a/r)^n, density f(r) = (a/r)^m, and
/// embedding F(rho) = -eps*c*sqrt(rho). The square root is what makes it
/// many-body: it cannot be decomposed into a sum over independent pairs.
struct SuttonChenRow {
    int z;              ///< atomic number
    const char* symbol; ///< element symbol (documentation only)
    const char* name;   ///< parameterization name
    int n;              ///< repulsive-pair exponent
    int m;              ///< density exponent
    double eps;         ///< energy scale [eV]
    double c;           ///< embedding coefficient [dimensionless]
    double a;           ///< length scale = fcc lattice constant [Angstrom]
};

/// @brief The Sutton-Chen fcc set (Sutton & Chen 1990).
inline constexpr SuttonChenRow kSuttonChen[] = {
    {28, "Ni", "sutton-chen", 9, 6, 1.5707e-2, 39.432, 3.52},
    {29, "Cu", "sutton-chen", 9, 6, 1.2382e-2, 39.432, 3.61},
    {45, "Rh", "sutton-chen", 12, 6, 4.9371e-3, 144.41, 3.80},
    {46, "Pd", "sutton-chen", 12, 7, 4.1790e-3, 108.27, 3.89},
    {47, "Ag", "sutton-chen", 12, 6, 2.5415e-3, 144.41, 4.09},
    {77, "Ir", "sutton-chen", 14, 6, 2.4489e-3, 334.94, 3.84},
    {78, "Pt", "sutton-chen", 10, 8, 1.9833e-2, 34.408, 3.92},
    {79, "Au", "sutton-chen", 10, 8, 1.2793e-2, 34.408, 4.08},
};

inline constexpr unsigned long kSuttonChenCount =
    sizeof(kSuttonChen) / sizeof(kSuttonChen[0]);

} // namespace eam_data
} // namespace cpu
} // namespace ATV
