/// @file cpu/potentials/sw_data/sw_tables.h
/// @brief Static parameter table for the Stillinger-Weber potential.
///
/// Transcribed once from the published table and committed as plain C++ data.
/// Nothing is parsed at build, test, or run time — no `.sw` reader exists, per the
/// GPW precedent (CLAUDE.md). Consumed only by the `SWPotential(molrs::Element, name)`
/// lookup constructor, which throws on an unsupported element or name.
///
/// Provenance: F. H. Stillinger and T. A. Weber, "Computer simulation of local
/// order in condensed phases of silicon", *Phys. Rev. B* **31**, 5262 (1985).
/// DOI 10.1103/PhysRevB.31.5262. Parameters exactly as published (eps [eV],
/// sigma [Angstrom]; A, B, p, q, a, lambda, gamma dimensionless).
#pragma once

namespace ATV {
namespace cpu {
namespace sw_data {

/// @brief One Stillinger-Weber entry.
///
/// Two-body:
///   phi2(r) = A*eps*[B*(sigma/r)^p - (sigma/r)^q] * exp(sigma/(r - a*sigma))
/// Three-body (vertex at i, legs to j and k):
///   phi3 = lambda*eps*(cos(theta) - cos(theta0))^2
///          * exp(gamma*sigma/(r_ij - a*sigma)) * exp(gamma*sigma/(r_ik - a*sigma))
/// Both vanish identically beyond r = a*sigma, which is therefore the cutoff.
struct SWRow {
    int z;              ///< atomic number
    const char* symbol; ///< element symbol (documentation only)
    const char* name;   ///< parameterization name
    double eps;         ///< energy scale [eV]
    double sigma;       ///< length scale [Angstrom]
    double A;
    double B;
    double p;
    double q;
    double a;          ///< cutoff in units of sigma
    double lambda;     ///< three-body strength
    double gamma;      ///< three-body exponential prefactor
    double cos_theta0; ///< the tetrahedral cosine, -1/3
};

/// @brief The original silicon parameterization (Stillinger & Weber 1985).
inline constexpr SWRow kStillingerWeber[] = {
    {14, "Si", "stillinger-weber", 2.1683, 2.0951, 7.049556277, 0.6022245584,
     4.0, 0.0, 1.80, 21.0, 1.20, -1.0 / 3.0},
};

inline constexpr unsigned long kStillingerWeberCount =
    sizeof(kStillingerWeber) / sizeof(kStillingerWeber[0]);

} // namespace sw_data
} // namespace cpu
} // namespace ATV
