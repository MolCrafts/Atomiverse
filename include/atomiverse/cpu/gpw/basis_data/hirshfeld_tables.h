/// @file cpu/gpw/basis_data/hirshfeld_tables.h
/// @brief Free-atom exponential parameters for Hirshfeld volume ratios.
///
/// Valence free-atom density model (in-tree only — no parsers):
///
///   ρ_Z^free(r) = A exp(−α r),  A = N_val α³ / (8π),  ∫ρ = N_val
///
/// N_val is the GTH default valence count for Z. α [1/Bohr] is static and
/// compact (valence-like) so free-atom support matches GTH valence density
/// extent. Used only for Hirshfeld weights / V_free; TS/MBD free-atom α₀/C₆/R
/// still come from `ts_tables.h`.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Highest Z with a Hirshfeld free-atom row (1-based inclusive).
constexpr int kHirshfeldNFree = 36;

/// @brief GTH-style valence electron count for atomic number Z.
/// @throws std::runtime_error if Z is outside the table (no fallback).
inline int hirshfeld_valence(int Z) {
    // clang-format off
    constexpr int kVal[kHirshfeldNFree] = {
        1, 2,                         // H He
        1, 2, 3, 4, 5, 6, 7, 8,       // Li–Ne
        1, 2, 3, 4, 5, 6, 7, 8,       // Na–Ar
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, // K–Zn (d in valence for late TM)
        3, 4, 5, 6, 7, 8              // Ga–Kr
    };
    // clang-format on
    if (Z < 1 || Z > kHirshfeldNFree)
        throw std::runtime_error(
            "GPW Hirshfeld: element Z=" + std::to_string(Z)
            + " is not in the free-atom valence table (no fallback)");
    return kVal[Z - 1];
}

/// @brief Exponential decay α [1/Bohr] for free-atom ρ_Z^free.
///
/// Compact valence-like decays so free-atom support matches GTH valence
/// density extent (diffuse α ≪ 1 produced unphysically small \(V/V_{\mathrm{free}}\)
/// vs GPW density). Light-element values are hydrogenic / Slater-inspired;
/// heavier rows scale as ~1.5 / period-index.
/// @throws std::runtime_error if Z is unsupported.
inline double hirshfeld_alpha(int Z) {
    // clang-format off
    constexpr double kAlpha[kHirshfeldNFree] = {
        2.00, // H
        2.50, // He
        1.20, // Li
        1.50, // Be
        1.70, // B
        1.80, // C
        1.95, // N
        2.20, // O
        2.40, // F
        2.60, // Ne
        1.10, // Na
        1.30, // Mg
        1.45, // Al
        1.55, // Si
        1.65, // P
        1.75, // S
        1.85, // Cl
        1.95, // Ar
        0.95, // K
        1.10, // Ca
        1.20, // Sc
        1.25, // Ti
        1.30, // V
        1.35, // Cr
        1.40, // Mn
        1.45, // Fe
        1.50, // Co
        1.55, // Ni
        1.55, // Cu
        1.50, // Zn
        1.40, // Ga
        1.45, // Ge
        1.50, // As
        1.55, // Se
        1.60, // Br
        1.65  // Kr
    };
    // clang-format on
    if (Z < 1 || Z > kHirshfeldNFree)
        throw std::runtime_error(
            "GPW Hirshfeld: element Z=" + std::to_string(Z)
            + " is not in the free-atom alpha table (no fallback)");
    return kAlpha[Z - 1];
}

/// @brief True when Z has a Hirshfeld free-atom row.
inline bool hirshfeld_supports_element(int Z) {
    return Z >= 1 && Z <= kHirshfeldNFree;
}

/// @brief Free-atom volume V_free = ∫ ρ_free(r) r³ d³r for the exponential model.
///        Analytic: V_free = 60 N_val / α³.
inline double hirshfeld_v_free(int Z) {
    const double n = static_cast<double>(hirshfeld_valence(Z));
    const double a = hirshfeld_alpha(Z);
    return 60.0 * n / (a * a * a);
}

/// @brief Free-atom density ρ_Z^free(r) [e/Bohr³] at radius r [Bohr].
inline double hirshfeld_rho_free(int Z, double r) {
    const double n = static_cast<double>(hirshfeld_valence(Z));
    const double a = hirshfeld_alpha(Z);
    const double A = n * a * a * a / (8.0 * 3.14159265358979323846);
    return A * std::exp(-a * r);
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
