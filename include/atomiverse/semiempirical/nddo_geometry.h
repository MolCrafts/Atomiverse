/// @file semiempirical/nddo_geometry.h
/// @brief Physical-contact check for NDDO molecular geometries.
///
/// An NDDO Hamiltonian is a molecular model fitted to chemically bonded and
/// non-bonded separations. Two nuclei much closer than any chemical bond have
/// no physical electronic state the SCF could converge to, so such inputs are
/// refused before the SCF with the offending atoms named. Backend-independent:
/// any NDDO engine that holds host coordinates calls the same check.
#pragma once

#include <cstddef>

#include "atomiverse/core/precision.h"
#include "atomiverse/semiempirical/nddo_result.h"

namespace ATV::semiempirical {

/// @brief Fraction of the covalent-radius sum below which a pair is refused.
///
/// The shortest chemical bonds sit at about 0.77 of the Cordero radius sum
/// (N≡N 1.098 Å / 1.42 Å = 0.77, C≡O 0.79, C≡C 0.79); H₂ at 0.74 Å is 1.19.
/// 0.6 leaves more than 20 % compression headroom below the tightest real bond
/// (thermal or reactive MD turning points) while refusing superposed atoms:
/// C–C < 0.912 Å, C–H < 0.642 Å, H–H < 0.372 Å.
inline constexpr double kNddoContactFraction = 0.6;

/// @brief Maximum number of offending pairs listed in the refusal message.
inline constexpr std::size_t kNddoContactReportLimit = 5;

/// @brief Single-bond covalent radius [Å].
///
/// Cordero et al., Dalton Trans. 2008, 2832 (DOI 10.1039/b801115j); the same
/// table molrs `Element::covalent_radius` carries (C sp³, low-spin Mn/Fe/Co).
/// @param z Atomic number, 1..86.
/// @return Radius [Å].
/// @throws std::invalid_argument for z outside 1..86 (no fallback radius).
double covalent_radius_angstrom(int z);

/// @brief Refuse atom pairs closer than `kNddoContactFraction` × (rᵢ + rⱼ).
///
/// Every pair is scanned (O(N²), cheaper than one Fock build). The message
/// names the method, the number of offending pairs, and the closest
/// `kNddoContactReportLimit` of them as 0-based atom index, element symbol,
/// distance and limit [Å].
/// @param method NDDO parameter family; supplies element symbols. Every
///        atomic number must already be supported by it.
/// @param n Atom count.
/// @param z Atomic numbers [n].
/// @param x,y,z_coord Cartesian coordinates [Å], [n] each; finite.
/// @throws std::invalid_argument when any pair is too close.
void require_nddo_contacts(SemiEmpiricalMethod method, std::size_t n,
                           const int* z, const StorageReal* x,
                           const StorageReal* y, const StorageReal* z_coord);

} // namespace ATV::semiempirical
