/// @file cpu/scf/minao.h
/// @brief MINAO initial-density projector for PySCF-compatible SCF guess.
///
/// Mirrors `pyscf.scf.hf.init_guess_by_minao`: projects atomic fragment
/// densities given in an ANO-derived minimal basis onto the target AO basis
/// via cross-overlap integrals. Produces the same guess PySCF uses by
/// default for RHF / UHF / KS.
///
/// Supported elements: H, C, N, O, F (matches the MINAO tables generated
/// by tools/gen_minao_data.py). Callers should fall back to the core-
/// Hamiltonian guess for unsupported elements.
#pragma once

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integrals.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief True when every atom in the pack has a MINAO basis entry.
bool minao_supported(const IntegralPack& integrals);

/// @brief Build the RHF-style MINAO initial density matrix.
///
/// Returns a symmetric [n_basis x n_basis] density in the target AO basis.
/// Total electron count (Tr[P S]) equals the sum of MINAO occupations,
/// which matches the molecular electron count exactly for neutral
/// closed-shell systems using the standard fractional-occupation scheme.
///
/// For cationic/anionic systems the caller should scale the returned matrix
/// to match the target electron number.
///
/// @param integrals Target-basis integral pack (provides S and atoms).
/// @return Density matrix P in the target AO basis (column-major).
RealMatrix build_minao_density(const IntegralPack& integrals);

} // namespace scf
} // namespace cpu
} // namespace ATV
