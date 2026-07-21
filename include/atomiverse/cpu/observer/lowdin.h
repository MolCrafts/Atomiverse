/// @file cpu/observer/lowdin.h
/// @brief Löwdin population analysis — stateless compute functions over a
///        mean-field method's shared outputs (`scf::MeanFieldOutputs`).
///
/// Löwdin symmetrically orthogonalizes the atom-centred basis before
/// partitioning the density, which removes the Mulliken scheme's equal-split
/// arbitrariness for the overlap population:
///
///   q_A     = Z_A − Σ_{μ∈A} (S^{1/2} P S^{1/2})_{μμ}      (Löwdin net charge)
///   σ_A     = Σ_{μ∈A} (S^{1/2} (P^α − P^β) S^{1/2})_{μμ}  (Löwdin spin density)
///
/// S is rebuilt from the handle's shells (Cartesian engine convention) and
/// sandwiched with the pack's `c2s` for spherical-convention bases so it
/// matches the determinant's AO dimension; S^{1/2} comes from a symmetric
/// eigendecomposition. Analysis is performed in the **unit-normalized AO
/// basis** (each AO rescaled to S_ii = 1, the density counter-scaled):
/// Löwdin populations are not invariant under per-AO rescaling, and
/// Cartesian d/f normalization conventions differ across engines in exactly
/// such diagonal factors — normalizing first makes the numbers
/// convention-independent (and matches the oracle protocol in
/// `tools/pin_charges_oracle.py`). The AO → atom map is derived per call by
/// matching
/// shell centers against the Frame's atom positions (Bohr, 1e-8), following
/// the `mulliken_charge` precedent — nothing is cached on Frame and no
/// transitional debt type appears in the signature.
///
/// Reference: Löwdin, J. Chem. Phys. 18, 365 (1950), DOI 10.1063/1.1747632.
#pragma once

#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {

/// @brief Löwdin net charges q_A = Z_A − Σ_{μ∈A}(S^{1/2} P S^{1/2})_{μμ}
///        with P the total density P^α + P^β of the handle's determinant.
///
/// @param outputs A mean-field method's `result_handle()` (determinant +
///        shells + optional c2s), read after the method ran.
/// @param frame   Geometry (positions [Å], atomic numbers in the type
///        column) — supplies the AO → atom map and the Z_A.
/// @return Per-atom Löwdin net charges [e], indexed by Frame atom order.
/// @throws std::invalid_argument when the handle is null or not ready.
/// @throws std::runtime_error when a shell center matches no atom position.
std::vector<double> lowdin_charge(const scf::MeanFieldOutputs& outputs,
                                  const Frame& frame);

/// @brief Löwdin spin densities σ_A = Σ_{μ∈A}(S^{1/2}(P^α−P^β)S^{1/2})_{μμ}.
///
/// Restricted determinants (P^α = P^β) yield exact zeros.
///
/// @param outputs A mean-field method's `result_handle()`.
/// @param frame   Geometry — supplies the AO → atom map.
/// @return Per-atom Löwdin spin densities [e], indexed by Frame atom order.
/// @throws std::invalid_argument when the handle is null or not ready.
/// @throws std::runtime_error when a shell center matches no atom position.
std::vector<double> lowdin_spin(const scf::MeanFieldOutputs& outputs,
                                const Frame& frame);

} // namespace cpu
} // namespace ATV
