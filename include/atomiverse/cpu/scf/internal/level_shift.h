/// @file cpu/scf/internal/level_shift.h
/// @brief internal::apply_level_shift — Saunders–Hillier virtual-orbital
///        level shift F' = F + λ(S − S·D_occ·S).
///
/// **Internal SCF helper.** Free-function math operation following the
/// `internal::diis_error` precedent: not a Component, never registered
/// into a Plan, not included from facade-public headers (tests may).
///
/// Spectral effect: with D_occ idempotent w.r.t. S (D_occ·S·D_occ = D_occ),
/// the generalized eigenvalues of F' shift every VIRTUAL orbital up by
/// exactly λ while the occupied ones are untouched — widening the
/// HOMO–LUMO gap and damping occupied/virtual oscillation.
///
/// Reference: V. R. Saunders & I. H. Hillier, Int. J. Quantum Chem. 7, 699
/// (1973), DOI 10.1002/qua.560070407. Ordering parity with PySCF
/// (scf/hf.py get_fock): damping → DIIS on the UNSHIFTED Fock → level
/// shift → diagonalize; skipped when |λ| <= 1e-4.
#pragma once

#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief PySCF skip rule: shifts at or below this magnitude [Hartree] are
///        a strict no-op — both in `apply_level_shift` and in the solver's
///        `shift_active` gate (single shared constant so the two sites can
///        never diverge).
constexpr double kLevelShiftSkipThreshold = 1e-4;

/// @brief Apply the virtual-orbital level shift F + λ(S − S·D_occ·S).
/// @param F      Fock matrix [Hartree], n × n.
/// @param S      Overlap matrix (dimensionless), n × n.
/// @param D_occ  S-idempotent occupied density (RHF: P/2 — occupancy-1
///               convention, matching PySCF's `dm*0.5`; UHF: the per-spin
///               P_sigma, dimensionless), n × n.
/// @param lambda Level shift λ [Hartree]. |λ| <= 1e-4 is a strict no-op
///               (PySCF skip rule): F is returned element-wise unchanged.
/// @return Shifted Fock matrix [Hartree], n × n.
RealMatrix apply_level_shift(const RealMatrix& F, const RealMatrix& S,
                             const RealMatrix& D_occ, double lambda);

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
