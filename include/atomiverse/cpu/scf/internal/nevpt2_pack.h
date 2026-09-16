/// @file cpu/scf/internal/nevpt2_pack.h
/// @brief Private pack-based SC-NEVPT2 entry (CASSCF + strongly contracted PT2).
#pragma once

#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/nevpt2.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell strongly contracted NEVPT2 energy on a CASSCF reference.
///
/// Runs CASSCF (dense active-space CI + orbital Newton), then evaluates the
/// eight SC-NEVPT2 perturber classes (Sr, Si, Sij, Srs, Sir, Sijr, Srsi, Sijrs).
/// When ncore = nvirt = 0 every class contributes identically zero (no external
/// correlation space), matching PySCF `mrpt.NEVPT` on full-valence CAS(2,2)
/// H₂/STO-3G.
///
/// @param ints     Integral pack with the ERI tensor materialized.
/// @param det      Converged closed-shell reference determinant.
/// @param e_scf_ha Converged SCF total energy [Hartree] (for CASSCF e_corr).
/// @param ncas     Number of active spatial orbitals (dimensionless).
/// @param nelecas  Number of active electrons (even; closed-shell).
/// @param orb_tol  CASSCF orbital-gradient convergence [Hartree].
/// @param max_macro Maximum CASSCF macro-iterations.
/// @return NEVPT2 total / PT2 correlation [Hartree] and bookkeeping.
/// @throws std::invalid_argument on open-shell, missing ERI, invalid active
///         space, or non-empty external space that requires 3-RDM intermediates
///         not yet implemented on this foothold path (no silent fallback).
NEVPT2Result nevpt2_energy(const internal::IntegralView& ints,
                           const SlaterDeterminant& det, double e_scf_ha,
                           int ncas, int nelecas, double orb_tol = 1e-8,
                           int max_macro = 50);

} // namespace scf
} // namespace cpu
} // namespace ATV
