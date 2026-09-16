/// @file cpu/scf/internal/ccsd_spatial.h
/// @brief Closed-shell spatial-orbital CCSD correlation energy (private).
///
/// The closed-shell energy entry (`MeanFieldMethod::ccsd → RHF::ccsd →
/// ccsd_energy`) routes a restricted RHF reference here. Amplitudes are kept in
/// the **spatial** MO basis — `t1[n_o, n_v]`, `t2[n_o, n_o, n_v, n_v]` (storage
/// n_o²·n_v², no spin expansion) — and the spin-summed residual is built from
/// the spatial MO ERIs of `ao2mo_full`, converging by Jacobi update accelerated
/// with `internal::CcAmplitudeDiis`. This is the fast path that reproduces the
/// spin-orbital `ccsd_solve` correlation energy on the same reference (the
/// permanent oracle-parity regression pins the equivalence).
///
/// Reference: Scuseria, Janssen, Schaefer, J. Chem. Phys. 89, 7382 (1988);
/// the closed-shell equations are those of Hirata et al., J. Chem. Phys. 120,
/// 2581 (2004) (PySCF `cc.RCCSD`), the same fixed point as the spin-orbital
/// Gauss–Stanton kernel.
#pragma once

#include "atomiverse/cpu/scf/ccsd.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell spatial-orbital CCSD correlation energy.
///
/// Spatial amplitudes \f$t_i^a\f$, \f$t_{ij}^{ab}\f$; correlation energy
/// \f$E_{\mathrm{corr}} = \sum_{ijab} L_{iajb}
///    (t_{ij}^{ab} + t_i^a t_j^b)\f$ with
/// \f$L_{iajb} = 2(ia|jb) - (ib|ja)\f$ (chemist notation). Converges when
/// \f$\max|\Delta t| < \texttt{tol}\f$.
///
/// @param ints     Integral pack with the dense ERI tensor materialized.
/// @param det      Converged closed-shell RHF determinant (MO coeffs, ε, n_e).
/// @param e_scf_ha Converged SCF total energy [Hartree].
/// @param tol      Amplitude residual tolerance (dimensionless max-norm).
/// @param max_iter Jacobi/DIIS iteration budget (dimensionless).
/// @param use_diis When true, accelerate the amplitude update with
///        `CcAmplitudeDiis`; when false, plain Jacobi (same fixed point).
/// @return CCSD correlation and total energies [Hartree], iteration count.
/// @throws std::invalid_argument on open-shell / missing ERI / non-aufbau
///         denominators / non-convergence (no silent fallback).
CCSDResult ccsd_energy_spatial(const internal::IntegralView& ints,
                               const SlaterDeterminant& det, double e_scf_ha,
                               double tol = 1e-10, int max_iter = 100,
                               bool use_diis = true);

} // namespace scf
} // namespace cpu
} // namespace ATV
