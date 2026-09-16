/// @file cpu/planewave/internal/restricted_scf.h
/// @brief Shared restricted fixed-occupation PW SCF loop for post-processors.
///
/// Extracted from `bands_dos.cpp` at its second caller
/// (`berry_polarization.cpp`). Runs the plain restricted SCF on
/// `method.k_mesh` and leaves the mixed density and a consistent
/// \(V_\mathrm{eff}\) on `prob` for NSCF diagonalization at arbitrary
/// k-lists (bands, DOS, Berry-phase strings).
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Restricted fixed-occupation SCF; leaves final density + V_eff on
///        `prob` (frozen for subsequent `rebuild_k_tables` + NSCF).
/// @param frame  Geometry: positions [Å], types (Z), simulation cell.
/// @param method Plane-wave method members (`ecut_ry`, `k_mesh`, `xc`, …).
/// @param prob   Problem to assemble and converge (must outlive NSCF use).
/// @return SCF result (energy [Ry]/[Ha], convergence flag, iterations).
PWResult run_restricted_scf(const cpu::Frame& frame, const PW& method,
                            PWProblem& prob);

/// @brief Collinear (nspin=2) fixed-occupation SCF for post-processors.
///
/// Requires `method.spin_polarized` and `method.smearing == none`. Leaves
/// total \(\rho\), spin XC channels, and a consistent \(V_\mathrm{eff}\) for
/// the **last** spin channel on `prob.ws`; callers that need both spins
/// re-bind `V_xc_G ← V_xc_spin_G[s]` and call `VeffBuilder::step` before
/// NSCF on channel \(s\). Occupation counts:
/// \(N_\uparrow=(N_e+M)/2\), \(N_\downarrow=(N_e-M)/2\) with
/// \(M=\mathrm{multiplicity}-1\).
///
/// @param frame  Geometry: positions [Å], types (Z), simulation cell.
/// @param method Plane-wave method (`spin_polarized` must be true).
/// @param prob   Problem to assemble and converge (must outlive NSCF use).
/// @param n_alpha Out: occupied α bands.
/// @param n_beta  Out: occupied β bands.
/// @return SCF result (energy [Ry]/[Ha] may be partial diagnostic; flags).
PWResult run_collinear_scf(const cpu::Frame& frame, const PW& method,
                           PWProblem& prob, int& n_alpha, int& n_beta);

/// @brief Collinear UKS + metallic occupation smearing for post-processors.
///
/// Serial twin of `PW::run` spin+smear: shared \(\mu\) over spin-orbitals
/// (\(f\in[0,1]\)), \(M=N_\uparrow-N_\downarrow\) free under smear, Mermin
/// free energy \(F\). Seeds \(\rho_\sigma\) from `multiplicity` then
/// iterates. Leaves total \(\rho\), \(\rho_\sigma\), and \(V_{\mathrm{xc},\sigma}\)
/// on `prob`; callers re-bind per-spin \(V_\mathrm{eff}\) for NSCF.
///
/// Requires `method.spin_polarized` and `method.smearing != none` with
/// `degauss > 0` [Ry]. Davidson subspace size is
/// \(\max(\lceil n_e/2\rceil+\mathrm{extra\_bands},\,n_\mathrm{bands\_min})\).
///
/// @param frame          Geometry [Å].
/// @param method         Plane-wave method (smear + spin members).
/// @param prob           Problem assembled and converged for NSCF.
/// @param n_bands_min    Lower bound on Davidson bands (e.g. path `n_bands`).
/// @return SCF result with `e_fermi` [Ry], free energy, magnetization.
PWResult run_collinear_smear_scf(const cpu::Frame& frame, const PW& method,
                                 PWProblem& prob, int n_bands_min = 1);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
