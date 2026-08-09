/// @file cpu/pw/internal/scf_loop.h
/// @brief internal::run_scf_fixed_occ — minimal restricted fixed-occupation
///        plane-wave SCF shared by the analytic force / stress / DFPT paths.
///
/// Promoted from the (previously duplicated) file-local `run_scf` helpers in
/// `pw_forces.cpp` and `pw_stress.cpp` on their third caller (DFPT). Honors
/// `method.xc` (the historical `pw_forces.cpp` copy silently evaluated the
/// default functional). Facade-private: never on the public include path.
#pragma once

#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

class PoissonSolver;
class XCBuilder;
class VeffBuilder;
class DensityBuilder;
class DavidsonSolver;

/// @brief Copy every `PW` **parameter** into a fresh, unbuilt solver.
///
/// `PW` is move-only (it owns the step helpers), so the finite-difference
/// oracles — which need one independent solver per displaced/strained
/// geometry — cannot simply copy it. This is the single definition of
/// "the same method, no state": `pw_forces_fd` and `scf_energy_ry` (strain
/// FD) both use it, and it must list **every** public member of `PW`.
///
/// Promoted from two file-local copies that each silently dropped the newly
/// added `gth_name`, so an FD oracle asked for a pseudopotential-free run
/// and threw. One definition is the only way that stays fixed.
///
/// @param m Source method (not mutated; helpers are not shared).
/// @return Parameter-identical `PW` with `built_ == false`.
PW copy_pw_params(const PW& m);

/// @brief Run the restricted fixed-occupation SCF loop to convergence.
///
/// Assembles the problem via `setup_pw_problem` and iterates
/// Poisson → XC (`method.xc`) → Veff → per-k Davidson → density + Pulay mix.
/// On exit `prob` holds the final mixed density on the Γ sphere; k-points are
/// distributed round-robin over MPI ranks when MPI is initialized.
///
/// @param frame    Geometry [Å]; not mutated.
/// @param method   Plane-wave method members (`ecut_ry`, `k_mesh`, `xc`, …).
/// @param prob     Problem storage to fill (must outlive the caller's use).
/// @param poisson  Hartree step helper.
/// @param xc       XC step helper.
/// @param veff     Effective-potential step helper.
/// @param density  Density accumulation / Pulay mixing helper.
/// @param davidson Band solver (caller chooses band count / tolerance).
/// @param tag      Short label for the rank-0 convergence log line.
/// @return Energy [Ry]/[Hartree], convergence flag, iteration count.
PWResult run_scf_fixed_occ(const cpu::Frame& frame, const PW& method,
                           PWProblem& prob, PoissonSolver& poisson,
                           XCBuilder& xc, VeffBuilder& veff,
                           DensityBuilder& density, DavidsonSolver& davidson,
                           const char* tag);

/// @brief Restricted SCF with metallic occupation smearing (FD / MP / Gaussian).
///
/// Same loop shape as `run_scf_fixed_occ`, but finds a shared \(\mu\) every
/// iteration and builds the density with fractional band occupations in
/// \([0,2]\). Requires `method.smearing != none`, `method.degauss > 0`, and
/// `davidson` sized to \(\lceil N_e/2\rceil + \mathrm{extra\_bands}\).
/// Collinear spin is not supported here (use `PW::run`).
///
/// @param n_bands  Band count (must match the Davidson solver).
/// @return Free energy \(F=E_{\mathrm{KS}}-TS\) [Ry]/[Hartree], flags, iters.
PWResult run_scf_smear_restricted(const cpu::Frame& frame, const PW& method,
                                  PWProblem& prob, PoissonSolver& poisson,
                                  XCBuilder& xc, VeffBuilder& veff,
                                  DensityBuilder& density,
                                  DavidsonSolver& davidson, int n_bands,
                                  const char* tag);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
