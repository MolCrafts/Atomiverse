/// @file cpu/gpw/internal/gpw_forces.h
/// @brief Analytic ionic forces for the closed-shell GPW SCF.
///
/// Replaces the 6N+1-SCF finite-difference oracle (`gpw_forces_fd`) with a
/// single SCF solve followed by one gradient pass. At the SCF stationary point
/// the force on atom A is
///
///   F_A = -Σ_μν P_μν d(T+V_pp+V_grid)_μν/dR_A
///         + Σ_μν W_μν dS_μν/dR_A
///         + ∫ V_es(r) ∂n_c^A(r)/∂R_A dr,
///
/// with the energy-weighted density matrix W = Σ_i 2 ε_i c_{μi} c_{νi} (closed
/// shell). All AO-derivative terms use the Gaussian center-derivative identity
/// ∂φ_a/∂A_x = 2α φ_{a+1_x} − a_x φ_{a−1_x}, so they reuse the existing overlap
/// / kinetic / GTH / collocation kernels with raised/lowered Cartesian powers.
/// The GTH operator-center derivative is obtained from the basis derivatives via
/// translational invariance (∂_C = −(∂_μ + ∂_ν)). The grid term holds the
/// converged effective potential V_eff fixed (Hellmann–Feynman for the density).
/// V_es held fixed likewise for the moving Gaussian ionic core n_c^A.
///
/// Reference: VandeVondele, Krack, Mohamed, Parrinello, Chassaing, Hutter,
/// Comput. Phys. Commun. 167, 103 (2005) (Quickstep GPW forces).
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h" // GpwAtom, GpwConfig

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Analytic GPW forces F_a = -dE/dR_a (one SCF + one gradient pass).
///
/// Runs `GPWSolver`, then assembles the Pulay (W·dS), core-Hamiltonian (P·dT,
/// P·dV_pp), grid (P·dV_grid against the converged V_eff) and ionic-core
/// Hellmann–Feynman (∫V_es ∂n_c) terms analytically.
///
/// @param atoms GPW atoms (basis + GTH PP), as for `GPWSolver::run`.
/// @param cfg   SCF configuration (grid, XC, electrostatics).
/// @return Forces [Hartree/Bohr], one xyz triple per atom, in input order.
/// @throws std::runtime_error if the SCF does not converge.
std::vector<std::array<double, 3>>
gpw_forces_analytic(const std::vector<GpwAtom>& atoms, const GpwConfig& cfg);

} // namespace gpw
} // namespace cpu
} // namespace ATV
