/// @file cpu/gpw/internal/gpw_forces.h
/// @brief Analytic ionic forces for the GPW SCF (Γ RKS/UKS; k-point RKS/UKS).
///
/// Replaces the 6N+1-SCF finite-difference oracle (`gpw_forces_fd`) with a
/// single SCF solve followed by one gradient pass. At the SCF stationary point
/// the force on atom A is
///
///   F_A = -Σ_μν P_μν d(T+V_pp+V_grid)_μν/dR_A
///         + Σ_μν W_μν dS_μν/dR_A
///         + ∫ V_es(r) ∂n_c^A(r)/∂R_A dr,
///
/// with the energy-weighted density matrix
/// \(W = \sum_i f_i \varepsilon_i c_{\mu i} c_{\nu i}^*\) (closed shell
/// \(f_i=2\) fixed, or fractional under metallic smear; UKS \(f_i=1\) fixed or
/// fractional spinor occupations). All AO-derivative terms use the Gaussian
/// center-derivative identity ∂φ_a/∂A_x = 2α φ_{a+1_x} − a_x φ_{a−1_x}, so they
/// reuse the existing overlap / kinetic / GTH / collocation kernels with
/// raised/lowered Cartesian powers. The GTH operator-center derivative is
/// obtained from the basis derivatives via translational invariance
/// (∂_C = −(∂_μ + ∂_ν)). The grid term holds the converged effective potential
/// V_eff fixed (Hellmann–Feynman for the density); UKS uses spin-resolved
/// \(V_\mathrm{eff}^{\alpha/\beta}\) matching the SCF. V_es held fixed likewise
/// for the moving Gaussian ionic core n_c^A.
///
/// On a Monkhorst–Pack / explicit k mesh the assembly is k-summed with
/// Bloch-phased complex matrices and complex χ^k (see `internal/kpoints.h`);
/// fixed occupations rebuild \(P_k^\sigma\) / \(W_k^\sigma\) from
/// `GpwResult::C_occ_k` (and β); metallic free-energy forces use full
/// `C_k` + fractional `f_k` (and β) so \(P,W\) match the Mermin SCF.
///
/// Dudarev DFT+U (`cfg.plus_u`): the Mulliken occupation depends on geometry
/// through the overlap, so the +U energy contributes an occupation-matrix
/// force term \(dE_U/dR=\sum_k w_k\,\mathrm{Re\,Tr}[M_k\,dS_k]\) with
/// \(M_k^\sigma=\tfrac12(\hat V^\sigma P_k^\sigma+P_k^\sigma\hat V^\sigma)\).
/// It is folded as \(W\leftarrow W-M\) into the Pulay contraction on Γ and
/// multi-k alike (`internal/dft_plus_u.h`, `subtract_dftu_pulay[_k]`).
///
/// Reference: VandeVondele, Krack, Mohamed, Parrinello, Chassaing, Hutter,
/// Comput. Phys. Commun. 167, 103 (2005) (Quickstep GPW forces).
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h" // GpwAtom, GpwDft

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Analytic GPW forces F_a = -dE/dR_a (one SCF + one gradient pass).
///
/// Runs `GpwDft`, then assembles the Pulay (W·dS), core-Hamiltonian (P·dT,
/// P·dV_pp), grid (P·dV_grid against the converged V_eff) and ionic-core
/// Hellmann–Feynman (∫V_es ∂n_c) terms analytically. Supports closed-shell RKS,
/// Γ-point UKS, multi-k collinear UKS (`cfg.n_alpha`/`n_beta` + k mesh), and
/// multi-k RKS/UKS **free-energy** forces under metallic FD/MP smearing
/// (\(F=-\partial F_{\mathrm{Mermin}}/\partial R\) via fractional \(P_k,W_k\)).
///
/// @param atoms GPW atoms (basis + GTH PP), as for `GpwDft::run`.
/// @param cfg   SCF configuration (grid, XC, electrostatics, UKS occupations,
///        optional `smearing`/`degauss`/`extra_bands`).
/// @param guess_C Optional warm-start MO coefficients for the internal SCF
///        (e.g. the energy pass just converged at this geometry — XL-BOMD
///        path). Default nullptr keeps the cold solve bit-for-bit. UKS ignores
///        this warm-start (same as energy path).
/// @return Forces [Hartree/Bohr], one xyz triple per atom, in input order.
/// @throws std::runtime_error if the SCF does not converge.
std::vector<std::array<double, 3>>
gpw_forces_analytic(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                    const linalg::RealMatrix* guess_C = nullptr);

/// @brief Analytic forces from a **converged** `GpwResult` (no second SCF).
///
/// Product path for `GPWEvaluator` after its energy SCF: builds the
/// energy-weighted density (RKS or UKS) and runs the same assembly as
/// `gpw_forces_analytic`. Forces match the density that produced `scf.e_total`.
///
/// @param atoms Geometry + basis/PP [Bohr].
/// @param cfg   Grid / XC / occupations (must match the SCF that produced `scf`).
/// @param scf   Converged result (`converged` required; UKS needs β blocks).
/// @return Forces [Hartree/Bohr], one xyz triple per atom.
/// @throws std::runtime_error if `!scf.converged` or UKS β state is missing.
std::vector<std::array<double, 3>>
gpw_forces_from_scf(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                    const GpwResult& scf);

/// @brief Analytic ionic forces at a **fixed** electronic state (no SCF).
///
/// Car–Parrinello path: nuclei feel \(-\partial E/\partial R\) at the current
/// density. Closed-shell RKS: \(P = 2 C C^\top\),
/// \(W = 2\,C\,(C^\top H C)\,C^\top\). Collinear UKS: pass spin-resolved
/// \(P_\alpha, C_\alpha, H_\alpha\) and \(P_\beta, C_\beta, H_\beta\) via the
/// overload; \(W = W_\alpha + W_\beta\) with occupancy-1 spin channels.
/// Same force assembly as `gpw_forces_analytic` after its SCF.
///
/// @param atoms Geometry + basis/PP [Bohr].
/// @param cfg   Grid / XC / electrostatics (same functional as the energy).
/// @param P     Closed-shell AO density [e], nb × nb.
/// @param C_occ Occupied MO coefficients, nb × n_occ (S-orthonormal).
/// @param H     KS matrix at this density [Hartree], nb × nb.
/// @return Forces [Hartree/Bohr], one xyz triple per atom.
/// @throws std::invalid_argument when `cfg.is_uks()` (use the UKS overload).
std::vector<std::array<double, 3>>
gpw_forces_at_density(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                      const linalg::RealMatrix& P,
                      const linalg::RealMatrix& C_occ,
                      const linalg::RealMatrix& H);

/// @brief Fixed-density analytic forces for collinear Γ-UKS (CPMD path).
///
/// \(W_\sigma = C_\sigma (C_\sigma^\top H_\sigma C_\sigma) C_\sigma^\top\),
/// total \(W = W_\alpha + W_\beta\); assembly uses dual \(V_{\mathrm{eff}}^\sigma\).
///
/// @param atoms Geometry + basis/PP [Bohr].
/// @param cfg   Grid / XC / occupations (`is_uks()` required).
/// @param P_a   α AO density [e], nb × nb.
/// @param C_a   α occupied coefficients, nb × n_alpha (S-orthonormal).
/// @param H_a   α KS matrix [Hartree], nb × nb.
/// @param P_b   β AO density [e], nb × nb.
/// @param C_b   β occupied coefficients, nb × n_beta (S-orthonormal; empty ok if n_beta=0).
/// @param H_b   β KS matrix [Hartree], nb × nb.
/// @return Forces [Hartree/Bohr], one xyz triple per atom.
std::vector<std::array<double, 3>>
gpw_forces_at_density(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                      const linalg::RealMatrix& P_a,
                      const linalg::RealMatrix& C_a,
                      const linalg::RealMatrix& H_a,
                      const linalg::RealMatrix& P_b,
                      const linalg::RealMatrix& C_b,
                      const linalg::RealMatrix& H_b);

} // namespace gpw
} // namespace cpu
} // namespace ATV
