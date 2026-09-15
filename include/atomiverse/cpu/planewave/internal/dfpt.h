/// @file cpu/planewave/internal/dfpt.h
/// @brief DFPT machinery for the plane-wave method (Γ and finite q): Ewald
///        force constants, Sternheimer conduction-space CG, and assembly of
///        the Cartesian dynamical matrix from self-consistent first-order
///        response (Baroni, de Gironcoli, Dal Corso, Giannozzi,
///        Rev. Mod. Phys. 73, 515 (2001)).
///
/// Facade-private: owned by `cpu::pw::DfptPhonon`, never on the public
/// include path and never parked on Frame.
#pragma once

#include <complex>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

#include <fftw3.h>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Analytic q = 0 ion–ion (Ewald) force-constant matrix.
///
/// Second derivatives of the Ewald energy used by the PW total energy
/// (same α selection, real/reciprocal cutoffs as `EwaldSummation`); the
/// self term and the G = 0 jellium term are position-independent.
///
/// @param pos_ang Positions [Å].
/// @param charges Point charges (valence Z for UPF).
/// @param cell    Simulation cell [Å].
/// @param ecut_hartree Wavefunction cutoff [Hartree] (sets recip G cut).
/// @return Row-major `3N × 3N` matrix [Hartree/Bohr²],
///         index `3·I + α`.
std::vector<AccReal> ewald_dynamical_matrix(const std::vector<Vec3>& pos_ang,
                                            const std::vector<AccReal>& charges,
                                            const cpu::Box& cell,
                                            AccReal ecut_hartree);

/// @brief Analytic finite-q ion–ion (Ewald) dynamical matrix \(C(q)\).
///
/// Same α / cutoffs as `ewald_dynamical_matrix`. At \(q = 0\) the result is
/// real and matches the Γ matrix to roundoff. Output is row-major complex
/// Hermitian `3N × 3N` [Hartree/Bohr²].
///
/// @param pos_ang Positions [Å].
/// @param charges Point charges (valence Z for UPF).
/// @param cell    Simulation cell [Å].
/// @param ecut_hartree Wavefunction cutoff [Hartree] (sets recip G cut).
/// @param q_frac  Phonon wavevector in fractional crystal coordinates.
std::vector<std::complex<AccReal>> ewald_dynamical_matrix_q(
    const std::vector<Vec3>& pos_ang, const std::vector<AccReal>& charges,
    const cpu::Box& cell, AccReal ecut_hartree, const Vec3& q_frac);

/// @brief Conduction-space projected preconditioned CG for the Sternheimer
///        equation `P_c (H − ε) P_c x = P_c b` at the bound k-sphere of `ws`.
///
/// `H` is the same T_G + V_eff(r) + KB Hamiltonian the Davidson solver
/// applies. The occupied manifold `occ_wfc` (row-major `n_occ × n_g`,
/// orthonormal) defines `P_c = 1 − Σ_j |ψ_j⟩⟨ψ_j|`. Positive definite on
/// the conduction space for insulating fillings (ε an occupied eigenvalue).
///
/// @param ws       Workspace with `gvec`/`n_g`/`T_G`/`V_eff_r`/`kb_proj` bound.
/// @param occ_wfc  Occupied wavefunctions at this k [G-normalized].
/// @param n_occ    Number of occupied bands.
/// @param eps      Eigenvalue of the band being perturbed [Hartree].
/// @param rhs      Right-hand side `b` (projected internally).
/// @param x        Solution (in/out; entry value is the initial guess).
/// @param tol      Residual L2 tolerance.
/// @param max_iter CG iteration cap.
/// @param work_buf Caller-owned FFT scratch of size `ws.nr`.
/// @return Iterations taken (max_iter + 1 if not converged).
int sternheimer_cg(const PWWorkspace& ws, const std::complex<AccReal>* occ_wfc,
                   int n_occ, AccReal eps, const std::complex<AccReal>* rhs,
                   std::complex<AccReal>* x, AccReal tol, int max_iter,
                   fftw_complex* work_buf);

/// @brief Result of the Γ-point DFPT force-constant assembly.
struct DfptGammaOutput {
    /// Cartesian force constants [Hartree/Bohr²], row-major `3N × 3N`,
    /// symmetrized; index `3·I + α`.
    std::vector<AccReal> force_constants;
    bool scf_converged = false;      ///< Ground-state SCF converged.
    bool response_converged = false; ///< All 3N response SCFs converged.
    /// Max |C − Cᵀ|/2 before symmetrization [Hartree/Bohr²] (diagnostic).
    AccReal max_asymmetry = AccReal(0);
};

/// @brief Result of a finite-q DFPT force-constant assembly.
struct DfptQOutput {
    /// Complex Hermitian Cartesian force constants [Hartree/Bohr²],
    /// row-major `3N × 3N`, Hermitian-projected; index `3·I + α`.
    std::vector<std::complex<AccReal>> force_constants;
    bool scf_converged = false;
    bool response_converged = false;
    /// Max |C − C†|/2 before projection [Hartree/Bohr²].
    AccReal max_asymmetry = AccReal(0);
};

/// @brief Assemble the q = 0 DFPT force-constant matrix for `frame`.
///
/// Pure LSDA or pure GGA with a UPF pseudopotential (throws
/// `std::invalid_argument` otherwise). Restricted **fixed occupations** or
/// **metallic smearing** (`method.smearing` + `degauss` + `extra_bands`), or
/// **collinear spin** with fixed occupations or metallic smearing
/// (`method.spin_polarized` + optional smearing). Runs the ground-state
/// SCF, then one
/// self-consistent Sternheimer response per Cartesian displacement with
/// occupations \(f_{nk}\) weighting the density and force-constant terms.
/// Pure GGA uses the spectral `XCBuilder` kernel (FD of \(V_{\mathrm{xc}}[n]\)
/// or `step_spin`); hybrids are rejected.
///
/// @param frame  Geometry [Å], types (Z), cell.
/// @param method Plane-wave method members (`xc` pure LSDA or pure GGA).
/// @param response_tol      Δn L2 convergence per perturbation.
/// @param response_max_iter Response SCF iteration cap.
/// @param cg_tol            Sternheimer CG residual tolerance.
/// @param cg_max_iter       Sternheimer CG iteration cap.
/// @param mix_beta          Linear mixing for the response density.
DfptGammaOutput
dfpt_gamma_force_constants(const cpu::Frame& frame, const PW& method,
                           AccReal response_tol, int response_max_iter,
                           AccReal cg_tol, int cg_max_iter, AccReal mix_beta);

/// @brief Assemble the finite-q DFPT force-constant matrix \(C(q)\).
///
/// Same scope as `dfpt_gamma_force_constants` for pure LSDA / pure GGA:
/// restricted **fixed occupations** or **metallic smearing**, or
/// **collinear spin** with fixed occupations or metallic smearing
/// (shared \(\mu\), spinor \(f\in[0,1]\)). Fractional \(f_{nk}\) (or
/// \(f_{nk\sigma}\)) weight the density response and force-constant
/// terms; the Sternheimer projector at \(\mathbf{k}+\mathbf{q}\) removes
/// bands with \(f\ge 10^{-8}\). No de Gironcoli \(\delta\mu\) at
/// \(q\neq 0\). When `q_frac` is the zero vector the result is real and
/// matches the Γ path (use the Γ entry for production Γ runs — this path
/// always builds k and k+q tables).
///
/// @param q_frac Phonon wavevector in fractional crystal coordinates.
DfptQOutput dfpt_q_force_constants(const cpu::Frame& frame, const PW& method,
                                   const Vec3& q_frac, AccReal response_tol,
                                   int response_max_iter, AccReal cg_tol,
                                   int cg_max_iter, AccReal mix_beta);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
