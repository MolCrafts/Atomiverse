/// @file cpu/scf/rhf.h
/// @brief Restricted Hartree–Fock (RHF) concrete mean-field method.
#pragma once

#include <string>

#include "atomiverse/cpu/scf/mean_field_method.h"
#include "atomiverse/cpu/scf/tdhf.h"
#include "atomiverse/cpu/vib/hessian_result.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Result of the Seeger-Pople stability analysis of a converged RHF
///        solution (Seeger & Pople, J. Chem. Phys. 66, 3045 (1977),
///        DOI 10.1063/1.434318).
///
/// Internal = RHF→RHF (singlet channel of the orbital Hessian); external =
/// RHF→UHF spin-symmetry breaking (triplet channel). The solution is stable
/// in a channel iff the lowest eigenvalue of that channel's (A + B) block is
/// non-negative.
struct StabilityResult {
    /// @brief Lowest eigenvalue of the singlet (A + B) block [Hartree].
    double internal_eigenvalue = 0.0;
    /// @brief Lowest eigenvalue of the triplet (A + B) block [Hartree].
    double external_eigenvalue = 0.0;
    /// @brief Internal (RHF→RHF) stability: internal_eigenvalue ≥ 0 up to
    ///        the numerical-zero guard (dimensionless flag).
    bool internal_stable = true;
    /// @brief External (RHF→UHF triplet) stability: external_eigenvalue ≥ 0
    ///        up to the numerical-zero guard (dimensionless flag).
    bool external_stable = true;
    /// @brief MO coefficients rotated along the external instability mode
    ///        [n × n_mo, dimensionless] — a broken-symmetry α-channel
    ///        initial guess for UHF (bridge to `scf_guess.h`: build a
    ///        per-spin density P_α = C_occ C_occᵀ from these columns and
    ///        pass it via `SCFInitialGuess`). Equals the unrotated C when
    ///        the solution is externally stable.
    RealMatrix rotated_C;
};

/// @brief Restricted Hartree–Fock — a mean-field electronic-structure method
///        (a single Slater determinant in which every electron moves in the
///        averaged field of all the others) whose spin-up (α) and spin-down (β)
///        electrons share one common set of spatial orbitals: the closed-shell,
///        fully paired case.
class RHF : public MeanFieldMethod<RHF> {
  public:
    /// @brief Construct with the RHF defaults (restricted HF; MINAO initial
    ///        guess; no density damping; energy tolerance 1e-9 Ha).
    RHF();
    std::string name() const { return "rhf"; }

    /// @brief Seeger-Pople stability analysis of the most recent `run()`:
    ///        lowest eigenvalue of the singlet (internal, RHF→RHF) and
    ///        triplet (external, RHF→UHF) (A + B) orbital-Hessian blocks,
    ///        plus a broken-symmetry rotated coefficient matrix when the
    ///        solution is externally unstable. The dense working-dimension
    ///        ERI tensor is rebuilt from the private shell workspace.
    /// @return Eigenvalues [Hartree], stability flags, and `rotated_C`.
    /// @throws std::runtime_error when called before `run()`, or
    ///         std::invalid_argument when the system has no virtual
    ///         orbitals (e.g. He/STO-3G — no fallback).
    StabilityResult stability() const;

    /// @brief Tamm-Dancoff excitation energies (CIS on the HF reference) of
    ///        the most recent `run()`: the lowest `n_states` roots of the
    ///        Hermitian A X = ωX eigenproblem in the requested spin channel
    ///        — the link-08 orbital-Hessian A/B kernel's fourth
    ///        configuration (after stability, SOSCF, and CPHF). The TDA
    ///        spectrum stays real on any reference, including ones carrying
    ///        a triplet instability (Hirata & Head-Gordon 1999,
    ///        DOI 10.1016/S0009-2614(99)01149-5). The dense
    ///        working-dimension ERI tensor is rebuilt from the private
    ///        shell workspace (stability() precedent).
    /// @param n_states Number of lowest excited states (1 ≤ n_states ≤
    ///                 n_occ·n_virt; dimensionless count).
    /// @param channel  Spin channel: singlet roots carry length-gauge
    ///                 transition dipoles [a.u.] and oscillator strengths
    ///                 f = (2/3)·ω[Ha]·|μ|²; triplet roots are
    ///                 spin-forbidden (μ = 0, f = 0).
    /// @return Excitation energies [eV], transition dipoles [a.u.],
    ///         oscillator strengths (dimensionless), `tda = true`.
    /// @throws std::runtime_error when called before `run()`.
    /// @throws std::invalid_argument when n_states is outside
    ///         [1, n_occ·n_virt] or the system has no virtual orbitals
    ///         (e.g. He/STO-3G — no fallback).
    ExcitedStatesResult
    tda(int n_states,
        ExcitationChannel channel = ExcitationChannel::singlet) const;

    /// @brief Full TDHF/RPA excitation energies of the most recent `run()`:
    ///        [[A,B],[−B,−A]](X;Y) = ω(X;Y), solved via the Hermitian
    ///        reduced form (A−B)^{1/2}(A+B)(A−B)^{1/2} Z = ω²Z (Dreuw &
    ///        Head-Gordon 2005, DOI 10.1021/cr0505627; Szabo & Ostlund §7).
    ///        Runs the link-08 stability analysis of the requested
    ///        channel's (A + B) block first: an unstable reference makes ω
    ///        imaginary in that channel, so the call throws with the lowest
    ///        eigenvalue instead of solving (use `tda()` — real on any
    ///        reference — or reconverge along `stability().rotated_C`).
    /// @param n_states Number of lowest excited states (1 ≤ n_states ≤
    ///                 n_occ·n_virt; dimensionless count).
    /// @param channel  Spin channel (transition properties as in `tda()`).
    /// @return Excitation energies [eV] (ω_RPA ≤ ω_TDA root by root),
    ///         transition dipoles [a.u.], oscillator strengths, `tda = false`.
    /// @throws std::runtime_error when called before `run()`, when the
    ///         requested channel's (A + B) block is unstable (message
    ///         carries the lowest eigenvalue [Hartree] and the guidance
    ///         above), or when (A − B) is not positive definite — no
    ///         fallback values.
    /// @throws std::invalid_argument as `tda()`.
    ExcitedStatesResult
    tdhf(int n_states,
         ExcitationChannel channel = ExcitationChannel::singlet) const;

    /// @brief Analytic nuclear Hessian d²E/dR² via CPHF linear response —
    ///        a method output beside `forces()`, feeding the existing vib
    ///        pipeline (`vib::MassWeighting` → `vib::HessianDiagonalizer`).
    ///
    /// Runs a fresh SCF on `frame` (writes the eV energy to
    /// `frame.meta[kEnergy]`), assembles the skeleton second-derivative
    /// integrals, solves the CPHF equations ¹(A+B)·U^x = −B^x for all 3N
    /// nuclear perturbations (the LHS is the link-08 `OrbitalHessian`
    /// singlet σ₊ kernel — its third configuration after stability and
    /// SOSCF), and assembles E^{xy} per Pople-Krishnan-Schlegel-Binkley
    /// 1979 (DOI 10.1002/qua.560160825).
    ///
    /// @param frame Geometry (Å) + Z in the type column; atoms not moved.
    /// @return `vib::HessianResult` with the dense symmetric `[3N, 3N]`
    ///         Hessian in eV/Å² and `n_atoms` set; the frequency /
    ///         thermochemistry fields stay empty for the vib components.
    /// @throws std::runtime_error when the SCF does not converge, on a
    ///         spherical-convention basis or ECP pack (derivative
    ///         integrals not wired — no fallback), with RI enabled, or on
    ///         CPHF non-convergence (message carries the iteration count).
    /// @throws std::invalid_argument when the system has no virtual
    ///         orbitals (e.g. He/STO-3G — no occ→virt response exists).
    vib::HessianResult hessian(Frame& frame);

  protected:
    /// @brief Closed-shell occupation: n_alpha = n_beta = n_electrons / 2
    ///        (every occupied spatial orbital holds one α and one β electron).
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
