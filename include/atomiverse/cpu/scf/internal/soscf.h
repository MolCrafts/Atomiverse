/// @file cpu/scf/internal/soscf.h
/// @brief **PRIVATE** second-order SCF stepper (SOSCF): co-iterative
///        augmented-Hessian (CIAH) orbital rotations replacing the Roothaan
///        diagonalization once the DIIS error is small — quadratic
///        convergence on cases where CDIIS alone oscillates.
///
/// One `step()` = one macro Newton iteration on the orbital-rotation
/// parameters κ (dimensionless, antisymmetric occ-virt generator):
///   1. trust-region accept/reject of the PREVIOUS step (energy of the
///      current iterate vs the last accepted one; a rise restores the saved
///      coefficients and halves the radius — "retry with a smaller step"),
///   2. semicanonicalization (block-diagonal occ-occ / virt-virt rotation —
///      leaves the density and energy invariant, makes the wired Fock blocks
///      diagonal so the ε-based σ-vector kernel is exact mid-SCF),
///   3. orbital gradient g_ai = ∂E/∂κ_ai [Hartree]: 4·F_ai for the
///      closed-shell (RHF) spatial rotation, 2·F^σ_ai per spin channel for
///      UHF (concatenated α-then-β rotation vector, virtual-major
///      p = a·n_occ + i within each channel),
///   4. the CIAH augmented-Hessian eigenproblem
///        [[0, gᵀ], [g, H]] (1; κ) = λ (1; κ)
///      solved matrix-free by a deterministic Davidson — H is NEVER built:
///      RHF products go through the 08-link `OrbitalHessian` σ-kernel
///      (H = 4·¹(A+B), the true second derivative of the closed-shell
///      energy under real rotations); UHF products contract the
///      unrestricted MO-ERI blocks held here (the 08 kernel is
///      restricted-only — see the audit note in the spec),
///   5. trust-region clipping ‖κ‖₂ ≤ radius, and
///   6. the unitary update C ← C·exp(κ) (scaling-and-squaring exponential
///      of the antisymmetric generator; CᵀSC = I is preserved to 1e-12).
///
/// Follows the `DIISMixer` precedent: an internal workspace class wired into
/// the stage components — NOT a `cpu::Component`, never in a Plan. All state
/// arrives as named constructor references onto the method driver's members.
///
/// References: Sun, arXiv:1610.08423 (2016) — CIAH; Helgaker, Jørgensen,
/// Olsen, *Molecular Electronic-Structure Theory* (Wiley, 2000), ch. 10 —
/// orbital-rotation E(κ), gradient and Hessian.
#pragma once

#include <functional>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief CIAH second-order SCF stepper: owns the Davidson subspace budget,
///        the trust-region state and the cached dense working-dimension ERI;
///        `step()` rotates the wired MO coefficients in place.
class SOSCFStepper {
  public:
    /// @brief Wire the stepper to the method driver's iteration state.
    /// @param restricted True drives the single (α) closed-shell channel
    ///        (β mirrors α); false drives independent α/β rotations.
    /// @param ints Method-private integral workspace — supplies the dense
    ///        working-dimension AO ERI tensor [Hartree] (materialized lazily
    ///        on first activation, cached until `reset()`).
    /// @param n_alpha α electron count (dimensionless) [read].
    /// @param n_beta β electron count (dimensionless) [read].
    /// @param Hcore Core Hamiltonian [Hartree], n × n [read].
    /// @param v_nuc Nuclear-repulsion energy [Hartree] [read].
    /// @param F_alpha α Fock matrix [Hartree] (UNMIXED — the mix stage skips
    ///        extrapolation while SOSCF is active) [read].
    /// @param F_beta β Fock matrix [Hartree] [read].
    /// @param P_alpha α density (dimensionless; restricted convention:
    ///        factor-2 total density) [read].
    /// @param P_beta β density (dimensionless) [read].
    /// @param C_alpha α MO coefficients (dimensionless), rotated in place
    ///        [write].
    /// @param C_beta β MO coefficients (dimensionless) [write].
    /// @param eig_alpha α orbital energies [Hartree] — semicanonical while
    ///        SOSCF iterates; the driver's epilogue re-diagonalizes once
    ///        after convergence for canonical ε [write].
    /// @param eig_beta β orbital energies [Hartree] [write].
    /// @param max_micro Davidson micro-iteration budget per macro step
    ///        (dimensionless); the solve returns its best Ritz pair at the
    ///        budget (co-iteration, not failure).
    /// @param trust_radius Initial trust radius on ‖κ‖₂ (dimensionless).
    SOSCFStepper(bool restricted, const IntegralPack& ints,
                 const int& n_alpha, const int& n_beta,
                 const RealMatrix& Hcore, const double& v_nuc,
                 const RealMatrix& F_alpha, const RealMatrix& F_beta,
                 const RealMatrix& P_alpha, const RealMatrix& P_beta,
                 RealMatrix& C_alpha, RealMatrix& C_beta,
                 RealVector& eig_alpha, RealVector& eig_beta, int max_micro,
                 double trust_radius);

    /// @brief Per-run reset: drop the cached ERI tensor, restore the initial
    ///        trust radius, clear the accepted-energy history.
    void reset();

    /// @brief One macro SOSCF iteration on the wired state (the
    ///        diagonalize-stage SOSCF branch). No-op when the rotation space
    ///        is empty or the gradient is at numerical zero (< 1e-14).
    /// @throws std::invalid_argument on wired-dimension inconsistencies
    ///         (AO dimension of C vs F/Hcore, α/β mismatch — no fallback).
    void step();

    /// @brief Hartree-Fock energy of the wired (P, F) point — the same
    ///        expression as the density stage's HF branch, evaluated at
    ///        step() time for the trust-region accept/reject.
    /// @return E = ½ Σ_σ Tr[P^σ (Hcore + F^σ)] + V_nuc [Hartree]
    ///         (restricted: single term with the factor-2 total density).
    double current_energy() const;

    /// @brief Orbital-rotation gradient ∂E/∂κ in the CURRENT wired MO basis
    ///        (no semicanonicalization applied): RHF 4·F_ai single channel;
    ///        UHF 2·F^σ_ai, α block then β block, virtual-major within each.
    /// @return Concatenated gradient vector [Hartree].
    RealVector gradient() const;

    /// @brief ‖κ‖₂ actually applied by the last accepted step (after
    ///        trust-region clipping).
    /// @return Step norm (dimensionless); 0 before any step.
    double last_step_norm() const { return last_step_norm_; }

    /// @brief Lowest augmented-Hessian eigenvalue of the last solve — the
    ///        CIAH level shift ((H − λ)κ = −g, λ ≤ 0).
    /// @return λ [Hartree]; 0 before any step.
    double last_lambda() const { return last_lambda_; }

    /// @brief Current trust radius (halved on every energy-rise rejection).
    /// @return Radius bound on ‖κ‖₂ (dimensionless).
    double trust_radius() const { return radius_; }

    /// @brief Whether the last `step()` rejected the previous rotation
    ///        (energy rose): coefficients were restored, no new rotation.
    /// @return True when the last call was a reject-and-shrink.
    bool last_step_rejected() const { return rejected_; }

    /// @brief Macro SOSCF iterations executed since `reset()` — every
    ///        `step()` entry with a non-empty rotation space counts
    ///        (accepted rotations, rejects, and stationary no-ops alike:
    ///        each consumed one SCF loop iteration / Fock build, the same
    ///        unit as a PySCF `mf.newton()` macro cycle).
    /// @return Macro iteration count (dimensionless).
    int macro_steps() const { return macro_steps_; }

    /// @brief One eigenpair of the augmented Hessian [[0, gᵀ], [g, H]].
    struct AugmentedRoot {
        double lambda = 0.0; ///< Lowest eigenvalue [Hartree].
        RealVector x;        ///< Unit-norm eigenvector, length 1 + n
                             ///< (dimensionless; x(0) is the scalar row).
    };

    /// @brief Deterministic matrix-free Davidson lowest root of the
    ///        augmented Hessian [[0, gᵀ], [g, H]] — H touched only through
    ///        `h_op` (the σ-vector products; H is never formed). Fixed start
    ///        vector e₀ = (1, 0, …), diagonal preconditioning by
    ///        (0, h_diag), subspace collapse on overflow.
    /// @param g Orbital gradient, length n [Hartree].
    /// @param h_op Matrix-free Hessian product v ↦ H·v [Hartree].
    /// @param h_diag Diagonal of H, length n — Davidson preconditioner
    ///        [Hartree].
    /// @param max_iter Expansion budget; at the budget the current Ritz
    ///        pair is returned (CIAH co-iteration semantics, no throw).
    /// @return Lowest eigenpair (λ, x) with ‖x‖₂ = 1.
    /// @throws std::invalid_argument when g and h_diag lengths differ.
    static AugmentedRoot augmented_hessian_root(
        const RealVector& g,
        const std::function<RealVector(const RealVector&)>& h_op,
        const RealVector& h_diag, int max_iter);

    /// @brief Orthogonal exponential of an antisymmetric generator by
    ///        scaling-and-squaring + Taylor series (deterministic; exact
    ///        orthogonality up to round-off).
    /// @param K Antisymmetric square matrix (dimensionless rotation
    ///        generator).
    /// @return exp(K), orthogonal (UᵀU = I to machine precision).
    /// @throws std::invalid_argument when K is not square or not
    ///         antisymmetric (max|K + Kᵀ| > 1e-12 — no fallback).
    static RealMatrix exp_antisymmetric(const RealMatrix& K);

  private:
    /// @brief Validate the wired dimensions (AO dimension of C/F/Hcore per
    ///        channel, α/β consistency, occupation bounds).
    void validate_() const;

    /// @brief Semicanonicalize one spin channel in place: diagonalize the
    ///        occ-occ and virt-virt blocks of CᵀFC (block-diagonal rotation;
    ///        density-invariant), write the semicanonical ε, and return the
    ///        occ-virt gradient block of F in the rotated basis.
    /// @param F AO Fock matrix [Hartree].
    /// @param C MO coefficients, rotated in place (dimensionless).
    /// @param eig Semicanonical orbital energies out [Hartree].
    /// @param n_occ Occupied count of this channel (dimensionless).
    /// @return F_ai block (n_virt × n_occ) in the semicanonical basis
    ///         [Hartree].
    static RealMatrix semicanonicalize_(const RealMatrix& F, RealMatrix& C,
                                        RealVector& eig, int n_occ);

    /// @brief Full four-index MO transform of the dense chemist-order AO
    ///        tensor (same reshape/dot pattern as
    ///        `OrbitalHessian::transform_block_`); used for the five
    ///        unrestricted occ-virt blocks.
    static xt::xtensor<double, 4>
    mo_transform_(const xt::xtensor<double, 4>& eri_ao,
                  const xt::xtensor<double, 2>& C0,
                  const xt::xtensor<double, 2>& C1,
                  const xt::xtensor<double, 2>& C2,
                  const xt::xtensor<double, 2>& C3);

    /// @brief Apply exp(κ) to one channel: build the full antisymmetric
    ///        generator from the occ-virt amplitudes and rotate C in place.
    /// @param kappa Occ-virt amplitudes, virtual-major (dimensionless).
    /// @param n_occ Occupied count of the channel.
    /// @param C MO coefficients, rotated in place.
    static void rotate_channel_(const RealVector& kappa, int n_occ,
                                RealMatrix& C);

    bool restricted_;          ///< Closed-shell (single channel) mode.
    const IntegralPack& ints_; ///< Integral workspace (ERI source).
    const int& n_alpha_;       ///< α electron count (read).
    const int& n_beta_;        ///< β electron count (read).
    const RealMatrix& Hcore_;  ///< Core Hamiltonian [Hartree] (read).
    const double& v_nuc_;      ///< Nuclear repulsion [Hartree] (read).
    const RealMatrix& F_alpha_; ///< α Fock [Hartree] (read).
    const RealMatrix& F_beta_;  ///< β Fock [Hartree] (read).
    const RealMatrix& P_alpha_; ///< α density (read).
    const RealMatrix& P_beta_;  ///< β density (read).
    RealMatrix& C_alpha_;       ///< α MO coefficients (write).
    RealMatrix& C_beta_;        ///< β MO coefficients (write).
    RealVector& eig_alpha_;     ///< α orbital energies [Hartree] (write).
    RealVector& eig_beta_;      ///< β orbital energies [Hartree] (write).
    int max_micro_;             ///< Davidson expansion budget per macro step.
    double trust_radius0_;      ///< Initial trust radius (reset value).

    xt::xtensor<double, 4> eri_; ///< Cached dense working-dim AO ERI
                                 ///< [Hartree] (per-run, lazily built).
    double radius_ = 0.0;        ///< Current trust radius on ‖κ‖₂.
    bool have_prev_ = false;     ///< An accepted step's energy is recorded.
    double prev_energy_ = 0.0;   ///< Last ACCEPTED iterate's energy [Hartree].
    RealMatrix saved_C_alpha_;   ///< α coefficients of the accepted iterate.
    RealMatrix saved_C_beta_;    ///< β coefficients of the accepted iterate.
    RealVector saved_eig_alpha_; ///< α energies of the accepted iterate.
    RealVector saved_eig_beta_;  ///< β energies of the accepted iterate.
    double last_step_norm_ = 0.0; ///< ‖κ‖₂ of the last applied rotation.
    double last_lambda_ = 0.0;    ///< λ of the last CIAH solve [Hartree].
    bool rejected_ = false;       ///< Last call was a reject-and-shrink.
    int macro_steps_ = 0;         ///< step() entries since reset().
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
