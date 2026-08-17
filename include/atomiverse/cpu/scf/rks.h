/// @file cpu/scf/rks.h
/// @brief Restricted Kohn–Sham (RKS) concrete mean-field method.
#pragma once

#include <string>

#include "atomiverse/cpu/scf/mean_field_method.h"
#include "atomiverse/cpu/scf/tdhf.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Restricted Kohn–Sham — Kohn–Sham density-functional theory (a
///        non-interacting orbital reference plus an exchange–correlation
///        functional that supplies the many-body effects) in its closed-shell
///        form, with spin-up (α) and spin-down (β) electrons sharing one set of
///        spatial orbitals. Defaults to the PBE functional on the MatchPySCF
///        integration grid.
class RKS : public MeanFieldMethod<RKS> {
  public:
    XCType xc = XCType::PBE; ///< Exchange–correlation functional (default PBE).

    /// @brief Construct with the RKS defaults (restricted KS; PBE functional;
    ///        MINAO initial guess; no damping; MatchPySCF grid; energy
    ///        tolerance 1e-9 Ha).
    RKS();
    /// @brief Construct RKS with an explicit exchange–correlation functional.
    /// @param xc Functional to use (e.g. LSDA, PBE, BLYP, B3LYP, PBE0,
    ///           CAMB3LYP, B2PLYP). For double hybrids, call `mp2(frame)` with
    ///           `mp2_os_scale = mp2_ss_scale = hybrid_pt2_fraction(xc)`
    ///           after/for the PT2 term (KS SCF alone is not the full energy).
    explicit RKS(XCType xc);
    std::string name() const { return "rks"; }

    /// @brief Vertical TDA excitation energies of the most recent `run()`.
    ///
    /// Casida TDA on the converged KS orbitals:
    /// - **LSDA:** full MO-product ALDA kernel
    ///   \((ia|f^{\mathrm{xc}}|jb)=\int\phi_i\phi_a\,f_{\mathrm{ALDA}}(n)\,
    ///   \phi_j\phi_b\,d\mathbf{r}\) (Dirac-exchange second derivative;
    ///   Casida 1995) — `ExcitedStatesResult::alda = true`.
    /// - **Global hybrid (B3LYP / PBE0):** exact-exchange fraction
    ///   \(\alpha_{\mathrm{hf}}\) on the Casida **A** matrix
    ///   (\(^{1}A=\delta\delta\Delta\varepsilon+2(ia|jb)-\alpha_{\mathrm{hf}}(ij|ab)\))
    ///   plus the semi-local \(f^{\mathrm{xc}}\) kernel from a finite
    ///   difference of \(V_{\mathrm{xc}}\) on the KS grid (Dreuw &
    ///   Head-Gordon, Chem. Rev. 105, 4009 (2005)) —
    ///   `alda = false`.
    /// - **Pure GGA (PBE / BLYP):** HF-like CIS@RKS fallback (`alda = false`).
    /// @param n_states Number of lowest excited states.
    /// @param channel  Singlet or triplet channel.
    /// @return Excitation energies [eV], transition properties;
    ///         `alda == true` for the LSDA full-grid path only.
    ExcitedStatesResult
    tda(int n_states,
        ExcitationChannel channel = ExcitationChannel::singlet) const;

    /// @brief Full linear-response TDDFT (Casida A,B / RPA) of the most
    ///        recent `run()` — **not** Tamm–Dancoff.
    ///
    /// Builds the closed-shell Casida matrices with the functional's exact-
    /// exchange fraction \(\alpha_{\mathrm{hf}}\) and the adiabatic
    /// semi-local \(f^{\mathrm{xc}}\) kernel (same kernel on A and B), then
    /// solves the Hermitian reduced RPA eigenproblem
    /// \((A-B)^{1/2}(A+B)(A-B)^{1/2} Z = \omega^2 Z\) (Casida 1995; Dreuw &
    /// Head-Gordon, Chem. Rev. 105, 4009 (2005)).
    ///
    /// - **LSDA:** full MO-product ALDA \(f^{\mathrm{xc}}\) (`alda = true`).
    /// - **Pure GGA (PBE / BLYP) and global hybrids (B3LYP / PBE0):**
    ///   FD of \(V_{\mathrm{xc}}\) on the KS grid for \(f^{\mathrm{xc}}\)
    ///   (`alda = false`); hybrids also scale the HF exchange blocks by
    ///   \(\alpha_{\mathrm{hf}}\).
    ///
    /// Dense path only (`n_ov ≤ 512`). Singlet channel is the calibrated
    /// production path (triplet non-LSDA omits the FD kernel).
    /// @param n_states Number of lowest excited states (1 ≤ n_states ≤ n_ov).
    /// @param channel  Singlet or triplet channel.
    /// @return Excitation energies [eV], transition properties;
    ///         `tda = false`; `alda` true only on the LSDA path.
    /// @throws std::runtime_error when called before `run()`, when
    ///         `n_ov > 512`, when (A−B) is not positive definite, or when a
    ///         reduced ω² is negative (imaginary root) — no fallback.
    /// @throws std::invalid_argument when n_states is outside [1, n_ov].
    ExcitedStatesResult
    tddft(int n_states,
          ExcitationChannel channel = ExcitationChannel::singlet) const;

    /// @brief Nuclear Hessian d²E/dR² of the RKS energy [eV/Å²].
    ///
    /// Central finite difference of the **analytic** RKS nuclear forces
    /// (`forces()` / KS gradient with grid-response XC). This is the
    /// correct second derivative of the RKS energy under the converged
    /// KS density; a fully analytic CPKS Hessian (PySCF `hessian.RKS`
    /// closed form) is a follow-up. Step \(h = 10^{-4}\) Bohr; the
    /// assembled matrix is symmetrized. Same algorithm class as
    /// `ROHF::hessian`.
    ///
    /// @param frame Geometry (Å) + Z in the type column; restored on return.
    /// @return `vib::HessianResult` with the dense symmetric `[3N, 3N]`
    ///         Hessian in eV/Å² and `n_atoms` set.
    /// @throws std::runtime_error when the reference or displaced SCF
    ///         does not converge, or with RI enabled.
    vib::HessianResult hessian(Frame& frame);

  protected:
    /// @brief Closed-shell occupation (n_alpha = n_beta = n_electrons / 2);
    ///        also latches the selected `xc` functional into the method state.
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
