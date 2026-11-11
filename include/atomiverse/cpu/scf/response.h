/// @file cpu/scf/response.h
/// @brief CPHF response attached beside a mean-field method.
///
/// Transplanted from PySCF, not re-derived:
/// - `pyscf.scf.stability.rhf_stability(mf)` — orbital stability is a
///   function of the converged mean field, not a member of `scf.hf.RHF`
///   beyond a thin delegate. The eigenvalues live on `Stability<RHF>`.
/// - `pyscf.tdscf.rhf.TDA(mf).kernel()` — CIS/TDA is `tdscf.TDA`, not
///   `RHF.tda`. The class is `TDA<RHF>` (`tdhf.h`).
/// - `pyscf.hessian.rhf.Hessian(mf).kernel()` — the nuclear Hessian object
///   holds `base = mf`. `pyscf.scf.cphf.solve` is a free function, not a
///   method of `scf.hf.RHF`.
/// - Static polarizability is the electric-field instance of that same
///   solver (`¹(A+B) U^F = −r^{vo}`), not an SCF method.
///
/// `MeanFieldMethod` keeps `run` / `forces` only.
#pragma once

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

namespace ATV {
namespace cpu {
namespace scf {

struct SCFInitialGuess;

/// @brief Nuclear Hessian beside a mean-field method. Not a member of it.
///
/// Mirrors `pyscf.hessian.rhf.Hessian(mf).kernel` and
/// `pyscf.hessian.uhf.Hessian(mf).kernel`: `run` drives a fresh SCF.
/// RHF and UHF solve nuclear CPHF (`pyscf.scf.cphf.solve` is not an SCF
/// method). RKS, UKS, and ROHF differentiate analytic `forces` with
/// `central_difference_nuclear_hessian` — PySCF has no `hessian.ROHF`, and
/// the KS closed form is not this path.
/// @tparam Method `RHF`, `UHF`, `RKS`, `UKS`, or `ROHF`.
template <typename Method>
class Hessian {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method (not copied).
    explicit Hessian(Method& mf);

    /// @brief Nuclear Hessian [eV/Å²]. RHF also fills the atomic polar tensor.
    /// @param frame Geometry [Å]. Energy [eV] written to `kEnergy`.
    /// @param guess Optional density guess for the reference SCF (UHF/UKS
    ///        multi-minima). Ignored by callers that pass nullptr.
    /// @return Symmetric `[3N, 3N]` Hessian. Frequency fields stay empty.
    vib::HessianResult run(Frame& frame,
                           const SCFInitialGuess* guess = nullptr);

  private:
    Method* mf_ = nullptr;
};

/// @brief Static dipole polarizability of a converged closed-shell RHF.
///
/// Mirrors a property object on the mean field, not `RHF` itself. Call
/// `mf.run` first. `run` solves the three electric-field CPHF responses
/// and returns α_ab = ∂μ_a/∂F_b [a.u.].
/// @tparam Method Concrete mean-field type. Instantiated for `RHF`.
template <typename Method>
class Polarizability {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit Polarizability(Method& mf);

    /// @brief Static polarizability of the most recent SCF.
    /// @return Dense symmetric 3×3 tensor [a.u.], row = dipole axis,
    ///         column = field axis.
    xt::xtensor<double, 2> run() const;

  private:
    Method* mf_ = nullptr;
};

/// @brief Result of the Seeger-Pople stability analysis of a converged RHF
///        solution (Seeger & Pople, J. Chem. Phys. 66, 3045 (1977),
///        DOI 10.1063/1.434318).
///
/// Internal = RHF→RHF (singlet channel of the orbital Hessian); external =
/// RHF→UHF spin-symmetry breaking (triplet channel). The solution is stable
/// in a channel iff the lowest eigenvalue of that channel's (A + B) block is
/// non-negative. Produced by `Stability<RHF>::run`, the twin of
/// `pyscf.scf.stability.rhf_stability` — not a member of `RHF`.
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
    ///        initial guess for UHF. Equals the unrotated C when the
    ///        solution is externally stable.
    RealMatrix rotated_C;
};

/// @brief Seeger-Pople stability of a converged closed-shell RHF.
///
/// Mirrors `pyscf.scf.stability.rhf_stability(mf)`: call `mf.run` first,
/// then `run`. Not a method of `RHF` or `MeanFieldMethod`.
/// @tparam Method Concrete mean-field type. Instantiated for `RHF`.
template <typename Method>
class Stability {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit Stability(const Method& mf);

    /// @brief Lowest singlet and triplet (A + B) eigenvalues of the most
    ///        recent SCF, plus a broken-symmetry rotated coefficient matrix
    ///        when the solution is externally unstable.
    /// @return Eigenvalues [Hartree], stability flags, and `rotated_C`.
    /// @throws std::runtime_error when called before `mf.run`, or
    ///         std::invalid_argument when the system has no virtual orbitals.
    StabilityResult run() const;

  private:
    const Method* mf_ = nullptr;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
