/// @file cpu/scf/tdhf.h
/// @brief Excited-state result types of the TDHF/RPA and TDA (CIS) solvers
///        (public, result-only header — mp2.h precedent).
///
/// User path: `TDA<RHF>{mf}.run` and `TDHF<RHF>{mf}.run` after `mf.run`
/// (`pyscf.tdscf.rhf.TDA` / `TDHF`). Neither is a member of `RHF`.
/// KS linear response is `TDA<RKS>` / `TDDFT<RKS>` / `TDA<UKS>` /
/// `TDDFT<UKS>` (`pyscf.tdscf.rks` / `uks`). Spin-flip is `SfTDA<UHF>` /
/// `SfTDHF<UHF>`. The eigenproblem machinery is private
/// (`internal::TDHFPack` over the link-08 `internal::OrbitalHessian` A/B
/// kernel) and is not part of this header.
///
/// References: Dreuw & Head-Gordon, Chem. Rev. 105, 4009 (2005),
/// DOI 10.1021/cr0505627 (TDHF/RPA response formalism); Hirata &
/// Head-Gordon, Chem. Phys. Lett. 314, 291 (1999),
/// DOI 10.1016/S0009-2614(99)01149-5 (TDA); Szabo & Ostlund §7.
#pragma once

#include <vector>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
class Frame;
namespace scf {

/// @brief Spin symmetry of the excitation out of a closed-shell reference
///        (dimensionless selector): `singlet` (spin-conserving, dipole-
///        allowed) or `triplet` (spin-flip manifold, dipole-forbidden from
///        a singlet ground state).
enum class ExcitationChannel { singlet, triplet };

/// @brief Excitation energies and length-gauge transition properties of a
///        TDA or TDHF/RPA solve on a converged closed-shell reference.
struct ExcitedStatesResult {
    /// @brief Excitation energies ω_0n [eV], ascending.
    std::vector<double> energy_ev;

    /// @brief Length-gauge transition dipole moments μ_0n [a.u., e·Bohr],
    ///        shape [n_states, 3]. Singlet channel:
    ///        μ_0n = √2 Σ_ia (X+Y)_ia ⟨φ_i|r|φ_a⟩ (overall sign is the
    ///        arbitrary eigenvector phase); triplet channel: exactly zero
    ///        (spin-forbidden from the singlet ground state).
    xt::xtensor<double, 2> transition_dipole_au;

    /// @brief Length-gauge oscillator strengths (dimensionless):
    ///        f_0n = (2/3)·ω_0n[Ha]·|μ_0n|²[a.u.]. Zero for the triplet
    ///        channel.
    std::vector<double> oscillator_strength;

    /// @brief Which spin channel was solved.
    ExcitationChannel channel = ExcitationChannel::singlet;

    /// @brief true = Tamm-Dancoff approximation (B = 0; CIS on an HF
    ///        reference), false = full TDHF/RPA (dimensionless flag).
    bool tda = false;

    /// @brief true when full MO-product ALDA \(f^{\mathrm{xc}}\) was applied
    ///        to the Casida A matrix (RKS LSDA TDA path).
    bool alda = false;
};

/// @brief Tamm–Dancoff (CIS on HF) excitations of a converged closed-shell
///        reference.
///
/// Mirrors `pyscf.tdscf.rhf.TDA(mf)`: `run` is `kernel`. Call `mf.run`
/// first. Not a method of `RHF` or `MeanFieldMethod`. RKS / UKS use the
/// same type with their own kernel (`TDA<RKS>` / `TDA<UKS>`).
/// @tparam Method Concrete mean-field type. Instantiated for `RHF`.
template <typename Method>
class TDA {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit TDA(const Method& mf);

    /// @brief Lowest `n_states` roots of the Hermitian A X = ωX problem.
    /// @param n_states Number of lowest excited states (1 ≤ n_states ≤
    ///                 n_occ·n_virt; dimensionless count).
    /// @param channel  Spin channel. Singlet roots carry length-gauge
    ///                 transition dipoles [a.u.] and oscillator strengths;
    ///                 triplet roots are spin-forbidden (μ = 0, f = 0).
    /// @return Excitation energies [eV], transition dipoles [a.u.],
    ///         oscillator strengths, `tda = true`.
    /// @throws std::runtime_error when called before `mf.run`.
    /// @throws std::invalid_argument when n_states is outside
    ///         [1, n_occ·n_virt] or the system has no virtual orbitals.
    ExcitedStatesResult
    run(int n_states,
        ExcitationChannel channel = ExcitationChannel::singlet) const;

    /// @brief Nuclear forces on TDA root `state` [eV/Å].
    ///
    /// Closed-shell RHF only (`TDA<RHF>`). Default path is analytic CIS
    /// (`pyscf.grad.tdrhf` shape: integral derivatives + Z-vector). Set
    /// `analytic` false for a central difference of the TDA energy with
    /// step `h` [Å].
    /// @param frame Geometry [Å]. Mutated and restored on the FD path.
    /// @param state Excited root index (0-based).
    /// @param h Finite-difference step [Å]. Ignored when `analytic` is true.
    /// @param analytic True (default) for the analytic CIS gradient.
    /// @return Per-atom forces, 3N interleaved (fx, fy, fz) [eV/Å].
    std::vector<double> forces(cpu::Frame& frame, int state = 0,
                               double h = 1e-3, bool analytic = true) const;

    /// @brief First-order NAC vector d_ij = ⟨Ψ_i|∇_R|Ψ_j⟩ [1/Å].
    ///
    /// Chernyak–Mukamel form from the TDA force difference over the energy
    /// gap. Phase fixed so the first large component of the ordered pair
    /// is positive. Closed-shell RHF only.
    /// @param frame Geometry [Å].
    /// @param state_i Root i (0-based).
    /// @param state_j Root j (0-based). Must differ from i.
    /// @return d_ij [1/Å], length 3N. d_ji = −d_ij.
    std::vector<double> nac_vectors(cpu::Frame& frame, int state_i = 0,
                                    int state_j = 1) const;

    /// @brief NAC strength ‖d_ij‖ [1/Å].
    /// @param frame Geometry [Å].
    /// @param state_i Lower root index when the caller orders them.
    /// @param state_j Upper root index.
    /// @param h Unused. Kept so existing call sites that passed an FD step
    ///        still compile; the vector uses the fixed 2e-3 Å FD forces.
    /// @return Euclidean norm of `nac_vectors` [1/Å].
    double nac_strength(cpu::Frame& frame, int state_i = 0, int state_j = 1,
                        double h = 2e-3) const;

  private:
    const Method* mf_ = nullptr;
};

/// @brief Full TDHF/RPA excitations of a converged closed-shell RHF.
///
/// Mirrors `pyscf.tdscf.rhf.TDHF(mf).kernel`. Call `mf.run` first. Not a
/// method of `RHF`. An unstable (A+B) block throws (`pyscf.scf.cphf` is
/// not involved; the check is `pyscf.scf.stability` on that channel).
/// @tparam Method Concrete mean-field type. Instantiated for `RHF`.
template <typename Method>
class TDHF {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit TDHF(const Method& mf);

    /// @brief Lowest `n_states` RPA roots.
    /// @param n_states Number of roots (1 ≤ n_states ≤ n_occ·n_virt).
    /// @param channel Singlet or triplet.
    /// @return Excitation energies [eV], transition dipoles [a.u.],
    ///         oscillator strengths, `tda = false`.
    /// @throws std::runtime_error before `mf.run`, on an unstable (A+B)
    ///         block, or when (A−B) is not positive definite.
    ExcitedStatesResult
    run(int n_states,
        ExcitationChannel channel = ExcitationChannel::singlet) const;

  private:
    const Method* mf_ = nullptr;
};

/// @brief Full linear-response TDDFT (Casida A/B) on a KS reference.
///
/// Mirrors `pyscf.tdscf.rks.TDDFT` / `pyscf.tdscf.uks.TDDFT`. Not a method
/// of `RKS` or `UKS`. Call `mf.run` first. Instantiated for `RKS` and `UKS`.
/// @tparam Method Concrete mean-field type.
template <typename Method>
class TDDFT {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit TDDFT(const Method& mf);

    /// @brief Lowest `n_states` Casida RPA roots.
    /// @param n_states Number of roots.
    /// @param channel Singlet or triplet for closed-shell RKS. Ignored for
    ///        UKS (no singlet/triplet split on the open-shell reference).
    /// @return Excitation energies [eV] and transition properties.
    ExcitedStatesResult
    run(int n_states,
        ExcitationChannel channel = ExcitationChannel::singlet) const;

  private:
    const Method* mf_ = nullptr;
};

/// @brief Collinear spin-flip TDA (SF-CIS) on a high-spin UHF reference.
///
/// Not `pyscf.tdscf.uhf.TDA` (that kernel is spin-conserving) and not a
/// method of `UHF`. Shao, Head-Gordon & Krylov, J. Chem. Phys. 118, 4807
/// (2003). Call `mf.run` first.
/// @tparam Method Concrete mean-field type. Instantiated for `UHF`.
template <typename Method>
class SfTDA {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit SfTDA(const Method& mf);

    /// @brief Lowest `n_states` spin-flip TDA roots.
    /// @param n_states Number of roots (1 ≤ n_states ≤ n_ov).
    /// @return Excitation energies [eV]. Oscillator strengths are zero.
    ExcitedStatesResult run(int n_states) const;

  private:
    const Method* mf_ = nullptr;
};

/// @brief Collinear spin-flip TDHF/RPA on a high-spin UHF reference.
///
/// Same singles space as `SfTDA`. Not a method of `UHF`.
/// @tparam Method Concrete mean-field type. Instantiated for `UHF`.
template <typename Method>
class SfTDHF {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit SfTDHF(const Method& mf);

    /// @brief Lowest positive real spin-flip RPA roots.
    /// @param n_states Number of roots (1 ≤ n_states ≤ n_ov).
    /// @return Excitation energies [eV]. Oscillator strengths are zero.
    ExcitedStatesResult run(int n_states) const;

  private:
    const Method* mf_ = nullptr;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
