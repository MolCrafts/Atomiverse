/// @file cpu/scf/rhf.h
/// @brief Restricted Hartree–Fock (RHF) concrete mean-field method.
#pragma once

#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/adc.h"
#include "atomiverse/cpu/scf/bse.h"
#include "atomiverse/cpu/scf/cisd.h"
#include "atomiverse/cpu/scf/eom_cc2.h"
#include "atomiverse/cpu/scf/eom_ea_ccsd.h"
#include "atomiverse/cpu/scf/eom_ee_ccsd.h"
#include "atomiverse/cpu/scf/eom_ip_ccsd.h"
#include "atomiverse/cpu/scf/g0w0.h"
#include "atomiverse/cpu/scf/gf2.h"
#include "atomiverse/cpu/scf/mean_field_method.h"
#include "atomiverse/cpu/scf/rpa.h"
#include "atomiverse/cpu/scf/tdhf.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

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
    /// @brief Construct with the RHF defaults (STO-3G; restricted HF; MINAO;
    ///        no density damping; energy tolerance 1e-9 Ha).
    ///
    /// Prefer the basis ctor: `RHF{BasisSet::STO3G}` — **do not** write
    /// `rhf.basis = …` after default construction.
    RHF();

    /// @brief Construct with an explicit basis (primary public form).
    /// @param basis_in Basis set (e.g. `BasisSet::STO3G`).
    explicit RHF(BasisSet basis_in);

    /// @brief Construct with basis and molecular charge.
    /// @param basis_in Basis set.
    /// @param charge_in Molecular charge [e] (electrons = ΣZ − charge).
    RHF(BasisSet basis_in, int charge_in);

    /// @brief Construct with basis, charge, SCF tolerance, and iteration cap.
    /// @param basis_in Basis set.
    /// @param charge_in Molecular charge [e].
    /// @param tol_in Energy / residual tolerance [Ha] (see base `tol`).
    /// @param max_iter_in Maximum SCF iterations [dimensionless].
    RHF(BasisSet basis_in, int charge_in, double tol_in, int max_iter_in = 100);

    std::string name() const { return "rhf"; }

    /// @brief Γ-point periodic RHF. Default false keeps the molecular
    ///        engine bit-identical. When true, Frame.box must be
    ///        orthorhombic; J, K, S, T, V and E_nn share one truncated
    ///        lattice-image window (no Ewald). A cell larger than the
    ///        Gaussian pair reach has only T = 0 and recovers molecular RHF.
    bool periodic = false;

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

    /// @brief Nuclear forces on TDA excited-state energy of root `state`
    ///        [eV/Å]. Default path is **analytic** CIS/TDA (integral
    ///        derivatives + relaxed difference density / Z-vector). Set
    ///        `analytic=false` to use central finite difference of
    ///        \(E_n^{\mathrm{TDA}}\) with step `h` [Å] (oracle / debug).
    /// @param frame Geometry (mutated temporarily on FD path; restored).
    /// @param state Excited root index (0-based).
    /// @param h Finite-difference step [Å] (FD path only; default 1e-3).
    /// @param analytic true (default) analytic CIS gradient; false = FD.
    /// @return Per-atom force array size 3N (fx,fy,fz interleaved) [eV/Å].
    std::vector<double> tda_forces(cpu::Frame& frame, int state = 0,
                                   double h = 1e-3, bool analytic = true) const;

    /// @brief Nonadiabatic coupling *strength* between TDA roots `i` and `j`:
    ///        ‖d_ij‖ [1/Å] where d_ij = tda_nac_vectors(...) (true NAC).
    /// @param frame Geometry [Å].
    /// @param state_i Lower root index (0-based).
    /// @param state_j Upper root index (0-based).
    /// @param h FD step if analytic forces fall back is not used [Å].
    double tda_nac_strength(cpu::Frame& frame, int state_i = 0, int state_j = 1,
                            double h = 2e-3) const;

    /// @brief True first-order NAC vectors d_ij = ⟨Ψ_i|∇_R|Ψ_j⟩ [1/Å],
    ///        length 3N (fx,fy,fz interleaved per atom). Built from analytic
    ///        TDA force difference / energy gap (Chernyak–Mukamel form) with
    ///        phase fixed so the first large component is positive.
    /// @param frame Geometry [Å].
    /// @param state_i Root i (0-based).
    /// @param state_j Root j (0-based); must differ from i.
    /// @return d_ij [1/Å]; d_ji = −d_ij.
    /// @throws on i==j, unconverged SCF, or degenerate gap.
    std::vector<double> tda_nac_vectors(cpu::Frame& frame, int state_i = 0,
                                        int state_j = 1) const;

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

    /// @brief CIS(D) doubles correction on the lowest `n_states` singlet CIS
    ///        (TDA) roots of the most recent `run()` — Head-Gordon, Rico,
    ///        Oumi & Lee, Chem. Phys. Lett. **219**, 21 (1994),
    ///        DOI 10.1016/0009-2614(94)00070-0. Size-consistent O(N⁵)
    ///        second-order correction; spin-orbital evaluation over the
    ///        closed-shell reference. Reuses the same TDA roots as
    ///        `tda(n_states, singlet)`.
    /// @param n_states Number of lowest CIS roots to correct (1 ≤ n_states ≤
    ///                 n_occ·n_virt; dimensionless count).
    /// @return CIS(D) and parent CIS excitation energies [eV] plus the
    ///         doubles correction U+V [eV] per root.
    /// @throws std::runtime_error when called before `run()`.
    /// @throws std::invalid_argument when n_states is outside
    ///         [1, n_occ·n_virt] or the system has no virtual orbitals.
    CISDResult cisd(int n_states = 1) const;

    /// @brief Strict ADC(2) electronic excitation energies of the most recent
    ///        `run()` — Schirmer, Phys. Rev. A **26**, 2395 (1982); Trofimov &
    ///        Schirmer, J. Phys. B **28**, 2299 (1995). Closed-shell singlet
    ///        manifold; matches PySCF `adc.ADC` with `method='adc(2)'` and
    ///        `method_type='ee'`. Builds the second-order singles intermediate
    ///        M_ss, first-order singles–doubles coupling, and zeroth-order
    ///        doubles diagonal; solves the non-symmetric ADC eigenvalue
    ///        problem with a matvec Davidson iteration (no ADC(2)-x or
    ///        ADC(3) terms).
    /// @param n_states Number of lowest ADC(2) roots (1 ≤ n_states ≤
    ///                 n_occ·n_virt; dimensionless count).
    /// @return Excitation energies [eV] and the MP2 correlation energy
    ///         [Hartree] of the RHF reference.
    /// @throws std::runtime_error when called before `run()`.
    /// @throws std::invalid_argument when n_states is outside
    ///         [1, n_occ·n_virt] or the system has no virtual orbitals.
    ADC2Result adc2(int n_states = 1) const;

    /// @brief Extended ADC(2)-x electronic excitation energies of the most
    ///        recent `run()` (closed-shell RHF). Same singles intermediate and
    ///        singles–doubles coupling as strict ADC(2), but the doubles block
    ///        includes first-order two-electron terms (vvvv, oovv, ovvo, oooo)
    ///        — PySCF `adc.ADC` / `method='adc(2)-x'` / `method_type='ee'`.
    /// @param n_states Number of lowest ADC(2)-x roots (1 ≤ n_states ≤
    ///                 n_occ·n_virt; dimensionless count).
    /// @return Excitation energies [eV] and the MP2 correlation energy
    ///         [Hartree] of the RHF reference.
    /// @throws std::runtime_error when called before `run()`.
    /// @throws std::invalid_argument when n_states is outside
    ///         [1, n_occ·n_virt] or the system has no virtual orbitals.
    ADC2Result adc2x(int n_states = 1) const;

    /// @brief Strict ADC(2) ionization potentials of the most recent
    ///        `run()` — Schirmer, Phys. Rev. A **26**, 2395 (1982).
    ///        Closed-shell 1h / 2h1p intermediate-state representation;
    ///        matches PySCF `adc.ADC` with `method='adc(2)'` and
    ///        `method_type='ip'`. Second-order M_ij (occupied–occupied),
    ///        first-order 1h–2h1p coupling via (ia|jk), zeroth-order 2h1p
    ///        diagonal D_ajk = ε_j + ε_k − ε_a; eigenvalues of the
    ///        sign-flipped ISR matrix are the ionization potentials.
    /// @param n_states Number of lowest IP roots (1 ≤ n_states ≤ n_occ).
    /// @return Ionization potentials [eV] and the MP2 correlation energy
    ///         [Hartree] of the RHF reference.
    /// @throws std::runtime_error when called before `run()`.
    /// @throws std::invalid_argument when n_states is outside [1, n_occ]
    ///         or the system has no virtual orbitals.
    IPADC2Result ip_adc2(int n_states = 1) const;

    /// @brief Strict EA-ADC(2) electron-attachment energies of the most recent
    ///        `run()` — Schirmer, Phys. Rev. A **26**, 2395 (1982); matches
    ///        PySCF `adc.ADC` with `method='adc(2)'` and `method_type='ea'`.
    ///        Singles = virtual (1p) space with second-order \(M_{ab}\);
    ///        doubles = 2p–1h with zeroth-order diagonal; first-order
    ///        singles–doubles coupling via \((ia|bc)\). No ADC(2)-x / ADC(3).
    /// @param n_states Number of lowest EA roots (1 ≤ n_states ≤
    ///                 n_virt + n_occ·n_virt²; dimensionless count).
    /// @return Attachment energies ω [eV] and the MP2 correlation energy
    ///         [Hartree] of the N-electron RHF reference.
    /// @throws std::runtime_error when called before `run()`.
    /// @throws std::invalid_argument when n_states is out of range or the
    ///         system has no virtual orbitals.
    EAADC2Result ea_adc2(int n_states = 1) const;

    /// @brief EOM-EA-CCSD electron-attachment energies — drives a fresh RHF
    ///        (materializes ERI), spin-orbital Gauss–Stanton CCSD, then the
    ///        non-symmetric EOM-EA Jacobian (Nooijen & Bartlett, J. Chem.
    ///        Phys. **102**, 3629 (1995)). Matches PySCF
    ///        `cc.CCSD(...).eaccsd()` / `eom_gccsd.EOMEA`.
    /// @param frame Geometry; drives SCF + CCSD + EOM.
    /// @param n_states Number of lowest unique attachment roots (≥ 1).
    /// @return Attachment ω [eV] plus CCSD energetics [Hartree].
    /// @throws std::invalid_argument on open-shell / missing virtuals /
    ///         non-converged SCF or CCSD / n_states out of range.
    EOMEACCSDResult eom_ea_ccsd(Frame& frame, int n_states = 1);

    /// @brief EOM-IP-CCSD ionization potentials — drives a fresh RHF
    ///        (materializes ERI), spin-orbital Gauss–Stanton CCSD, then the
    ///        non-symmetric EOM-IP Jacobian (Tu, Wang & Li, J. Chem. Phys.
    ///        **136**, 174102 (2012)). Matches PySCF
    ///        `cc.CCSD(...).ipccsd()` / `eom_gccsd.EOMIP`.
    /// @param frame Geometry; drives SCF + CCSD + EOM.
    /// @param n_states Number of lowest unique ionization roots (≥ 1).
    /// @return Ionization ω [eV] plus CCSD energetics [Hartree].
    /// @throws std::invalid_argument on open-shell / missing occupieds /
    ///         non-converged SCF or CCSD / n_states out of range.
    EOMIPCCSDResult eom_ip_ccsd(Frame& frame, int n_states = 1);

    /// @brief Direct RPA (dRPA) correlation energy of the most recent `run()`
    ///        (closed-shell RHF reference). Coulomb-only Casida A/B
    ///        (α_hf = 0); correlation via the Furche plasmon formula
    ///        \(E_c = \frac12(\sum_s\Omega_s - \mathrm{Tr} A)\). Exact
    ///        (non-DF) twin of PySCF `gw.rpa.RPA` / `DirectRPA` in the
    ///        complete-auxiliary limit; stock PySCF uses density fitting.
    /// @return Correlation and total energy [Hartree], plus the dRPA
    ///         poles \(\Omega_s\) [Hartree].
    /// @throws std::runtime_error when called before `run()`, on a
    ///         non-restricted reference, or when dRPA is ill-conditioned.
    /// @throws std::invalid_argument when n_occ = 0 or n_virt = 0.
    DirectRPAResult direct_rpa() const;

    /// @brief Singlet BSE@G0W0 excitation energies of the most recent
    ///        `run()` (closed-shell RHF). Spectral G0W0 poles
    ///        (`g0w0()`) enter the ph energy differences and the static
    ///        direct-RPA screened interaction \(W\) (exact 4-centre form).
    ///        Full BSE (default) solves the Casida A/B RPA problem; with
    ///        `tda = true` only A is diagonalized (BSE-TDA).
    ///        Working equations match PySCF `pyscf.gw.bse.BSE` (non-DF ERIs
    ///        + frequency-exact G0W0 poles — see `reg_bse-h2-sto3g`).
    /// @param n_states Number of lowest singlet roots in `[1, n_occ·n_virt]`.
    /// @param tda If true, Tamm–Dancoff (B = 0); else full BSE.
    /// @return Excitation energies [eV] plus the underlying G0W0 result.
    /// @throws std::runtime_error when called before `run()`, on a
    ///         non-restricted reference, or when G0W0 / BSE is ill-conditioned.
    /// @throws std::invalid_argument when `n_states` is out of range or the
    ///         system has no virtual orbitals.
    BSEResult bse(int n_states = 1, bool tda = false) const;

    /// @brief One-shot second-order Green's function (G0F2 / GF2)
    ///        quasiparticle energies of the most recent `run()` (closed-shell
    ///        RHF). Builds the 2h1p+2p1h auxiliary self-energy from the HF
    ///        Green's function and solves the Dyson equation once with
    ///        F = diag(ε_HF). Matches PySCF `agf2.RAGF2` one-shot poles
    ///        (SE from `init_gf` → `se.get_greens_function(F_HF)`). Not
    ///        self-consistent AGF2 (separate roadmap item).
    /// @return QP energies [Hartree], HOMO–LUMO gap / IP / EA [eV],
    ///         Galitskii–Migdal e_corr [Hartree].
    /// @throws std::runtime_error when called before `run()` or on a
    ///         non-restricted reference.
    /// @throws std::invalid_argument when n_occ = 0 or n_virt = 0.
    GF2Result gf2() const;

    /// @brief Singlet EOM-EE-CCSD excitation energies — drives a fresh RHF
    ///        (materializes ERI), spin-orbital Gauss–Stanton CCSD, then the
    ///        non-symmetric EOM-EE Jacobian (Wang, Tu & Wang, JCTC **10**,
    ///        5567 (2014); Stanton & Bartlett, J. Chem. Phys. **98**, 7029
    ///        (1993)). Matches PySCF `cc.CCSD(...).eomee_ccsd_singlet()`.
    /// @param frame Geometry; drives SCF + CCSD + EOM.
    /// @param n_states Number of lowest singlet roots (≥ 1).
    /// @return Singlet ω [eV] plus CCSD energetics [Hartree].
    /// @throws std::invalid_argument on open-shell / missing virtuals /
    ///         non-converged SCF or CCSD / n_states out of range.
    EOMEECCSDResult eom_ee_ccsd(Frame& frame, int n_states = 1);

    /// @brief Singlet EOM-CC2 excitation energies — drives a fresh RHF
    ///        (materializes ERI), spin-orbital Christiansen–Koch–Jørgensen
    ///        CC2 ground state, then the same non-symmetric EOM-EE Jacobian
    ///        evaluated at the CC2 amplitudes (matches PySCF
    ///        `cc.RCCSD` with `cc2=True` followed by `eomee_ccsd_singlet`).
    /// @param frame Geometry; drives SCF + CC2 + EOM.
    /// @param n_states Number of lowest singlet roots (≥ 1).
    /// @return Singlet ω [eV] plus CC2 energetics [Hartree].
    /// @throws std::invalid_argument on open-shell / missing virtuals /
    ///         non-converged SCF or CC2 / n_states out of range.
    EOMCC2Result eom_cc2(Frame& frame, int n_states = 1);

    /// @brief Spectral G0W0 quasiparticle energies of the most recent `run()`
    ///        (closed-shell RHF reference). Direct-RPA (dRPA, α_hf = 0)
    ///        screens W; the diagonal correlation self-energy is the
    ///        frequency-exact spectral sum over RPA poles; each orbital's
    ///        quasiparticle equation is solved by Newton (non-linear, not
    ///        linearized). Matches PySCF `gw.gw_exact.GWExact` on
    ///        `dft.RKS(mol, xc='hf')` (same HF density / orbitals).
    /// @return QP energies [Hartree], HOMO–LUMO gap / IP / EA [eV].
    /// @throws std::runtime_error when called before `run()`, on a
    ///         non-restricted reference, or when dRPA is ill-conditioned.
    /// @throws std::invalid_argument when n_occ = 0 or n_virt = 0.
    G0W0Result g0w0() const;

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

    /// @brief Static dipole polarizability α_ab = ∂μ_a/∂F_b [a.u.] of the
    ///        most recent `run()` via CPHF electric-field response.
    ///
    /// Solves ¹(A+B)·U^{F_b} = −r_b^{vo} for the three Cartesian field
    /// directions (the LHS is the link-08 `OrbitalHessian` singlet σ₊ —
    /// the same kernel used by nuclear CPHF in `hessian()`), then assembles
    ///
    ///   α_ab = −4 Σ_ai U^{F_b}_ai ⟨a|r_a|i⟩
    ///
    /// with length-gauge dipole integrals about the center of nuclear charge
    /// (same origin as `MultipoleObserver` / `ExternalField`). Closed-shell
    /// RHF only; S^{(F)} = 0 so the nuclear multipole does not enter the
    /// second derivative. Matches PySCF finite-field / CPHF polarizability
    /// on the same geometry and basis (see regressions/scf/static-polariz-h2o).
    ///
    /// @return Dense symmetric 3×3 tensor [a.u. = e²·Bohr²/Ha], row a =
    ///         Cartesian dipole axis, column b = field axis.
    /// @throws std::runtime_error when called before `run()`.
    /// @throws std::invalid_argument when the system has no virtual
    ///         orbitals (e.g. He/STO-3G — no occ→virt response exists).
    xt::xtensor<double, 2> polarizability() const;

  protected:
    /// @brief Closed-shell occupation: n_alpha = n_beta = n_electrons / 2
    ///        (every occupied spatial orbital holds one α and one β electron).
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
