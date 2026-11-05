/// @file cpu/scf/mean_field_method.h
/// @brief Shared CRTP (curiously recurring template pattern) base for the
///        concrete Gaussian mean-field methods: `MeanFieldMethod<D>` (base of
///        RHF / ROHF / UHF / RKS / UKS).
///
/// There is no bag named `SCFSolver` / freestanding integral bag / `SCFConfig`.
/// Each method is a concrete class; parameters are **constructor arguments**
/// (e.g. `RHF{BasisSet::STO3G, charge, tol}`), fixed after construction;
/// SCF knobs are public members. The only I/O currency is `Frame`. Integral
/// **fields** live as method-private members; engines take
/// `internal::IntegralView` (non-owning refs).
///
/// SCF self-consistency is a **Driver loop**: `MeanFieldMethod<D>` derives
/// `cpu::Driver<ATV::scf::Phase>` and the pure tag `HF` (not a Component).
/// Post-HF (`post_hf.h`), CIS(D) (`cisd.h`), spectral G0W0 (`g0w0.h`),
/// ADC (`adc.h`), direct RPA (`rpa.h`), BSE (`bse.h`), one-shot GF2
/// (`gf2.h`), TDHF (`tdhf.h`), continuum solvent (`solvent.h`), the nuclear
/// FD Hessian (`nuclear_hessian.h`), and CPHF response (`response.h`) are
/// separate types — not methods here.
/// Nested under an outer MD/Evaluate/Minimize force or energy phase it runs
/// via `run_nested` → `drive_scf_`. Standalone entry is `run(frame)`. Integrals
/// and AO workspaces are private. The concrete methods live in `rhf.h` /
/// `uhf.h` / `rks.h` / `uks.h`; `methods.h` is their umbrella.
///
/// @code
///   cpu::scf::RHF rhf{BasisSet::STO3G};           // preferred
///   // or: RHF{BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-10, /*max_iter=*/100};
///   rhf.run(frame);                              // eV on frame.meta[kEnergy]
///   // forces: RHF{…}.with_gradient(true) nested at force phase, or
///   //         GradientResult g = rhf.forces(frame);
/// @endcode
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/component/timestep_protocol.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/electronic/cpmd.h"
#include "atomiverse/cpu/electronic/xl_bomd.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/scf_enums.h"
#include "atomiverse/cpu/scf/scf_guess.h"
#include "atomiverse/cpu/scf/slater_determinant.h"
#include "atomiverse/cpu/scf/stages/density_stage.h"
#include "atomiverse/cpu/scf/stages/diagonalize_stage.h"
#include "atomiverse/cpu/scf/stages/fock_stage.h"
#include "atomiverse/cpu/scf/stages/mix_stage.h"
#include "atomiverse/cpu/scf/stages/rohf_couple_stage.h"
#include "atomiverse/cpu/scf/stages/scf_converged.h"
#include "atomiverse/cpu/xc/xc_functional.h"

namespace ATV {
namespace cpu {

namespace electronic {
class ASPCPredictor; // Kolafa density extrapolator (aspc_predictor.h)
template <typename Method>
class CarParrinelloMethod;
template <typename Method>
class XlBomdMethod;
} // namespace electronic

namespace scf {

namespace internal {
class Pcm; // continuum solvent private workspace (pcm.h)
}

// Post-HF, solvent, and response are friends so they can read the converged
// reference. They are not methods of this class (PySCF: MP2(mf),
// SCFWithSolvent, cphf.solve).
template <typename Method>
class MP2;
template <typename Method>
class MP3;
template <typename Method>
class MP4;
template <typename Method>
class FCI;
template <typename Method>
class CASCI;
template <typename Method>
class CASSCF;
template <typename Method>
class NEVPT2;
template <typename Method>
class CCSD;
template <typename Method>
class CC2;
template <typename Method>
class WithSolvent;
template <typename Method>
class Hessian;
template <typename Method>
class Polarizability;
template <typename Method>
class Stability;
template <typename Method>
class TDA;
template <typename Method>
class TDHF;
template <typename Method>
class TDDFT;
template <typename Method>
class SfTDA;
template <typename Method>
class SfTDHF;
template <typename Method>
class CISD;
template <typename Method>
class G0W0;
template <typename Method>
class ADC;
template <typename Method>
class DirectRPA;
template <typename Method>
class BSE;
template <typename Method>
class GF2;

/// @brief Shared view for observers/recorders after a method run.
///
/// `shells` is the basis-function set (pointer storage owned by the method's
/// private integral workspace via shared_ptr aliasing). Spin treatment is
/// read from the wavefunction (`result->restricted()`). Integral storage is
/// method-private (not a public bag).
struct MeanFieldOutputs {
    std::shared_ptr<const SlaterDeterminant> result;
    std::shared_ptr<const std::vector<native::Shell>> shells;
    std::shared_ptr<const bool> ready;

    /// @brief Cartesian→spherical transform of the pack (empty tensor for
    ///        Cartesian-convention bases). Consumers that rebuild integrals
    ///        from `shells` (Cartesian) sandwich with this to match the
    ///        determinant's AO dimension.
    std::shared_ptr<const xt::xtensor<double, 2>> c2s;
};

/// @brief Shared members and Frame API for concrete mean-field methods.
///
/// CRTP (`Derived` = `RHF`/`UHF`/`RKS`/`UKS`) so chaining
/// (`RHF{}.predictor(3)`) keeps the concrete type for `on(phase).add`.
///
/// Derives `cpu::Driver<ATV::scf::Phase>` (the self-consistency loop node)
/// and pure tag `HF` (not a Component). Nested binding is a native driver node;
/// the sole compute entry is `drive_scf_` (via `run` / `forces` / nested
/// `run_nested_erased`).
///
/// Also a `cpu::TimestepConsumer`, but only under `cpmd = true`: Car–Parrinello
/// propagates the electrons on the nuclear clock, which the MD integrator owns
/// (`timestep_required()` tracks the flag exactly, so a Born–Oppenheimer run
/// never demands an integrator).
template <typename Derived>
class MeanFieldMethod : public cpu::Driver<ATV::scf::Phase>,
                        public HF,
                        public cpu::TimestepConsumer,
                        public cpu::QmEngine,
                        public cpu::EmbeddingEngine {
  public:
    /// @brief Set OpenMP threads for both integral setup and SCF stages.
    /// @param n Positive thread count; zero keeps the runtime default.
    void set_threads(int n);
    bool requires_full_frame() const override { return true; }
    bool provides_energy() const override { return true; }
    bool provides_force() const override { return compute_force; }
    cpu::TimestepConsumer* timestep_consumer() override { return this; }

  private:
    template <typename Method>
    friend class ATV::cpu::electronic::CarParrinelloMethod;
    template <typename Method>
    friend class ATV::cpu::electronic::XlBomdMethod;
    template <typename Method>
    friend class MP2;
    template <typename Method>
    friend class MP3;
    template <typename Method>
    friend class MP4;
    template <typename Method>
    friend class FCI;
    template <typename Method>
    friend class CASCI;
    template <typename Method>
    friend class CASSCF;
    template <typename Method>
    friend class NEVPT2;
    template <typename Method>
    friend class CCSD;
    template <typename Method>
    friend class CC2;
    template <typename Method>
    friend class WithSolvent;
    template <typename Method>
    friend class Hessian;
    template <typename Method>
    friend class Polarizability;
    template <typename Method>
    friend class Stability;
    template <typename Method>
    friend class TDA;
    template <typename Method>
    friend class TDHF;
    template <typename Method>
    friend class TDDFT;
    template <typename Method>
    friend class SfTDA;
    template <typename Method>
    friend class SfTDHF;
    template <typename Method>
    friend class CISD;
    template <typename Method>
    friend class G0W0;
    template <typename Method>
    friend class ADC;
    template <typename Method>
    friend class DirectRPA;
    template <typename Method>
    friend class BSE;
    template <typename Method>
    friend class GF2;

  public:
    // ── Identity (constructor arguments, fixed after construction) ────
    /// @brief Gaussian basis set.
    const BasisSet basis;
    /// @brief Molecular charge [e] (electrons = ΣZ − charge).
    const int charge;

    /// @brief Local atom indices treated as ghosts (BSSE counterpoise).
    ///
    /// Ghost atoms contribute basis functions to the AO space but carry
    /// zero nuclear charge and contribute no electrons (Boys–Bernardi
    /// counterpoise: basis-only sites). Empty = no ghosts (default;
    /// bit-identical to the pre-ghost path). Ghost markers live on the
    /// method, never on Frame (Frame purity; molpy Element forbids Z ≤ 0).
    /// Non-empty + Kohn–Sham (RKS/UKS) throws — KS grid radii key off Z
    /// and are not wired for ghosts in this link.
    std::vector<int> ghost_atoms;

    /// @brief Effective-core-potential family name; "" = all-electron.
    ///
    /// Currently "def2-ECP" (Weigend-Ahlrichs def2 family, Rb–Rn): atoms
    /// present in the table run valence-only (Z_val nuclear terms, core
    /// electrons removed, semi-local U added to the core Hamiltonian
    /// [Hartree]); atoms absent stay all-electron (def2 convention: Z ≤ 36).
    /// Unknown names throw std::runtime_error — no fallback.
    std::string ecp;

    /// @brief Spin-free exact-two-component scalar relativity (x2c-1e).
    ///
    /// When true, the one-electron core Hamiltonian is replaced by the
    /// one-step x2c-1e h^X2C built from {S, T, V, W=⟨∇|V|∇⟩} in the
    /// contracted AO basis (`internal::X2cWorkspace`; Kutzelnigg–Liu /
    /// Peng–Reiher). S, ERI, and the Fock loop are unchanged (two-electron
    /// integrals stay non-relativistic — that is the x2c-1e definition).
    ///
    /// Combining x2c with an ECP-carrying atom throws at run(): relativistic
    /// ECPs already bake scalar relativity into their parameterization, so
    /// stacking x2c on top would double-count it (per-element gate).
    /// Analytic forces replace Tr[P·(dT+dV)] by Tr[P·dh^X2C]
    /// (`pyscf.x2c.sfx2c1e_grad`, contracted basis). Property operators
    /// are not picture-change transformed. Car–Parrinello with x2c throws.
    ///
    /// **Picture-change warning:** expectation values of property operators
    /// (dipole, ESP, multipoles) evaluated on x2c orbitals with the
    /// untransformed non-relativistic operator contain picture-change error
    /// (PCE). This method does **not** apply a picture-change transform to
    /// property operators — document or correct PCE externally if needed.
    bool x2c = false;

    // ── Convergence (members, not a Config bag) ───────────────────────
    // "DIIS" below = direct inversion of the iterative subspace: the standard
    // SCF convergence accelerator that extrapolates the Fock matrix from a short
    // history of the previous iterations' error vectors.
    double tol = 1e-9;         ///< |ΔE| threshold [Hartree].
    double conv_tol_grad = -1; ///< DIIS-error tol; <0 → sqrt(tol).
    int max_iter = 100;
    bool use_diis = true;
    int diis_max_vectors = 8;
    int diis_start_iter = 1;
    double damping_factor = 0.0;
    InitGuess init_guess = InitGuess::MINAO;

    /// @brief Fock-mixer family (dimensionless selector). Default `CDIIS`
    ///        keeps the shipping loop bit-identical. `EDIIS_CDIIS` /
    ///        `ADIIS_CDIIS` run the far-field mixer first and permanently
    ///        switch to CDIIS once max|DIIS error| < `mixer_switch_tol`
    ///        (EDIIS/ADIIS are far-from-convergence mixers only; see
    ///        `internal/ediis_adiis.h` for the DOIs).
    MixerKind mixer = MixerKind::CDIIS;

    /// @brief max|DIIS error| threshold (dimensionless) for the permanent
    ///        EDIIS/ADIIS → CDIIS escalation switch. Ignored when
    ///        `mixer == MixerKind::CDIIS`.
    double mixer_switch_tol = 1e-2;

    double level_shift = 0.0;
    double lindep_threshold = 1e-8;

    // ── Density fitting (RI) — hf-capability-gaps-12-df ───────────────
    /// @brief Density-fitting mode for the two-electron Fock build
    ///        (dimensionless selector). `RI::None` (default) keeps the
    ///        exact-ERI path bit-identical; `RI::J` fits the Coulomb
    ///        matrix only; `RI::JK` fits Coulomb and exchange over the
    ///        def2 universal JKFIT auxiliary basis. Wired for RHF/UHF
    ///        and the RKS/UKS Fock builds; combining with ROHF, GHF,
    ///        SOSCF, or analytic forces throws (not wired — no silent
    ///        fallback). See scf_enums.h for the DOI trail.
    RI ri = RI::None;

    /// @brief Auxiliary basis for the RI fit (ignored when
    ///        `ri == RI::None`). An element without a JKFIT table row
    ///        throws at Fock-build setup — no fallback.
    AuxBasis aux_basis = AuxBasis::Def2JKFIT;

    // ── Coulomb algorithm (CFMM) — hf-capability-gaps-21-cfmm ─────────
    /// @brief J-matrix algorithm. `Coulomb::Direct` (default) keeps the
    ///        exact-ERI path bit-identical. `Coulomb::CFMM` selects the
    ///        continuous fast multipole method for J (K stays direct/RI).
    ///        Combining `Coulomb::CFMM` with `ri = RI::J/JK` throws.
    Coulomb coulomb = Coulomb::Direct;

    /// @brief Multipole truncation order p for CFMM-J (dimensionless).
    ///        Error of the far-field M2L decays geometrically with p under
    ///        the well-separated criterion. Default 10.
    int cfmm_order = 10;

    /// @brief Extent / multipole threshold ε_thr for CFMM-J (dimensionless
    ///        charge fraction outside the WJGH ball). Sets
    ///        r_ext = √(2/ζ)·erfc⁻¹(ε) [Bohr]. Default 1e-9.
    double cfmm_threshold = 1e-9;

    // ── Second-order SCF (SOSCF / CIAH; hf-capability-gaps-09) ────────
    /// @brief Enable the second-order finishing path: DIIS runs the early
    ///        iterations, then — once max|DIIS error| < `soscf_start` — the
    ///        orbital update switches permanently (per run) to CIAH
    ///        augmented-Hessian Newton rotations C ← C·exp(κ)
    ///        (Sun, arXiv:1610.08423; quadratic convergence). HF only
    ///        (RHF/UHF); enabling on RKS/UKS throws (the XC second-order
    ///        kernel is out of scope). Default false — the shipping path is
    ///        bit-identical when off.
    bool soscf = false;

    /// @brief SOSCF activation threshold on max|DIIS error| (dimensionless,
    ///        ∞-norm of the AO-basis DIIS error matrix). <= 0 activates at
    ///        the earliest eligible iteration. Eligibility starts at
    ///        iteration 1: iteration 0 always runs one Roothaan
    ///        diagonalization so the coefficients are synchronized with a
    ///        density-only initial guess (MINAO / caller-provided) before
    ///        Newton rotations act on them.
    double soscf_start = 0.25;

    /// @brief Davidson micro-iteration budget of each CIAH augmented-Hessian
    ///        solve (dimensionless count per macro step); at the budget the
    ///        best Ritz pair is used (co-iteration, not failure).
    int soscf_max_micro = 12;

    /// @brief Trust radius bound on the rotation step norm ‖κ‖₂
    ///        (dimensionless); steps are clipped to it and it halves
    ///        whenever a step raises the energy (reject-and-retry).
    double soscf_trust = 0.4;

    // ── KS grid (ignored for pure HF) ─────────────────────────────────
    GridLevel grid_level = GridLevel::SG0;
    int n_radial = 50;
    int n_angular = 110;

    // ── Debug / regression knobs ──────────────────────────────────────
    bool xc_grad_project_translation = false;
    double hybrid_alpha_override = -1.0;
    const MolecularGrid* fixed_grid = nullptr;

    /// @brief Process memory budget [MB], using PySCF's 95% cache admission.
    /// The exact J/K engine streams screened quartets when s8 does not fit.
    double max_memory_mb = 4000.0;

    /// @brief Distribute exact HF J/K over MPI_COMM_WORLD (default false).
    /// Requires ATV_USE_MPI and initialized MPI; every rank calls the solve
    /// with identical controls and a complete replicated molecular Frame.
    bool mpi = false;

    /// @brief When true (default), nested force-slot runs write forces.
    bool compute_force = true;

    /// @brief Maximum-overlap occupation selection for ΔSCF (default Off =
    ///        energy Aufbau; bit-identical to the pre-MOM path). Set to
    ///        `MomStyle::IMOM` (PySCF `mom_occ_`) or `MomStyle::MOM` and
    ///        seed the reference occupied subspace via `set_mom_reference`
    ///        (or `UHF::delta_scf`, which does both) before `run`.
    MomStyle mom = MomStyle::Off;

    /// @brief Install the MOM/IMOM reference occupied MO columns from a
    ///        (possibly non-Aufbau) determinant. Copies occupied columns of
    ///        each spin channel into the private reference buffers used by
    ///        the density stage. No-op side effect on the determinant.
    /// @param det Seed wavefunction; occupations select the columns.
    /// @throws std::invalid_argument on empty occupation or dimension
    ///         mismatch with a subsequent `run` basis.
    void set_mom_reference(const SlaterDeterminant& det);

    // ── Environment (hf-capability-gaps-15-embed / esmm-qmmm-01) ───────
    /// @brief External MM-like point charges (positions [Å], charges [e]).
    ///
    /// Empty = no embedding (bit-identical vacuum path). Wired into private
    /// Hcore / classical energy in `drive_scf_core_`; never stored on Frame.
    /// Type is the shared public `cpu::EmbeddingSite` (no nested PointCharge).
    std::vector<cpu::EmbeddingSite> point_charges;

    /// @brief Include MM–MM self Coulomb in the embedding energy/gradient.
    ///
    /// Default true preserves standalone embedding energy pins
    /// (`h2o_embedding_field`). Additive QM/MM couplings set false so MM–MM
    /// lives in exactly one place (the MM force field).
    bool embed_self_energy = true;

    /// @brief Forces on embedding sites from the last `forces()` / nested force
    ///        evaluation [eV/Å]. Empty when `point_charges` was empty.
    /// @return Const reference to the republished site forces.
    const std::vector<cpu::EmbeddingForce>&
    embedding_site_forces() const override {
        return embed_site_forces_;
    }

    // ── QmEngine / EmbeddingEngine (esmm-qmmm-02) ─────────────────────
    std::string qm_engine_name() const override;
    void prepare(const cpu::BuildContext& qm_ctx) override;
    void evaluate(const cpu::LaunchContext& qm_ctx) override;
    cpu::EmbeddingEngine* as_embedding() override { return this; }
    void reset_history() override;
    void set_embedding_sites(std::vector<cpu::EmbeddingSite> sites) override {
        point_charges = std::move(sites);
    }
    void set_site_self_energy(bool enabled) override {
        embed_self_energy = enabled;
    }

    /// @brief Uniform external electric field (Ex, Ey, Ez) [a.u.].
    ///
    /// Origin is the center of nuclear charge (same as the multipole
    /// observer). **Finite-field collapse hazard:** keep |F| ≲ 1e-4 a.u.;
    /// a static field makes the energy unbounded below (diffuse bases can
    /// "ionize"). Zero field = vacuum path (bit-identical).
    std::array<double, 3> external_field{{0.0, 0.0, 0.0}};

    // ── Dispersion (molecular DFT-D3 BJ / DFT-D4) ─────────────────────
    /// @brief When true, add PBE-D3(BJ) dispersion (Grimme et al. 2011) to
    ///        the total energy after SCF. Geometry-only additive term —
    ///        does not enter the Fock build. Default false (bit-identical
    ///        vacuum path). Public standalone evaluator: `scf::D3BJ`.
    bool d3bj = false;
    /// @brief When true, add PBE-D4 dispersion (Caldeweyher et al. 2019) to
    ///        the total energy after SCF. Geometry-only; not in Fock.
    ///        Mutually exclusive with `d3bj` in practice (last writer wins
    ///        if both set — prefer one). Public evaluator: `scf::D4`.
    bool d4 = false;

    /// @brief True when Car–Parrinello electronic dynamics is enabled (via wrapper).
    bool is_cpmd() const { return cpmd; }
    /// @brief True when XL-BOMD auxiliary-density dynamics is enabled.
    bool is_xl_bomd() const { return xl_bomd; }
    /// @brief XL-BOMD dissipation order K (meaningful when `is_xl_bomd()`).
    int xl_bomd_order() const { return xl_bomd_k; }

  protected:
    int requested_threads_ = 0;
    // ── Car–Parrinello molecular dynamics (esmm-cpmd-04) ──────────────
    /// @brief Propagate the occupied orbitals with a fictitious mass instead
    ///        of re-converging the SCF at every nuclear step (Car &
    ///        Parrinello, PRL **55**, 2471 (1985)).
    ///
    /// **Nested MD force slot only.** The electrons follow a nuclear
    /// trajectory, so there is nothing to propagate outside an MD loop:
    /// standalone `run()` / `forces()` with `cpmd = true` throw
    /// `std::logic_error`. Bind the method at `md::Phase::force` and let the
    /// integrator publish the timestep (`TimestepConsumer`; there is
    /// deliberately no `cpmd_dt` member).
    ///
    /// The first nested step is a cold seed: one full SCF (+ Born–Oppenheimer
    /// forces) whose converged occupied coefficients seed the propagator.
    /// Every later step is a fixed-density evaluation plus one electronic
    /// velocity-Verlet/RATTLE advance — `iterations()` is 0 forever after.
    /// RHF/RKS propagate one channel at occ = 2; UHF/UKS propagate α and β
    /// independently at occ = 1 each. ROHF/ROKS/GHF are refused, as are the
    /// configuration rows listed on `validate_cpmd_config_`.
    ///
    /// `determinant()` is NOT refreshed on production steps: propagated
    /// orbitals have no canonical eigenvalues, so the published wavefunction
    /// stays the seed SCF's — a documented limitation, not an oversight.
    bool cpmd = false;

    /// @brief Fictitious electronic mass μ [a.u., electron-mass units].
    ///        Typical CPMD values are 10²–10³; the AO residual is stiffer
    ///        than a plane-wave one, so a too-large μ breaks adiabaticity.
    double cpmd_mu = 400.0;

    /// @brief Couple a Nosé–Hoover thermostat to the fictitious electronic
    ///        degrees of freedom (Blöchl & Parrinello, PRB **45**, 9413
    ///        (1992)). Default false = NVE electrons.
    bool cpmd_thermostat = false;

    /// @brief Target fictitious electronic kinetic energy K₀ [Hartree] for
    ///        the electronic thermostat. Ignored when `cpmd_thermostat` is
    ///        false; must be finite and > 0 when it is true.
    double cpmd_ekin_target_ha = 1e-5;

    /// @brief Electronic thermostat coupling time τ_e [fs]. Ignored when
    ///        `cpmd_thermostat` is false; must be finite and > 0 when true.
    double cpmd_tau_e = 50.0;

    // ── Extended-Lagrangian BOMD (esmm-cpmd-05) ───────────────────────
    /// @brief Propagate an auxiliary density with its own equation of motion
    ///        and hand it to the SCF as the initial guess instead of reusing
    ///        the previous step's converged one (Niklasson et al.,
    ///        J. Chem. Phys. **130**, 214109 (2009)).
    ///
    /// Reusing the last density is not time-reversible — the guess always lags
    /// the nuclei — and the one-sided error pumps energy into the trajectory.
    /// The auxiliary density has its own (almost time-reversible) dynamics, so
    /// warm steps converge in a few iterations without ASPC's corrector bias.
    /// Dimensionless flag; default false = plain Born–Oppenheimer.
    ///
    /// **Nested-MD state.** The auxiliary density is a dynamical variable on
    /// the nuclear clock: it advances exactly once per **converged nested**
    /// run, so bind the method at `md::Phase::force`. Standalone `run()` /
    /// `forces()` with `xl_bomd = true` are allowed (XL is a guess strategy,
    /// not a propagator of the SCF state) but a single point is somewhere else
    /// in configuration space, so a standalone entry leaves the auxiliary
    /// history **untouched** — the same line GPW takes. An unconverged nested
    /// run drops the history instead of propagating an SCF residual into every
    /// later guess.
    ///
    /// **Exclusive with the other guess owners.** `predictor(...)` (ASPC
    /// extrapolates the density) and `cpmd` (Car–Parrinello propagates the
    /// coefficients) each answer "what is the density now", so combining
    /// either with `xl_bomd` throws `std::invalid_argument` on the first
    /// nested entry (`GPWEvaluator::build` refuses the same pairs).
    ///
    /// Channel convention: RHF/RKS propagate the factor-2 total density in one
    /// channel; UHF/UKS/ROHF/ROKS propagate α and β independently; GHF
    /// propagates the single 2n-dimensional spinor density.
    bool xl_bomd = false;

    /// @brief XL-BOMD dissipation order K — history depth K+1 (dimensionless).
    ///
    /// (κ, α, c_k) are tabulated rows, not a formula: Table I of Niklasson et
    /// al. covers K ∈ [3, 9] and anything else throws `std::invalid_argument`
    /// (lookup table, no fallback). Default 5, the order `GPWEvaluator` ships.
    /// Changing it between runs re-seats the propagators (history dropped).
    int xl_bomd_k = 5;

  public:
    /// @brief XL-BOMD auxiliary-density propagation steps taken (one per
    ///        converged nested run). @return Count; 0 under BOMD.
    int xl_advances() const { return xl_advances_; }

    /// @brief Fictitious electronic kinetic energy T_e of the last CPMD step.
    /// @return T_e [eV]; 0 before any electron has moved (including the cold
    ///         seed step). Also published on `frame.meta[kEnergyElectronKinetic]`.
    double last_electronic_ke_ev() const { return last_te_ev_; }

    /// @brief Electronic Nosé–Hoover thermostat energy of the last CPMD step.
    /// @return E_therm [eV]; exactly 0 when `cpmd_thermostat` is false. Also
    ///         published on `frame.meta[kEnergyElectronThermostat]`.
    double last_electronic_thermostat_ev() const { return last_etherm_ev_; }

    /// @brief True when this method cannot run without the nuclear timestep
    ///        — i.e. exactly when `cpmd` is set (`TimestepConsumer`).
    /// @return `cpmd`.
    bool timestep_required() const override { return cpmd; }

    /// @brief Receive the nuclear MD timestep from the integrator
    ///        (`cpu::md::MD::build()` pushes it before any component builds).
    /// @param dt_fs Nuclear timestep [fs]; must be finite and > 0.
    /// @throws std::invalid_argument when `dt_fs` is not finite and > 0.
    void set_timestep(double dt_fs) override;

    /// @brief Fix the method identity; concrete methods set their defaults.
    /// @param basis Gaussian basis set.
    /// @param charge Molecular charge [e].
    MeanFieldMethod(BasisSet basis, int charge);
    ~MeanFieldMethod() override;
    MeanFieldMethod(MeanFieldMethod&&);
    MeanFieldMethod& operator=(MeanFieldMethod&&) = delete;
    MeanFieldMethod(const MeanFieldMethod&) = delete;
    MeanFieldMethod& operator=(const MeanFieldMethod&) = delete;

    /// @brief Nested budget defaults to the SCF iteration cap.
    int default_budget() const override { return max_iter; }

    /// @brief Nested force/energy slot: one full SCF (+ optional forces).
    /// @param ctx Parent Context (borrowed; status isolation is internal).
    /// @param f   System state.
    /// @param budget SCF iteration budget; <0 → `max_iter`.
    /// @return SCF iterations actually executed.
    int run_nested_erased(cpu::Context& ctx, cpu::Frame& f,
                          int budget) override;

    /// @brief Nested build is deferred to first `drive_scf_` (needs geometry).
    void build_on(cpu::Runtime& rt) override;

    /// @brief Drop the method's private Plan / Backend wiring.
    void destroy_on(cpu::Runtime& rt) override;

    /// @brief Run SCF on `frame` (driver loop). Integrals are private
    ///        workspace. Frame-out: always writes the eV energy to
    ///        `frame.meta[kEnergy]`; Hartree internals via `energy()`,
    ///        loop state via `converged()`/`stop_reason()`/`iterations()`,
    ///        wavefunction via `determinant()`.
    /// @param frame  Geometry (Å) + Z in type column.
    /// @param guess  Optional density guess.
    /// @throws std::logic_error when `cpmd` is set — Car–Parrinello is a
    ///         nested-MD-force-slot method (there is no nuclear trajectory
    ///         for the electrons to follow at a standalone single point).
    void run(Frame& frame, const SCFInitialGuess* guess = nullptr);
    /// @brief Const-geometry energy (does not write Frame meta or move
    ///        atoms). Results via `energy()`/`converged()`/`determinant()`.
    /// @param frame  Geometry; left untouched.
    /// @param guess  Optional density guess.
    void run(const Frame& frame, const SCFInitialGuess* guess = nullptr);

    /// @brief Analytic forces [eV/Å] after (or including) a fresh SCF.
    /// @param frame  Geometry; forces written to fx/fy/fz and energy to meta.
    /// @param guess  Optional density-matrix guess for the SCF (needed for
    ///               multi-minima / continuous FD paths such as ROHF Hessian).
    /// @return Forces + energy in eV / eV·Å⁻¹.
    /// @throws std::logic_error when `cpmd` is set (nested MD force slot only;
    ///         see `run`).
    GradientResult forces(Frame& frame, const SCFInitialGuess* guess = nullptr);

    /// @brief The wavefunction of the most recent run (last iterate when not
    ///        converged). Empty (0 electrons, 0 orbitals) before any run.
    /// @return The owned Slater determinant (Hartree internals).
    const SlaterDeterminant& determinant() const { return *det_sp_; }
    /// @brief Total SCF energy of the most recent run.
    /// @return E_total [Hartree] (includes E_disp when `d3bj`/`d4` is set).
    double energy() const { return e_total_; }
    /// @brief DFT-D3(BJ)/DFT-D4 dispersion contribution of the most recent run.
    ///        Zero when both `d3bj` and `d4` are false.
    /// @return E_disp [Hartree].
    double energy_disp() const { return e_disp_; }
    /// @brief SCF iterations executed by the most recent run.
    /// @return Iteration count (dimensionless).
    int iterations() const { return iterations_; }
    /// @brief Macro iterations the CIAH stepper executed in the most recent
    ///        run (0 when `soscf` is off or never activated) — the count
    ///        comparable to a PySCF `mf.newton()` macro-cycle count, since
    ///        each unit is one Fock build in Newton mode.
    /// @return SOSCF macro iteration count (dimensionless).
    int soscf_iterations() const {
        return soscf_stepper_ ? soscf_stepper_->macro_steps() : 0;
    }
    /// @brief Whether a result has been produced by any run.
    bool has_result() const { return *has_result_sp_; }

    /// @brief Observer handle (shared_ptrs survive driver ownership moves).
    /// @return Shared view of determinant / shells for cube writers.
    MeanFieldOutputs result_handle() const {
        return {
            det_sp_,
            std::shared_ptr<const std::vector<native::Shell>>(ints_shells_sp_),
            has_result_sp_,
            std::shared_ptr<const xt::xtensor<double, 2>>(ints_c2s_sp_)};
    }

    Derived& with_gradient(bool enabled) & noexcept {
        compute_force = enabled;
        return static_cast<Derived&>(*this);
    }
    Derived&& with_gradient(bool enabled) && noexcept {
        compute_force = enabled;
        return static_cast<Derived&&>(*this);
    }

    Derived& predictor(int order = 3, int corrector_steps = 2,
                       double corrector_tol = 1e-4, bool fallback = true) & {
        set_predictor_(order, corrector_steps, corrector_tol, fallback);
        return static_cast<Derived&>(*this);
    }
    Derived&& predictor(int order = 3, int corrector_steps = 2,
                        double corrector_tol = 1e-4, bool fallback = true) && {
        set_predictor_(order, corrector_steps, corrector_tol, fallback);
        return static_cast<Derived&&>(*this);
    }

    /// @brief Immutable XC provider retained by this method (null for HF/ROKS).
    /// @return Shared provider for independent geometry evaluations.
    const std::shared_ptr<const ATV::electronic::XcFunctional>&
    xc_functional() const {
        return xc_functional_;
    }

  protected:
    /// @brief Admit a CPU molecular XC provider and retain its lifetime.
    /// @param functional Immutable provider. @param spin Method spin convention.
    /// @return Built-in identity or Custom for an external semi-local provider.
    xc::XCType select_functional_(
        std::shared_ptr<const ATV::electronic::XcFunctional> functional,
        ATV::electronic::XcSpin spin);
    std::shared_ptr<const ATV::electronic::XcFunctional> xc_functional_;

    void set_predictor_(int order, int corrector_steps, double corrector_tol,
                        bool fallback);

    /// @brief Stages are fixed in `ensure_driver_built_`; nothing is
    ///        user-suppliable, so there is nothing to validate.
    void validate_components() const override {}

    /// @brief Fixed by the concrete subclass (RHF/UHF/RKS/UKS).
    SpinType spin_ = SpinType::Restricted;
    MeanFieldType method_ = MeanFieldType::HF;
    xc::XCType xc_type_ = xc::XCType::LSDA;
    int n_alpha_ = 0;
    int n_beta_ = 0;

    /// @brief Set n_alpha/n_beta from electron count (and subclass rules).
    virtual void resolve_occupation(int n_electrons) = 0;

    /// @brief Materialize working-basis ERIs only for a post-HF consumer.
    void materialize_eri_();

    /// @brief Non-owning view of this method's private integral **members**
    ///        (shells, S, c2s, dense ERI when materialized) for derived
    ///        post-run analyses (`Stability`, `TDA`). Not a bag type —
    ///        only binds refs to the expanded fields below. Never public.
    [[nodiscard]] internal::IntegralView integral_view_() const {
        return internal::IntegralView{
            ints_S_,
            ints_T_,
            ints_V_,
            ints_Hcore_,
            ints_W_,
            ints_ERI_,
            ints_V_nuc_,
            ints_n_basis_,
            ints_n_basis_cart_,
            *ints_c2s_sp_,
            ints_n_electrons_,
            ints_ecp_,
            ints_has_ecp_,
            ints_ecp_centers_,
            ints_embed_charges_,
            ints_embed_self_energy_,
            ints_external_field_,
            ints_solvent_eps_,
            ints_solvent_radii_scale_,
            ints_solvent_x_,
            ints_solvent_model_,
            ints_atoms_,
            *ints_shells_sp_,
            ints_shell_data_,
        };
    }

  private:
    // ── Per-run SCF iteration state — plainly-named private members ────────────
    //
    // Not a state bag: each field is wired into a stage component by ctor
    // reference in `ensure_driver_built_` (const& reads, & writes), per the
    // driver-world ruling against `*Workspace` state bags. xtensor identity is
    // stable, so the per-run resize inside `drive_scf_` never invalidates a
    // bound reference; a move of the method drops the wiring
    // (`reset_built_driver_`) so the next `drive_scf_` re-binds against the
    // moved-to object's members — no dangling reference.
    RealMatrix S_;         ///< Overlap matrix (dimensionless).
    RealMatrix Hcore_;     ///< Core Hamiltonian [Hartree].
    RealMatrix X_;         ///< Orthogonalization transform.
    RealMatrix F_alpha_;   ///< α Fock matrix [Hartree].
    RealMatrix F_beta_;    ///< β Fock matrix [Hartree].
    RealMatrix J_;         ///< Coulomb matrix [Hartree].
    RealMatrix err_alpha_; ///< α DIIS error channel.
    RealMatrix err_beta_;  ///< β DIIS error channel.
    RealMatrix C_alpha_;   ///< α MO coefficients.
    RealMatrix C_beta_;    ///< β MO coefficients.
    RealVector eig_alpha_; ///< α orbital energies [Hartree].
    RealVector eig_beta_;  ///< β orbital energies [Hartree].
    RealMatrix P_alpha_;   ///< α density (old-in / new-out).
    RealMatrix P_beta_;    ///< β density (old-in / new-out).
    // ROHF-only channels (untouched in the other spin modes): the couple
    // stage preserves the physical spin Focks here (the density stage's
    // energy functional reads them, never F_eff) and publishes P_α+P_β for
    // the single-channel DIIS commutator.
    RealMatrix F_alpha_uncoupled_; ///< ROHF: true α Fock [Hartree].
    RealMatrix F_beta_uncoupled_;  ///< ROHF: true β Fock [Hartree].
    RealMatrix P_total_;           ///< ROHF: total density P_α + P_β.
    double e_total_ = 0;           ///< Total SCF energy [Hartree].
    double e_one_ = 0;             ///< One-electron energy [Hartree].
    double e_coulomb_ = 0;         ///< Coulomb (J) energy [Hartree].
    double e_xc_grid_ = 0;         ///< Grid XC energy [Hartree].
    double e_exact_exchange_ = 0;  ///< Hybrid exact-exchange energy [Hartree].
    /// Bound by `WithSolvent` only. Vacuum is eps 0.
    double solvent_eps_ = 0.0;
    double solvent_radii_scale_ = 1.2;
    double solvent_x_ = 0.0;
    SolventModel solvent_model_ = SolventModel::CPCM;
    double e_solvent_ = 0; ///< Continuum ΔG_el [Hartree] (0 when off).
    double e_disp_ = 0;    ///< DFT-D3(BJ)/D4 E_disp [Hartree] (0 when off).
    double e_pcm_fock_half_ = 0; ///< ½ Tr[V_pcm P] [Hartree] for energy peel.
    double int_n_alpha_ = 0;     ///< Grid-integrated α electron count.
    double int_n_beta_ = 0;      ///< Grid-integrated β electron count.
    double v_nuc_ = 0;           ///< Nuclear-repulsion energy [Hartree].
    int n_basis_ = 0;            ///< Basis dimension.
    int n_dropped_ = 0;          ///< Overlap eigen-directions dropped.
    int iter_ = 0;               ///< Current SCF iteration index.
    int diis_pairs_ = 0;         ///< DIIS history-size mirror.
    bool diis_active_ = false;   ///< DIIS extrapolating (≥ 2 vectors).
    bool restricted_ = false;    ///< spin_ == Restricted (per-run cache).
    bool is_hf_ = false;         ///< method_ == HF (per-run cache).

    /// MOM/IMOM reference occupied MO columns (n × n_σ). Empty when
    /// `mom == Off`. Updated in-place by DensityStage under MomStyle::MOM.
    RealMatrix mom_ref_C_alpha_;
    RealMatrix mom_ref_C_beta_;

    /// @brief Options selecting which of the shared entry behaviors apply.
    ///        Private, per-call — routing flags, not a config bag.
    struct DriveOpts {
        bool write_energy = false; ///< Write eV energy to `frame.meta`
                                   ///< (mutating entries; forces path writes
                                   ///< its own energy).
        bool forces = false;       ///< Compute + write analytic forces.
        bool unwrap = true;        ///< Unwrap positions (mutating paths only).
        int max_iter = -1;         ///< Budget override; <0 ⇒ member `max_iter`.
        double tol_override = -1;  ///< |ΔE| tol override; <0 ⇒ member `tol`.
        bool apply_aspc = true; ///< ASPC front/back; false on recursive call.
        bool nested = false;    ///< Entered from an outer loop's force/energy
                                ///< slot (the nuclear clock ticks) — XL-BOMD
                                ///< propagates only on these entries.
        /// Accumulate (`+=`) into `kEnergy` / `kEnergyElectronic` / `kAtom_f*`
        /// instead of overwriting. True on the **in-plan** entry
        /// (`run_nested_erased`), where the hosting driver owns zeroing and
        /// this method is one co-author among however many sources the loop
        /// binds. False on the **standalone** entries (`run`, `forces`,
        /// post-HF forces): "the mutating entry always writes" is
        /// their documented Frame-out contract, and overwrite is the correct
        /// self-zeroing for a sole author that cannot have a co-author.
        bool accumulate = false;
    };

    /// @brief RAII guard: unwrap positions for the integral build on entry,
    ///        restore the wrapped storage on exit (absorbs the old free
    ///        `unwrap_for_scf`/`restore_wrapped`; no free helper functions).
    class PositionGuard {
      public:
        PositionGuard(Frame& frame, bool active);
        ~PositionGuard();
        PositionGuard(const PositionGuard&) = delete;
        PositionGuard& operator=(const PositionGuard&) = delete;

      private:
        Frame& frame_;
        bool active_ = false;
        std::vector<StorageReal> x_, y_, z_;
    };

    /// @brief THE funnel: guess ladder → prologue (per-run assembly) → Driver
    ///        loop → epilogue (de-shift / spin / result fill / energy /
    ///        forces) → propagator bookkeeping.
    ///
    ///        The guess is resolved **once**, by one ordered ladder — explicit
    ///        guess > XL-BOMD aux > ASPC prediction > last-density reuse — and
    ///        only the winning rung runs: an ASPC prediction that lost is
    ///        neither used nor corrected. ASPC record and the last-density
    ///        snapshot happen on every entry, gated on `opts.apply_aspc`
    ///        (false on the corrector/fallback recursion, so it never
    ///        re-enters); the XL advance additionally needs `opts.nested`.
    /// @param frame Geometry; mutated only on the mutating entries.
    /// @param guess Optional density guess.
    /// @param opts  Which shared behaviors apply on this entry.
    void drive_scf_(Frame& frame, const SCFInitialGuess* guess,
                    const DriveOpts& opts);

    /// @brief Core SCF (no ASPC wrapper) — used by drive_scf_ after ASPC
    ///        front-end resolves the density guess.
    void drive_scf_core_(Frame& frame, const SCFInitialGuess* guess,
                         const DriveOpts& opts);

    /// @brief Rebuild private integral **members** at the current geometry
    ///        and resolve occupation — shared by `drive_scf_core_` and
    ///        `cpmd_step_`. Returns a view bound to those members.
    /// @param frame Geometry [Å]; already position-guarded by the caller.
    /// @param compute_pvp When true, assemble ⟨∇|V|∇⟩ (x2c).
    internal::IntegralView rebuild_integrals_(const Frame& frame,
                                              bool compute_pvp);

    /// @brief Lazily wire the five stages + `ScfConverged` and compile the Plan.
    ///        Deferred to first `drive_scf_` so member config (grid, level
    ///        shift, xc) is finalized before the stages bind references to the
    ///        method's members. Idempotent, and self-healing after a move: if
    ///        the object was relocated since it wired (detected via
    ///        `wired_self_`), it drops the stale wiring and rebinds to `this`.
    void ensure_driver_built_();

    /// @brief Drop the wired stages + their Plan / criterion registration and
    ///        clear `driver_built_`, so the next `drive_scf_` rebuilds. Invoked
    ///        when a move leaves the stages bound to a stale address.
    void reset_built_driver_();

    // ── Private integral members (expanded fields — not a freestanding bag) ─
    // Host AO matrices / shells / env retained across the SCF loop and post-HF.
    // `ints_shells_sp_` / `ints_c2s_sp_` are shared so result_handle can alias
    // them for cube writers without copying.
    xt::xtensor<double, 2> ints_S_;
    xt::xtensor<double, 2> ints_T_;
    xt::xtensor<double, 2> ints_V_;
    xt::xtensor<double, 2> ints_Hcore_;
    xt::xtensor<double, 2> ints_W_;
    xt::xtensor<double, 4> ints_ERI_;
    double ints_V_nuc_ = 0;
    int ints_n_basis_ = 0;
    int ints_n_basis_cart_ = 0;
    int ints_n_electrons_ = 0;
    std::string ints_ecp_;
    bool ints_has_ecp_ = false;
    std::vector<internal::EcpCenter> ints_ecp_centers_;
    std::vector<cpu::EmbeddingSite> ints_embed_charges_;
    bool ints_embed_self_energy_ = true;
    std::array<double, 3> ints_external_field_{{0.0, 0.0, 0.0}};
    double ints_solvent_eps_ = 0;
    double ints_solvent_radii_scale_ = 1.2;
    double ints_solvent_x_ = 0;
    SolventModel ints_solvent_model_ = SolventModel::CPCM;
    std::vector<native::Atom> ints_atoms_;
    std::shared_ptr<std::vector<native::Shell>> ints_shells_sp_{
        std::make_shared<std::vector<native::Shell>>()};
    std::vector<std::vector<double>> ints_shell_data_;
    std::shared_ptr<xt::xtensor<double, 2>> ints_c2s_sp_{
        std::make_shared<xt::xtensor<double, 2>>()};

    // Determinant slot is stable (handle-before-run contract).
    std::shared_ptr<SlaterDeterminant> det_sp_{
        std::make_shared<SlaterDeterminant>(RealMatrix{}, RealVector{},
                                            RealVector{}, 0)};
    std::shared_ptr<bool> has_result_sp_{std::make_shared<bool>(false)};
    int iterations_ = 0; ///< SCF iterations executed by the most recent run.

    /// Continuum (C-PCM / COSMO / IEF-PCM) private workspace (null when
    /// solvent_eps == 0). Built in drive_scf_core_ before the loop;
    /// FockStage borrows the raw pointer.
    std::unique_ptr<internal::Pcm> pcm_;

    // Stage components (heap-owned; borrowed into the Plan by add_ref). Null
    // until ensure_driver_built_ wires them to this method's members.
    std::unique_ptr<FockStage> fock_stage_;
    std::unique_ptr<RohfCoupleStage> rohf_couple_stage_;
    std::unique_ptr<MixStage> mix_stage_;
    std::unique_ptr<DiagonalizeStage> diag_stage_;
    std::unique_ptr<DensityStage> density_stage_;
    std::unique_ptr<ScfConverged> converged_stage_;
    /// CIAH workspace (DIISMixer precedent: internal class, not a
    /// Component); null unless `soscf` was set when the driver wired.
    std::unique_ptr<internal::SOSCFStepper> soscf_stepper_;
    /// SOSCF activation latch: set by MixStage when max|DIIS error| falls
    /// under `soscf_start`, read by DiagonalizeStage; reset each run.
    bool soscf_active_ = false;
    bool driver_built_ = false;
    /// Address the stages were wired to; a mismatch on `this` flags a move.
    const void* wired_self_ = nullptr;
    /// Parent Runtime of a nested mount (`build_on`); the SCF stage plan runs
    /// on it instead of a private backend. Null for standalone entries.
    cpu::Runtime* host_runtime_ = nullptr;
    /// The stage plan is registered on `host_runtime_` for the solve in
    /// progress (built at its start, released at its end or on unwind).
    bool host_registered_ = false;
    /// @brief Release the stage plan from the parent Runtime, if registered.
    void release_host_plan_();

    // Most recent analytic gradient (forces() return + force-slot write).
    GradientResult last_grad_;
    /// Republished from `last_grad_.embedding_forces` after each force eval.
    std::vector<cpu::EmbeddingForce> embed_site_forces_;

    // ── Car–Parrinello state (cpmd = true only) ───────────────────────
    /// Occupied-orbital propagators: α always, β only for UHF/UKS. Engaged
    /// (`ready()`) from the cold seed onward; `std::nullopt` under BOMD.
    std::optional<electronic::Cpmd> cpmd_alpha_;
    std::optional<electronic::Cpmd> cpmd_beta_;
    double cpmd_dt_fs_ = 0; ///< Nuclear timestep pushed by the integrator [fs].
    double last_te_ev_ = 0; ///< Last fictitious T_e [eV].
    double last_etherm_ev_ = 0; ///< Last electronic thermostat energy [eV].

    /// @brief Refuse every configuration whose CPMD semantics are not
    ///        productized, before any electron moves.
    ///
    /// Static combinations (spin shape, competing propagator, Fock builds the
    /// CPMD gradient does not cover, energy terms with no force counterpart)
    /// throw `std::invalid_argument`; state discovered while driving (a
    /// near-singular overlap at the seed) throws `std::runtime_error` from
    /// `cpmd_seed_`.
    /// @throws std::invalid_argument on ROHF/ROKS/GHF, ASPC predictor, RI,
    ///         x2c, solvent, MOM, SOSCF, point charges, external field,
    ///         d3bj/d4, or invalid μ / thermostat parameters.
    void validate_cpmd_config_() const;

    /// @brief Seed the propagators from the just-converged occupied
    ///        coefficients (S-orthonormal by construction).
    /// @throws std::runtime_error when the seed SCF has no usable
    ///         wavefunction or dropped overlap directions (`n_dropped_ > 0`).
    void cpmd_seed_();

    // ── Extended-Lagrangian BOMD state (xl_bomd = true only) ──────────
    /// Auxiliary-density propagators: `xl_alpha_` always (the factor-2 total
    /// density when restricted, the spinor density under GHF, the α channel
    /// otherwise), `xl_beta_` only where an independent β density exists
    /// (UHF/UKS/ROHF/ROKS). Constructed on the first nested entry, `nullopt`
    /// under BOMD.
    std::optional<electronic::XlBomd> xl_alpha_;
    std::optional<electronic::XlBomd> xl_beta_;
    int xl_advances_ = 0; ///< Auxiliary propagations taken (public witness).

    /// @brief Refuse the configurations that give XL-BOMD a second guess owner
    ///        and construct the propagators, before any electron moves.
    ///
    /// Called on every nested entry and idempotent: it constructs (or, after a
    /// `xl_bomd_k` change, re-seats) the propagators exactly when `xl_bomd` is
    /// set, and drops them when it is not. Construction is where the Table-I
    /// order is validated, so a bad K is reported before an SCF is spent
    /// rather than after the first converged solve.
    /// @throws std::invalid_argument on `xl_bomd` with the ASPC predictor, on
    ///         `xl_bomd` with `cpmd`, or on `xl_bomd_k` outside [3, 9].
    void engage_xl_bomd_();

    /// @brief One production Car–Parrinello step: rebuild integrals at the
    ///        new geometry, transport C under the new metric, evaluate the
    ///        energy and forces at the propagated density, then advance the
    ///        electrons one velocity-Verlet/RATTLE step.
    /// @param frame Geometry in / forces + energies out (accumulated with
    ///        `+=`, since MD zeroes them at `pre_force`).
    /// @param advance_electrons False for force preparation: evaluate a copy
    ///        of the electronic state without transport/history mutations.
    void cpmd_step_(Frame& frame, bool advance_electrons);

    // ASPC (density extrapolation across outer MD steps)
    bool use_predictor_ = false;
    int corrector_steps_ = 2;
    double corrector_tol_ = 1e-4;
    bool fallback_full_scf_ = true;
    std::unique_ptr<electronic::ASPCPredictor> aspc_alpha_;
    std::unique_ptr<electronic::ASPCPredictor> aspc_beta_;
    bool has_last_density_ = false;
    xt::xtensor<double, 2> last_P_alpha_;
    xt::xtensor<double, 2> last_P_beta_;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
