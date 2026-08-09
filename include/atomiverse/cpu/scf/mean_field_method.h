/// @file cpu/scf/mean_field_method.h
/// @brief Shared CRTP (curiously recurring template pattern) base for the
///        concrete Gaussian mean-field methods: `MeanFieldMethod<D>` (base of
///        RHF / ROHF / UHF / RKS / UKS).
///
/// There is no bag named `SCFSolver` / freestanding integral bag / `SCFConfig`.
/// Each method is a concrete class; parameters are **constructor arguments**
/// (e.g. `RHF{BasisSet::STO3G, charge, tol}`); **never** `rhf.basis = …`
/// after default construction. The only I/O currency is `Frame`. Integral
/// **fields** live as method-private members; engines take
/// `internal::IntegralView` (non-owning refs).
///
/// SCF self-consistency is a **Driver loop**: `MeanFieldMethod<D>` derives
/// `cpu::Driver<ATV::scf::Phase>` and the pure tag `HF` (not a Component).
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
#include <functional>
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
#include "atomiverse/cpu/scf/casci.h"
#include "atomiverse/cpu/scf/casscf.h"
#include "atomiverse/cpu/scf/cc2.h"
#include "atomiverse/cpu/scf/ccsd.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/fci.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/mp2.h"
#include "atomiverse/cpu/scf/mp3.h"
#include "atomiverse/cpu/scf/mp4.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/nevpt2.h"
#include "atomiverse/cpu/scf/scf_enums.h"
#include "atomiverse/cpu/scf/scf_guess.h"
#include "atomiverse/cpu/scf/slater_determinant.h"
#include "atomiverse/cpu/scf/stages/density_stage.h"
#include "atomiverse/cpu/scf/stages/diagonalize_stage.h"
#include "atomiverse/cpu/scf/stages/fock_stage.h"
#include "atomiverse/cpu/scf/stages/mix_stage.h"
#include "atomiverse/cpu/scf/stages/rohf_couple_stage.h"
#include "atomiverse/cpu/scf/stages/scf_converged.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

namespace ATV {
namespace cpu {

namespace electronic {
class ASPCPredictor; // Kolafa density extrapolator (aspc_predictor.h)
}

namespace scf {

namespace internal {
class Pcm; // continuum solvent private workspace (pcm.h)
}

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
    // ── Geometry / basis (inputs from Frame + these members) ──────────
    BasisSet basis = BasisSet::STO3G;
    int charge = 0;

    /// @brief Local atom indices treated as ghosts (BSSE counterpoise).
    ///
    /// Ghost atoms contribute basis functions to the AO space but carry
    /// zero nuclear charge and contribute no electrons (Boys–Bernardi
    /// counterpoise: basis-only sites). Empty = no ghosts (default;
    /// bit-identical to the pre-ghost path). Ghost markers live on the
    /// method, never on Frame (Frame purity; molrs Element forbids Z ≤ 0).
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
    /// Analytic forces with x2c=true throw (h^X2C gradient not derived).
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

    /// @brief Number of lowest occupied spatial MOs excluded from the MP2
    ///        correlation sum (frozen core). Default 0 = correlate all
    ///        occupied orbitals. Matches PySCF `mp.MP2(mf, frozen=N)`.
    ///        Closed-shell RMP2 / RI-MP2 only; UMP2 throws if `frozen > 0`.
    int frozen = 0;

    /// @brief Opposite-spin scale for MP2 correlation (dimensionless).
    ///
    /// Default 1.0 recovers plain RMP2/UMP2/RI-MP2. Grimme SCS-MP2 uses
    /// 1.2 (with `mp2_ss_scale = 1/3`); SOS-MP2 uses 1.3 (with
    /// `mp2_ss_scale = 0`). Applied as
    /// \(E_{\mathrm{corr}}=c_{\mathrm{OS}}E_{\mathrm{OS}}+c_{\mathrm{SS}}E_{\mathrm{SS}}\);
    /// unscaled channels remain on `MP2Result::e_corr_os` /
    /// `e_corr_ss`.
    double mp2_os_scale = 1.0;

    /// @brief Same-spin scale for MP2 correlation (dimensionless).
    ///
    /// Default 1.0 = plain MP2; SCS-MP2 = 1/3; SOS-MP2 = 0.
    /// See `mp2_os_scale`.
    double mp2_ss_scale = 1.0;

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

    /// @brief When true, materialize the dense ERI tensor (MP2/FCI/CUDA).
    bool compute_eri = false;

    /// @brief When true (default), nested force-slot runs write forces.
    bool compute_force = true;

    /// @brief Accelerate the closed-shell CCSD amplitude iteration with Pulay
    ///        amplitude DIIS (`internal::CcAmplitudeDiis`). Default true.
    ///        When false the closed-shell energy path (`RHF::ccsd`) uses plain
    ///        Jacobi updates — the same fixed point, only slower — which is the
    ///        DIIS-vs-Jacobi parity leg of the CCSD regression. Dimensionless.
    bool cc_diis = true;

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

    // ── Implicit solvent (hf-capability-gaps-16-solvent) ───────────────
    /// @brief Solvent dielectric constant ε (dimensionless). Default 0
    ///        disables continuum solvent entirely (bit-identical vacuum
    ///        path). Typical water: 78.4.
    double solvent_eps = 0.0;

    /// @brief Multiplier on Bondi vdW radii for the cavity spheres
    ///        (dimensionless; default 1.2, Q-Chem / PySCF convention).
    double solvent_radii_scale = 1.2;

    /// @brief Dielectric scaling convention in f(ε)=(ε−1)/(ε+x) for
    ///        conductor-like models: 0 → C-PCM (Barone–Cossi / PySCF
    ///        default); 0.5 → original COSMO (Klamt). Ignored when
    ///        `solvent_model == SolventModel::IEFPCM` (which uses
    ///        f=(ε−1)/(ε+1)). Dimensionless.
    double solvent_x = 0.0;

    /// @brief Continuum model. Default `CPCM` (conductor-like). Set
    ///        `IEFPCM` for integral-equation-formalism PCM (Cancès–
    ///        Mennucci–Tomasi; PySCF `method='IEF-PCM'`).
    SolventModel solvent_model = SolventModel::CPCM;

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

    MeanFieldMethod();
    ~MeanFieldMethod() override;
    MeanFieldMethod(MeanFieldMethod&&) noexcept;
    MeanFieldMethod& operator=(MeanFieldMethod&&) noexcept;
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

    /// @brief Canonical MP2 on top of a fresh SCF run (materializes ERI).
    ///
    /// Closed-shell `RHF` → RMP2 (or RI-MP2 when `ri != RI::None`);
    /// unrestricted `UHF` → UMP2 (spin-orbital, unprojected);
    /// restricted-open `ROHF` → ROMP2 (PySCF: shared spatial MOs +
    /// semi-canonical \(\varepsilon_i^\sigma = (C^\dagger F_\sigma C)_{ii}\),
    /// then the UMP2 doubles sum). Closed-shell `RKS` with a double-hybrid
    /// XC (e.g. `XCType::B2PLYP`) is the same RMP2 kernel on the KS
    /// orbitals — set `mp2_os_scale = mp2_ss_scale = hybrid_pt2_fraction(xc)`
    /// so \(E_{\mathrm{tot}}=E_{\mathrm{KS}}+a_c E_{\mathrm{MP2}}\)
    /// (Grimme B2PLYP: a_c = 0.27). GHF throws. Optional density guess is
    /// forwarded to the reference SCF (needed for multi-minima cases such as
    /// triplet O₂ UHF). Member `frozen` excludes the lowest N occupied spatial
    /// MOs from the correlation sum (RMP2 / RI-MP2); UMP2/ROMP2 reject
    /// `frozen > 0`. Members `mp2_os_scale` / `mp2_ss_scale` apply Grimme
    /// spin-component scaling (defaults 1 / 1 = plain MP2; SCS-MP2 = 1.2 / ⅓;
    /// B2PLYP = 0.27 / 0.27).
    /// @param frame  Geometry; drives a fresh SCF then adds the correction.
    /// @param guess  Optional density-matrix guess for the reference SCF.
    /// @return MP2 correlation + total energy [Hartree]; unscaled OS/SS on
    ///         the result fields.
    MP2Result mp2(Frame& frame, const SCFInitialGuess* guess = nullptr);

    /// @brief Analytic nuclear forces on the total MP2 energy E_HF + E_corr
    ///        [eV/Å]. Canonical RMP2 (RHF) or UMP2 (UHF); no frozen core, no
    ///        RI, Cartesian AO. Drives a fresh SCF (materializes ERI), builds
    ///        the MP2 Lagrangian densities + Z-vector response, and contracts
    ///        with skeleton integral derivatives. Matches PySCF
    ///        `mp.MP2` / `mp.UMP2` `.nuc_grad_method().kernel()` on the same
    ///        geometry.
    /// @param frame  Geometry; forces written to fx/fy/fz; energy to meta.
    /// @param guess  Optional density-matrix guess for the reference SCF
    ///               (needed for multi-minima UHF cases such as triplet O₂).
    /// @return Forces [eV/Å] and total MP2 energy [eV].
    /// @throws std::invalid_argument on ROHF / RI / frozen>0 / SCS scales /
    ///         non-converged SCF / spherical basis (no fallback).
    GradientResult mp2_forces(Frame& frame,
                              const SCFInitialGuess* guess = nullptr);

    /// @brief Canonical closed-shell MP3 on top of a fresh RHF run
    ///        (materializes ERI). \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}\)
    ///        in the spin-orbital Szabo–Ostlund / Psi4 diagram sum. Open-shell,
    ///        RI, and SCS scales are rejected (no fallback). Member `frozen`
    ///        freezes the lowest N occupied spatial MOs.
    /// @param frame  Geometry; drives a fresh SCF then adds the correction.
    /// @param guess  Optional density-matrix guess for the reference SCF.
    /// @return MP3 correlation pieces + total energy [Hartree].
    MP3Result mp3(Frame& frame, const SCFInitialGuess* guess = nullptr);

    /// @brief Canonical closed-shell MP4(SDTQ) on top of a fresh RHF run
    ///        (materializes ERI). \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}+E^{(4)}\)
    ///        with fourth-order singles, doubles, triples, and renormalization
    ///        (disconnected-Q) channels. Open-shell, RI, and SCS scales are
    ///        rejected (no fallback). Member `frozen` freezes the lowest N
    ///        occupied spatial MOs.
    /// @param frame  Geometry; drives a fresh SCF then adds the correction.
    /// @param guess  Optional density-matrix guess for the reference SCF.
    /// @return MP4 correlation pieces + total energy [Hartree].
    MP4Result mp4(Frame& frame, const SCFInitialGuess* guess = nullptr);

    /// @brief Tiny full-CI on top of a fresh RHF-style run (materializes ERI).
    /// @param frame  Geometry; drives a fresh SCF then adds the correction.
    /// @return Full-CI correlation + total energy.
    FCIResult fci(Frame& frame);

    /// @brief Closed-shell CASCI on top of a fresh RHF run (materializes ERI).
    ///
    /// Dense complete-active-space CI: frozen doubly-occupied core + full CI
    /// in `ncas` spatial orbitals with `nelecas` active electrons. Matches
    /// PySCF `mcscf.CASCI(mf, ncas, nelecas)` on the same RHF orbitals.
    /// Open-shell (UHF/ROHF/GHF) throws — no silent fallback.
    /// @param frame   Geometry; drives a fresh SCF then the active-space CI.
    /// @param ncas    Number of active spatial orbitals (dimensionless).
    /// @param nelecas Number of active electrons (even; closed-shell).
    /// @return CASCI total + correlation energy [Hartree].
    CASCIResult casci(Frame& frame, int ncas, int nelecas);

    /// @brief Closed-shell CASSCF on top of a fresh RHF run (materializes ERI).
    ///
    /// Dense complete-active-space CI with orbital optimization: frozen
    /// doubly-occupied core + full CI in `ncas` spatial orbitals; non-redundant
    /// orbital rotations (core–active, active–virtual, core–virtual) by
    /// diagonal Newton on the generalized Fock gradient. Matches PySCF
    /// `mcscf.CASSCF(mf, ncas, nelecas)` on the same closed-shell foothold.
    /// Open-shell (UHF/ROHF/GHF) throws — no silent fallback.
    /// @param frame   Geometry; drives a fresh SCF then the CASSCF macro loop.
    /// @param ncas    Number of active spatial orbitals (dimensionless).
    /// @param nelecas Number of active electrons (even; closed-shell).
    /// @return CASSCF total + correlation energy [Hartree].
    CASSCFResult casscf(Frame& frame, int ncas, int nelecas);

    /// @brief Closed-shell strongly contracted NEVPT2 on a fresh RHF→CASSCF
    ///        reference (materializes ERI).
    ///
    /// Runs CASSCF(ncas, nelecas) then the eight Angeli SC-NEVPT2 perturber
    /// classes (PySCF `mrpt.NEVPT`). On a full-valence CAS with no external
    /// orbitals the PT2 correction is identically zero. Open-shell
    /// (UHF/ROHF/GHF) throws — no silent fallback.
    /// @param frame   Geometry; drives a fresh SCF then CASSCF + NEVPT2.
    /// @param ncas    Number of active spatial orbitals (dimensionless).
    /// @param nelecas Number of active electrons (even; closed-shell).
    /// @return NEVPT2 total + PT2 correlation [Hartree] vs CASSCF.
    NEVPT2Result nevpt2(Frame& frame, int ncas, int nelecas);

    /// @brief Closed-shell CCSD on top of a fresh RHF run (materializes ERI).
    ///
    /// Spin-orbital Gauss–Stanton CCSD (Stanton et al. 1991; Gauss & Stanton
    /// 1995). Matches PySCF `cc.CCSD` on the same RHF reference. Open-shell
    /// (UHF/ROHF/GHF) throws — no silent fallback.
    /// @param frame Geometry; drives a fresh SCF then the amplitude solve.
    /// @return CCSD correlation + total energy [Hartree], iteration count.
    CCSDResult ccsd(Frame& frame);

    /// @brief Nuclear forces on the total closed-shell CCSD energy
    ///        E_HF + E_CCSD [eV/Å]. Drives a fresh RHF (materializes ERI) then
    ///        forms forces by central finite difference of the analytic CCSD
    ///        energy (h = 1e-4 Bohr). Matches PySCF
    ///        `cc.CCSD.nuc_grad_method().kernel()` force values on the same
    ///        geometry (Cartesian AO, no frozen core) to the regression gate.
    ///        Open-shell / RI / frozen-core throw — no silent fallback.
    /// @param frame Geometry; forces written to fx/fy/fz; energy to meta.
    /// @param guess Optional density-matrix guess for the reference SCF.
    /// @return Forces [eV/Å] and total CCSD energy [eV].
    GradientResult ccsd_forces(Frame& frame,
                               const SCFInitialGuess* guess = nullptr);

    /// @brief Closed-shell CC2 on top of a fresh RHF run (materializes ERI).
    ///
    /// Approximate CC doubles (Christiansen, Koch, Jørgensen, Chem. Phys.
    /// Lett. **243**, 409 (1995)): full singles residual, doubles residual
    /// truncated to ≤ linear in T₂ with the T₁-dressed Fock. Matches PySCF
    /// `cc.RCCSD` with `cc2=True`. Open-shell throws — no silent fallback.
    /// @param frame Geometry; drives a fresh SCF then the amplitude solve.
    /// @return CC2 correlation + total energy [Hartree], iteration count.
    CC2Result cc2(Frame& frame);

    /// @brief Closed-shell CCSD(T) on top of a fresh RHF run (materializes ERI).
    ///
    /// Runs the same CCSD amplitude solve as `ccsd`, then the non-iterative
    /// (T) correction (Raghavachari 1989; Watts–Gauss–Bartlett 1993). Matches
    /// PySCF `cc.CCSD(...).ccsd_t()` on the same RHF reference. Open-shell
    /// throws — no silent fallback.
    /// @param frame Geometry; drives a fresh SCF then CCSD + (T).
    /// @return CCSD(T) correlation pieces + total energy [Hartree].
    CCSDTResult ccsd_t(Frame& frame);

    /// @brief The wavefunction of the most recent run (last iterate when not
    ///        converged). Empty (0 electrons, 0 orbitals) before any run.
    /// @return The owned Slater determinant (Hartree internals).
    const SlaterDeterminant& determinant() const { return *det_sp_; }
    /// @brief Total SCF energy of the most recent run.
    /// @return E_total [Hartree] (includes E_disp when `d3bj`/`d4` is set).
    double energy() const { return e_total_; }
    /// @brief Electrostatic continuum free-energy contribution of the most
    ///        recent run (ΔG_el = ½ Σ_k q_k V_k). Zero when
    ///        `solvent_eps == 0`. Non-electrostatic terms are not included.
    ///        Valid for C-PCM, COSMO, and IEF-PCM.
    /// @return ΔG_el [Hartree].
    double e_solvent() const { return e_solvent_; }
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
        return {det_sp_,
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

  protected:
    void set_predictor_(int order, int corrector_steps, double corrector_tol,
                        bool fallback);

    /// @brief Stages are fixed in `ensure_driver_built_`; nothing is
    ///        user-suppliable, so there is nothing to validate.
    void validate_components() const override {}

    /// @brief Fixed by the concrete subclass (RHF/UHF/RKS/UKS).
    SpinType spin_ = SpinType::Restricted;
    MeanFieldType method_ = MeanFieldType::HF;
    XCType xc_type_ = XCType::LSDA;
    int n_alpha_ = 0;
    int n_beta_ = 0;

    /// @brief Set n_alpha/n_beta from electron count (and subclass rules).
    virtual void resolve_occupation(int n_electrons) = 0;

    /// @brief Non-owning view of this method's private integral **members**
    ///        (shells, S, c2s, dense ERI when materialized) for derived
    ///        post-run analyses (e.g. `RHF::stability()`). Not a bag type —
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

    /// @brief Central finite-difference nuclear Hessian [eV/Å²] from analytic
    ///        forces — the shared skeleton of the RKS/UKS/ROHF FD Hessian.
    ///
    /// Snapshots the reference geometry r0, then loops atoms (outer) × axes
    /// (inner) with the atom-major/axis-minor column index `col = 3A+d`.
    /// For each column it evaluates the caller's forces functor at +h then
    /// −h (step **h = 1e-4 Bohr**, frozen to the offline PySCF pin), central-
    /// differences `H_{row,col} = −(F(+h)−F(−h))/(2h)` with the eV/Å↔Ha/Bohr
    /// unit constants, restores the geometry to r0, and symmetrizes
    /// `½(H+Hᵀ)` to absorb FD noise. The displacement loop is serial (no
    /// added parallelism — the float reduction order is a frozen replay
    /// invariant). Constructing a fresh solver per ±h displacement (no
    /// DIIS/MO warm-start bleed) and throwing on non-convergence are the
    /// **functor's** responsibility; this skeleton only differences forces.
    /// @param frame     Geometry (Å, atom-major); mutated during the sweep and
    ///                  restored to r0 on return. The reference wavefunction /
    ///                  meta energy is the caller's to rebuild after return.
    /// @param forces_at Analytic forces at the frame's current geometry
    ///                  [eV/Å]; called 2·3N times (fresh solver each call).
    /// @return Symmetric nuclear Hessian [eV/Å²], shape [3N, 3N], `col = 3A+d`.
    vib::HessianResult central_difference_nuclear_hessian(
        Frame& frame, const std::function<GradientResult(Frame&)>& forces_at);

    /// @brief Throw if the converged reference cannot support an analytic
    ///        second derivative: SCF non-convergence, spherical AOs, or ECP.
    /// @param method_name Fully-qualified method label, e.g. "RHF::hessian".
    /// @param ints        View of this method's integral members.
    /// @throws std::runtime_error on non-convergence / spherical / ECP.
    void check_hessian_preconditions(std::string_view method_name,
                                     const internal::IntegralView& ints) const;

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
    double e_solvent_ = 0;         ///< Continuum ΔG_el [Hartree] (0 when off).
    double e_disp_ = 0; ///< DFT-D3(BJ)/D4 E_disp [Hartree] (0 when off).
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
        /// `mp2_forces`, `ccsd_forces`): "the mutating entry always writes" is
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
    internal::IntegralView rebuild_integrals_(const Frame& frame, bool compute_pvp);

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
    void cpmd_step_(Frame& frame);

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
