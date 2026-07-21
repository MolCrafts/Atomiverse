/// @file cpu/scf/mean_field_method.h
/// @brief Shared CRTP (curiously recurring template pattern) base for the
///        concrete Gaussian mean-field methods: `MeanFieldMethod<D>` (base of
///        RHF / ROHF / UHF / RKS / UKS).
///
/// There is no bag named `SCFSolver` and no user-facing `SCFConfig` /
/// `IntegralPack` step. Each method is a concrete class; parameters are
/// **members**; the only I/O currency is `Frame`.
///
/// SCF self-consistency is a **Driver loop**: `MeanFieldMethod<D>` derives
/// `cpu::CPUDriver<ATV::scf::Phase>` and the pure tag `HF` (not a Component).
/// Nested under an outer MD/Evaluate/Minimize force or energy phase it runs
/// via `run_nested` → `drive_scf_`. Standalone entry is `run(frame)`. Integrals
/// and AO workspaces are private. The concrete methods live in `rhf.h` /
/// `uhf.h` / `rks.h` / `uks.h`; `methods.h` is their umbrella.
///
/// @code
///   cpu::scf::RHF rhf;
///   rhf.basis = BasisSet::STO3G;
///   rhf.tol = 1e-10;
///   rhf.run(frame);                    // eV energy on frame.meta[kEnergy];
///                                      // Hartree on rhf.energy();
///                                      // wavefunction on rhf.determinant()
///   // forces: rhf.with_gradient(true) then nest at force phase, or
///   //         GradientResult g = rhf.forces(frame);
/// @endcode
#pragma once

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/driver/cpu_driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/fci.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/mp2.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/gradient_result.h"
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

namespace ATV {
namespace cpu {
namespace scf {

namespace internal {
class Pcm; // C-PCM private workspace (pcm.h)
}

/// @brief Shared view for observers/recorders after a method run.
///
/// `shells` is the basis-function set (pointer storage owned by the method's
/// private integral workspace via shared_ptr aliasing). Spin treatment is
/// read from the wavefunction (`result->restricted()`). `IntegralPack` is
/// not public.
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
/// (`RHF{}.predictor(3)`) keeps the concrete type for `Driver::add`.
///
/// Derives `cpu::CPUDriver<ATV::scf::Phase>` (the self-consistency loop node)
/// and pure tag `HF` (not a Component). Nested binding is a native driver node;
/// the sole compute entry is `drive_scf_` (via `run` / `forces` / nested
/// `run_nested_erased`).
template <typename Derived>
class MeanFieldMethod : public cpu::CPUDriver<ATV::scf::Phase>, public HF {
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
    bool force_native_two_electron_deriv = false;
    bool xc_grad_project_translation = false;
    double hybrid_alpha_override = -1.0;
    const MolecularGrid* fixed_grid = nullptr;

    /// @brief When true, materialize the dense ERI tensor (MP2/FCI/CUDA).
    bool compute_eri = false;

    /// @brief When true (default), nested force-slot runs write forces.
    bool compute_force = true;

    // ── Environment (hf-capability-gaps-15-embed) ─────────────────────
    /// @brief External point charge (QM/MM electrostatic embedding, QM side).
    ///
    /// Positions in Å, charge in e. Empty vector = no embedding (bit-identical
    /// to the vacuum path). Wired into the private Hcore / classical energy
    /// in `drive_scf_core_`; never stored on Frame.
    struct PointCharge {
        double x = 0; ///< Position x [Å].
        double y = 0; ///< Position y [Å].
        double z = 0; ///< Position z [Å].
        double q = 0; ///< Charge [e].
    };

    /// @brief External MM-like point charges (positions [Å], charges [e]).
    std::vector<PointCharge> point_charges;

    /// @brief Uniform external electric field (Ex, Ey, Ez) [a.u.].
    ///
    /// Origin is the center of nuclear charge (same as the multipole
    /// observer). **Finite-field collapse hazard:** keep |F| ≲ 1e-4 a.u.;
    /// a static field makes the energy unbounded below (diffuse bases can
    /// "ionize"). Zero field = vacuum path (bit-identical).
    std::array<double, 3> external_field{{0.0, 0.0, 0.0}};

    // ── Implicit solvent (hf-capability-gaps-16-solvent) ───────────────
    /// @brief Solvent dielectric constant ε (dimensionless). Default 0
    ///        disables C-PCM entirely (bit-identical vacuum path).
    ///        Typical water: 78.4.
    double solvent_eps = 0.0;

    /// @brief Multiplier on Bondi vdW radii for the cavity spheres
    ///        (dimensionless; default 1.2, Q-Chem / PySCF convention).
    double solvent_radii_scale = 1.2;

    /// @brief Dielectric scaling convention in f(ε)=(ε−1)/(ε+x):
    ///        0 → C-PCM (Barone–Cossi / PySCF default); 0.5 → original
    ///        COSMO (Klamt). Dimensionless.
    double solvent_x = 0.0;

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
    void build_on(cpu::Backend& rt) override;

    /// @brief Drop the method's private Plan / Backend wiring.
    void destroy_on(cpu::Backend& rt) override;

    /// @brief Run SCF on `frame` (driver loop). Integrals are private
    ///        workspace. Frame-out: always writes the eV energy to
    ///        `frame.meta[kEnergy]`; Hartree internals via `energy()`,
    ///        loop state via `converged()`/`stop_reason()`/`iterations()`,
    ///        wavefunction via `determinant()`.
    /// @param frame  Geometry (Å) + Z in type column.
    /// @param guess  Optional density guess.
    void run(Frame& frame, const SCFInitialGuess* guess = nullptr);
    /// @brief Const-geometry energy (does not write Frame meta or move
    ///        atoms). Results via `energy()`/`converged()`/`determinant()`.
    /// @param frame  Geometry; left untouched.
    /// @param guess  Optional density guess.
    void run(const Frame& frame, const SCFInitialGuess* guess = nullptr);

    /// @brief Analytic forces [eV/Å] after (or including) a fresh SCF.
    /// @param frame  Geometry; forces written to fx/fy/fz and energy to meta.
    /// @return Forces + energy in eV / eV·Å⁻¹.
    GradientResult forces(Frame& frame);

    /// @brief Closed-shell MP2 on top of a fresh RHF-style run (materializes ERI).
    /// @param frame  Geometry; drives a fresh SCF then adds the correction.
    /// @return MP2 correlation + total energy.
    /// @pre Derived method is closed-shell HF (use `RHF`).
    MP2Result mp2(Frame& frame);

    /// @brief Tiny full-CI on top of a fresh RHF-style run (materializes ERI).
    /// @param frame  Geometry; drives a fresh SCF then adds the correction.
    /// @return Full-CI correlation + total energy.
    FCIResult fci(Frame& frame);

    /// @brief The wavefunction of the most recent run (last iterate when not
    ///        converged). Empty (0 electrons, 0 orbitals) before any run.
    /// @return The owned Slater determinant (Hartree internals).
    const SlaterDeterminant& determinant() const { return *det_sp_; }
    /// @brief Total SCF energy of the most recent run.
    /// @return E_total [Hartree].
    double energy() const { return e_total_; }
    /// @brief Electrostatic C-PCM free-energy contribution of the most
    ///        recent run (ΔG_el = ½ Σ_k q_k V_k). Zero when
    ///        `solvent_eps == 0`. Non-electrostatic terms are not included.
    /// @return ΔG_el [Hartree].
    double e_solvent() const { return e_solvent_; }
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
                std::shared_ptr<const std::vector<native::Shell>>(
                    integrals_sp_, &integrals_sp_->shells),
                has_result_sp_,
                std::shared_ptr<const xt::xtensor<double, 2>>(
                    integrals_sp_, &integrals_sp_->c2s)};
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

    /// @brief Read access to the private integral workspace of the most
    ///        recent run (shells, S, c2s, dense ERI when materialized) for
    ///        derived-method post-run analyses (e.g. `RHF::stability()`).
    ///        Never exposed publicly — Frame stays the sole I/O currency.
    /// @return The method-owned IntegralPack (borrowed const view).
    const IntegralPack& integral_workspace_() const { return *integrals_sp_; }

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
    RealMatrix S_;                ///< Overlap matrix (dimensionless).
    RealMatrix Hcore_;            ///< Core Hamiltonian [Hartree].
    RealMatrix X_;                ///< Orthogonalization transform.
    RealMatrix F_alpha_;          ///< α Fock matrix [Hartree].
    RealMatrix F_beta_;           ///< β Fock matrix [Hartree].
    RealMatrix J_;                ///< Coulomb matrix [Hartree].
    RealMatrix err_alpha_;        ///< α DIIS error channel.
    RealMatrix err_beta_;         ///< β DIIS error channel.
    RealMatrix C_alpha_;          ///< α MO coefficients.
    RealMatrix C_beta_;           ///< β MO coefficients.
    RealVector eig_alpha_;        ///< α orbital energies [Hartree].
    RealVector eig_beta_;         ///< β orbital energies [Hartree].
    RealMatrix P_alpha_;          ///< α density (old-in / new-out).
    RealMatrix P_beta_;           ///< β density (old-in / new-out).
    // ROHF-only channels (untouched in the other spin modes): the couple
    // stage preserves the physical spin Focks here (the density stage's
    // energy functional reads them, never F_eff) and publishes P_α+P_β for
    // the single-channel DIIS commutator.
    RealMatrix F_alpha_uncoupled_; ///< ROHF: true α Fock [Hartree].
    RealMatrix F_beta_uncoupled_;  ///< ROHF: true β Fock [Hartree].
    RealMatrix P_total_;           ///< ROHF: total density P_α + P_β.
    double e_total_ = 0;          ///< Total SCF energy [Hartree].
    double e_one_ = 0;            ///< One-electron energy [Hartree].
    double e_coulomb_ = 0;        ///< Coulomb (J) energy [Hartree].
    double e_xc_grid_ = 0;        ///< Grid XC energy [Hartree].
    double e_exact_exchange_ = 0; ///< Hybrid exact-exchange energy [Hartree].
    double e_solvent_ = 0;        ///< C-PCM ΔG_el [Hartree] (0 when off).
    double e_pcm_fock_half_ = 0;  ///< ½ Tr[V_pcm P] [Hartree] for energy peel.
    double int_n_alpha_ = 0;      ///< Grid-integrated α electron count.
    double int_n_beta_ = 0;       ///< Grid-integrated β electron count.
    double v_nuc_ = 0;            ///< Nuclear-repulsion energy [Hartree].
    int n_basis_ = 0;             ///< Basis dimension.
    int n_dropped_ = 0;           ///< Overlap eigen-directions dropped.
    int iter_ = 0;                ///< Current SCF iteration index.
    int diis_pairs_ = 0;          ///< DIIS history-size mirror.
    bool diis_active_ = false;    ///< DIIS extrapolating (≥ 2 vectors).
    bool restricted_ = false;     ///< spin_ == Restricted (per-run cache).
    bool is_hf_ = false;          ///< method_ == HF (per-run cache).

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
        bool apply_aspc = true;    ///< ASPC front/back; false on recursive call.
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

    /// @brief THE funnel: prologue (per-run assembly) → CPUDriver loop →
    ///        epilogue (de-shift / spin / result fill / energy / forces).
    ///        ASPC predict/correct/record is gated only on `use_predictor_`
    ///        (and `opts.apply_aspc` to prevent recursive re-entry).
    /// @param frame Geometry; mutated only on the mutating entries.
    /// @param guess Optional density guess.
    /// @param opts  Which shared behaviors apply on this entry.
    void drive_scf_(Frame& frame, const SCFInitialGuess* guess,
                    const DriveOpts& opts);

    /// @brief Core SCF (no ASPC wrapper) — used by drive_scf_ after ASPC
    ///        front-end resolves the density guess.
    void drive_scf_core_(Frame& frame, const SCFInitialGuess* guess,
                         const DriveOpts& opts);

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

    // Shared workspace (shells published via result_handle aliasing). The
    // determinant slot is stable (handle-before-run contract): it starts as
    // the empty determinant (0 electrons, 0 orbitals) and each run assigns
    // the fresh wavefunction into the same pointee.
    std::shared_ptr<IntegralPack> integrals_sp_{
        std::make_shared<IntegralPack>()};
    std::shared_ptr<SlaterDeterminant> det_sp_{
        std::make_shared<SlaterDeterminant>(RealMatrix{}, RealVector{},
                                            RealVector{}, 0)};
    std::shared_ptr<bool> has_result_sp_{std::make_shared<bool>(false)};
    int iterations_ = 0; ///< SCF iterations executed by the most recent run.

    /// C-PCM private workspace (null when solvent_eps == 0). Built in
    /// drive_scf_core_ before the loop; FockStage borrows the raw pointer.
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

    // ASPC (density extrapolation across outer MD steps)
    bool use_predictor_ = false;
    int corrector_steps_ = 2;
    double corrector_tol_ = 1e-4;
    bool fallback_full_scf_ = true;
    std::unique_ptr<class ASPCPredictor> aspc_alpha_;
    std::unique_ptr<class ASPCPredictor> aspc_beta_;
    bool has_last_density_ = false;
    xt::xtensor<double, 2> last_P_alpha_;
    xt::xtensor<double, 2> last_P_beta_;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
