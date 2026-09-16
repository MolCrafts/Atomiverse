/// @file cpu/semiempirical/dftb2.h
/// @brief SCC-DFTB2 multi-element method (static in-tree SK tables, CPU).
#pragma once

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/electronic/aspc_predictor.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_spin_constants.h"
#include "atomiverse/cpu/semiempirical/td_dftb2.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief Which static SK parameter set DFTB2 / DFTB3 uses.
///
/// Parameters are a public member on the method (no `*Config` bag). Hubbard
/// derivatives \(U^{\mathrm d}\) and collinear spin constants \(W\) are
/// **not** part of the SK set — they are explicit user input
/// (`DFTB3::hubbard_du_ha`, `DFTB2::spin_w_ha`), both
/// [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
/// Renaming `MioSketch` is out of scope.
enum class DftbSkSet {
    /// @brief Sentinel: the method has not been told which SK set to use.
    ///
    /// DFTB3 defaults here and will not pick a set — `DFTB3::launch` throws
    /// `std::invalid_argument`. DFTB2 defaults to `MioSketch` and throws
    /// `std::invalid_argument` from `DFTB2::launch` if the caller assigns
    /// this value. Assignment itself does not throw.
    Unset = -1,
    /// DFTB.org **mio-1-1** H/C/N/O: onsite energies, Hubbard U, the full
    /// two-centre SK grids and the spline repulsive, all transcribed from the
    /// released `.skf` by `tools/gen_dftb_mio_tables.py`
    /// (`basis_data/dftb_mio_sk_tables.h`).
    ///
    /// The name is historical: this path carried exponential sketches before
    /// the SK transcription landed, and every pair has been real mio-1-1
    /// since.
    MioSketch = 0,
    /// DFTB.org 3ob-3-1 organic CHNO radial tables (static header).
    ThreeOb = 1,
};

/// @brief Boundary treatment of the DFTB2 Hamiltonian and its SCC kernel.
///
/// This is the same fork DFTB+ makes through the geometry block: a `C`
/// (cluster) geometry is an isolated molecule, an `S`/`F` geometry is a
/// supercell. They are **different Hamiltonians**, not an approximation of one
/// another — under `lattice` the second-order term carries the Ewald-summed
/// \(1/R\) interaction of every periodic image, which does not vanish for a
/// neutral cell because the charge distribution still has a dipole.
enum class DftbBoundary {
    /// Isolated molecule: no lattice images anywhere in \(H^0\), \(S\),
    /// \(\gamma\) or the repulsive. DFTB+ `C` geometry.
    cluster = 0,
    /// 3-D lattice: image-summed \(H^0\)/\(S\)/repulsive inside the SK
    /// cutoff and an **Ewald-summed** \(\gamma\). DFTB+ `S`/`F` geometry.
    lattice = 1,
};

/// @brief SCC-DFTB2 energy (+ optional D3 dispersion, nuclear forces, cell
///        stress) for H/C/N/O.
///
/// Self-consistent-charge DFTB2 (Elstner et al., Phys. Rev. B **58**, 7260
/// (1998), DOI 10.1103/PhysRevB.58.7260) with optional **collinear** spin
/// polarization (Köhler / DFTB+ spin channel): dual α/β Fock matrices,
/// Mulliken Δq and spin population m = Nα−Nβ.
///
/// - `sk_set = MioSketch` (default): full mio-1-1 H/C/N/O SK + repulsive
///   (`basis_data/dftb_mio_sk_tables.h`; DFTB+ pins
///   `regressions/semiempirical/{dftb2-dftbplus-energy-pin,h2o_dftb2_mio}/`).
///   **Γ-point PBC** (`boundary = lattice`): lattice-summed H⁰/S and
///   repulsive within the SK cutoffs, and an **Ewald-summed** γ — the 1/R
///   half of the SCC kernel is only conditionally convergent, so truncating it
///   at a sphere is not an approximation but a different (non-positive-definite)
///   operator. Set `boundary = cluster` for an isolated molecule; a large box
///   is **not** a substitute, because Ewald has no cutoff to hide behind.
///   **k-mesh beyond Γ:** set
///   `k_mesh = {n1,n2,n3}` (Monkhorst–Pack, CP2K/DFTB+ convention) for
///   Bloch-phased complex \(H^0(\mathbf{k})\), \(S(\mathbf{k})\), and
///   k-weighted SCC energy (`reg_dftb2-kpoints`). Default `{1,1,1}` keeps the
///   real Γ solver (pin-safe). Multi-k also carries **analytic nuclear forces
///   and cell stress** (restricted MioSketch): the Bloch-sum gradient contracts
///   the image-resolved densities
///   \(\tilde P(\mathbf T)=\sum_k w_k\,\mathrm{Re}[P(k)e^{-2\pi i\kappa\cdot n}]\)
///   (and the energy-weighted \(\tilde W\)) with the same lattice-image walk
///   the energy used, so \(F=-\nabla E\) of the sampled functional and
///   \(\sigma\) is its pair virial (`reg_dftb2-multi-k-forces`). Collinear
///   open-shell multi-k ships (`scc_energy_unrestricted_kmesh`,
///   `dftb2-multi-k-open-shell`). **ThreeOb still throws** on `k_mesh≠{1,1,1}`
///   (no lattice sum / no Bloch H0(k)/S(k) — `dftb2-threeob-periodic` then
///   `dftb2-threeob-multi-k`).
///   **Analytic** nuclear forces for closed shell **and** collinear open shell
///   (HF + Pulay + SCC γ + repulsive; spin shifts on ∂S) use the **same
///   lattice-image walk** as the energy — not min-image only — so
///   \(F=-\nabla E\) under Γ-PBC (`reg_dftb2-periodic-forces`,
///   `reg_dftb2-analytic-forces`, `reg_dftb2-spin-analytic-forces`). Both spin
///   cases also offer an **analytic** intensive cell stress on `kStress`
///   [eV/Å³] — the pair virial of those same image bond terms, spin-resolved
///   over the α/β channels (`reg_dftb2-analytic-stress`,
///   `reg_dftb2-spin-analytic-stress`). No strain-FD is left on the MioSketch
///   product path.
/// - `sk_set = ThreeOb`: `basis_data/dftb_3ob_tables.h` 3ob-3-1 radial H/S +
///   repulsive (Gaus et al., JCTC 2013; organic CHNO). Closed shell: **analytic**
///   nuclear forces (same HF + Pulay + γ + repulsive assembly as MioSketch, on
///   the Ha/Bohr 3ob tables; `reg_dftb2-analytic-forces-threeob`) **and** an
///   analytic intensive cell stress on `kStress` [eV/Å³] — the pair virial of
///   those same bond terms, same sign and Voigt order as the MioSketch path
///   (`reg_dftb2-analytic-stress-threeob`). The 3ob path is a **molecular
///   cluster** sum (no lattice images), so its virial is the isolated-molecule
///   virial and the cell enters only through Ω. Spin / nonzero charge are not
///   supported on the ThreeOb path.
///
/// Optional **Grimme DFT-D3(BJ)** (`d3 = true`) is a geometry-only additive
/// correction using in-tree Grimme tables and the published DFTB3-D3(BJ)
/// damping set (s6, s8, a1, a2) as used by DFTB+ / simple-dftd3 `--bj dftb3`.
/// It is added on top of **either** `sk_set` path, with its own analytic
/// gradient and (when `with_stress`) pair-virial / strain-FD stress, so
/// \(F = -\nabla E\) and \(\sigma = -(1/\Omega)\partial E/\partial\varepsilon\)
/// hold for the total. Three-body ATM is off.
/// Default (`d3_periodic = false`): isolated-cluster D3 (no lattice images) —
/// correct for molecules / large boxes and the water-dimer pin. Set
/// `d3_periodic = true` for an orthorhombic lattice-image D3 sum.
///
/// Open shell uses `charge` and `multiplicity` to set nα, nβ. Units: energy
/// [eV], positions [Å], forces [eV/Å], stress [eV/Å³] Voigt
/// `(xx,yy,zz,xy,xz,yz)` with mechanical convention
/// \(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\).
///
/// Molecule, open shell, dispersion, forces + stress:
/// @code
///   cpu::semiempirical::DFTB2 dftb;
///   dftb.multiplicity = 3;   // O2 triplet (MioSketch)
///   dftb.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;
///   dftb.d3 = true;
///   dftb.with_forces = true;
///   dftb.with_stress = true;
///   eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
/// @endcode
///
/// Periodic cell sampled beyond Γ — energy + analytic forces + analytic stress
/// (collinear open shell ships on MioSketch; `ThreeOb` throws):
/// @code
///   cpu::semiempirical::DFTB2 dftb;
///   dftb.k_mesh = {4, 1, 1};
///   dftb.with_forces = true;
///   dftb.with_stress = true;
///   eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
/// @endcode
class DFTB2 : public cpu::Semiempirical, public cpu::QmEngine {
  public:
    /// @brief SCC residual tolerance on max |Δq| (and |Δm| when open shell) [e].
    double scc_tol = 1e-8;
    /// @brief Maximum SCC iterations (dimensionless). Exhausting it throws.
    int max_scc = 50;
    /// @brief Anderson charge-mixing relaxation β ∈ (0,1] (dimensionless).
    ///
    /// Undamped Picard (no mixing) is a period-2 limit cycle for polar
    /// systems: it never meets `scc_tol` and the reported energy is picked by
    /// the parity of `max_scc`. Charges (and the spin population m, mixed as
    /// one vector with them) go through `internal::ChargeMixer`. The default
    /// converges every in-tree case in ≤ 14 iterations; β ≳ 0.7 stops
    /// converging for hydrogen-bonded systems.
    double mix_beta = 0.3;
    /// @brief When true, write F=−∇E [eV/Å] on launch.
    bool with_forces = true;
    /// @brief When true, write intensive cell stress to `kStress` [eV/Å³].
    ///
    /// Closed-form pair virial of the analytic force terms on **every**
    /// product path, σ = −(1/Ω)∂E/∂ε, Voigt `(xx,yy,zz,xy,xz,yz)`. No
    /// strain-FD is left anywhere. MioSketch Γ is spin-resolved — the α/β
    /// Hellmann–Feynman + Pulay ∂S (including the ±W_A m_A spin shift), the
    /// SCC γ virial on the total Mulliken Δq, and the repulsive pair virial;
    /// ThreeOb is the same assembly on the 3ob tables; multi-k restricted
    /// MioSketch has its own closed-form Bloch pair virial.
    ///
    /// Every closed-form virial walks the **self-image** A–(A+T) bonds as well
    /// as A–B: those carry no nuclear force (the bond is T, independent of
    /// every nuclear coordinate) but do scale with the cell, so dropping them
    /// is a silent error for any cell whose own images sit inside `r_cut`.
    ///
    /// When `d3=true`, D3 stress is folded in on top from the analytic pair
    /// virial of `dftd3_pbe_bj` (image-aware when `d3_periodic=true`).
    bool with_stress = false;
    /// @brief Static SK parameter set (member — not a Config bag).
    DftbSkSet sk_set = DftbSkSet::MioSketch;
    /// @brief Cluster or lattice Hamiltonian. Default `lattice`.
    ///
    /// A Frame always carries a `Box`, so the box alone cannot say whether the
    /// caller means a crystal or a molecule in a large empty cell; the method
    /// has to be told, exactly as DFTB+ is told by the geometry type. The
    /// default is `lattice` because a declared cell is a declared lattice —
    /// an isolated molecule must **say so**, rather than relying on a cutoff
    /// to hide its own images.
    ///
    /// `k_mesh` beyond Γ requires `lattice` and throws under `cluster`.
    DftbBoundary boundary = DftbBoundary::lattice;
    /// @brief Ewald real-space cutoff for the periodic \(\gamma\) [Å].
    ///
    /// Also fixes the splitting parameter \(\alpha\) through the LAMMPS
    /// heuristic \(\alpha=(1.35-0.15\ln\varepsilon)/r_c\). Energy is
    /// invariant to this choice; it trades real-space against reciprocal work.
    /// Ignored under `cluster`.
    double ewald_cutoff = 10.0;
    /// @brief Ewald relative accuracy in (0, 1); sets the \(|k|\) cutoff.
    double ewald_accuracy = 1e-8;
    /// @brief Net molecular charge [e] (subtracted from valence electron count).
    int charge = 0;
    /// @brief Spin multiplicity 2S+1 (dimensionless). Default 1 = closed shell.
    int multiplicity = 1;
    /// @brief Opt-in Grimme DFT-D3 with Becke–Johnson damping (two-body).
    bool d3 = false;
    /// @brief When true with `d3`, lattice-sum D3 over the orthorhombic box.
    ///        Default false preserves the molecular water-dimer pin.
    bool d3_periodic = false;
    /// @brief D3 real-space pair cutoff [Å]. Default 49.8485 Å = 94.2 Bohr,
    ///        the legacy `dftd3` `rthr` convention.
    ///
    /// Irrelevant for an isolated cluster (every pair is inside it), but it
    /// **defines** the value of a `d3_periodic` lattice sum: the tail beyond
    /// the cutoff is discarded, and different D3 engines truncate differently
    /// (simple-dftd3 defaults to 60 Bohr = 31.75 Å for the two-body term).
    /// Set it to the oracle's cutoff when pinning a periodic dispersion
    /// energy — see `regressions/semiempirical/dftb2-dispersion-d3-periodic/`.
    double d3_cutoff = 49.84849326866401;
    /// @brief D3 C6 scale (dimensionless). DFTB3-D3(BJ) default = 1.
    double d3_s6 = 1.0;
    /// @brief D3 C8 scale (dimensionless). DFTB3-D3(BJ) default = 0.5883.
    double d3_s8 = 0.5883;
    /// @brief BJ steepness (dimensionless). DFTB3-D3(BJ) default = 0.5719.
    double d3_a1 = 0.5719;
    /// @brief BJ offset [Bohr]. DFTB3-D3(BJ) default = 3.6017.
    double d3_a2 = 3.6017;
    /// @brief Monkhorst–Pack mesh subdivisions per reciprocal axis
    ///        (dimensionless). Default `{1,1,1}` = Γ only (real solver).
    ///
    /// CP2K/DFTB+ convention: \(\kappa_d = (2r - N_d - 1)/(2 N_d)\),
    /// \(r = 1..N_d\). Any \(N_d < 1\) throws. Multi-k MioSketch delivers
    /// energy, analytic nuclear forces, and analytic cell stress for closed
    /// shell **and** collinear open shell. `ThreeOb` throws.
    std::array<int, 3> k_mesh{{1, 1, 1}};
    /// @brief Fold \(\pm\mathbf{k}\) pairs (double weight). Default true.
    ///        Exact for real-Hamiltonian closed-shell densities.
    ///
    /// Energy and forces are invariant under this flag: the folded entry
    /// stands for its own conjugate partner (\(P(-k)=\overline{P(k)}\)).
    /// Setting it false costs the full mesh and additionally arms the
    /// imaginary-residue check inside the multi-k gradient.
    bool k_time_reversal = true;

    /// @brief Per-Z DFTB+U effective Hubbard \(U_{\mathrm{eff}}\) [eV].
    ///
    /// Index = atomic number \(Z\) (1…95). Zero means off for that element.
    /// After the SCC, a Dudarev equal-filling correction is added on the
    /// atomic Mulliken population \(n_A = n_{\mathrm{val},A}-\Delta q_A\):
    ///
    /// \[
    ///   E_U = \sum_A \tfrac12 U_A\, n_A\bigl(1 - n_A/N_A\bigr),\quad
    ///   N_A = 2\,n_{\mathrm{AO},A}
    /// \]
    ///
    /// (mean occupation of \(N_A\) spin-orbitals). MioSketch Γ only.
    /// Forces: central FD of \(E_{\mathrm{SCC}}+E_U\) (re-SCC). Stress and
    /// multi-k throw when any \(U_{\mathrm{eff}}\) is nonzero.
    /// See `.claude/notes/dftb-plus-u.md`.
    std::array<double, 96> u_eff{};

    /// @brief Per-element collinear spin constants \(W^A_{ll'}\)
    ///        [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
    ///
    /// Empty = unset. Required when the spin channel is taken
    /// (`multiplicity != 1` on `launch()`, or `TdDftbChannel::triplet` on
    /// `td_casida()`); otherwise ignored. The published PBE table is
    /// `basis_data::kDftbSpinConstantsPbeHa`.
    ///
    /// `launch()` / `td_casida()` convert Ha→eV with `units::kHartreeInEv`
    /// and collapse the shell matrix to the highest-occupied atomic orbital
    /// (HOAO) diagonal: H \(W_{ss}=w_{ha}[0][0]\), C/N/O
    /// \(W_{pp}=w_{ha}[1][1]\). The implemented spin energy [eV] is then
    ///
    /// \[
    ///   E_{\mathrm{spin}}=\tfrac12\sum_A W_A\, m_A^2
    /// \]
    ///
    /// with \(W_A\) that HOAO scalar [eV] and
    /// \(m_A=N_{\alpha,A}-N_{\beta,A}\) the Mulliken spin population [e].
    /// This is DFTB+ `ShellResolvedSpin = No`. Setting
    /// `shell_resolved_spin` true throws (`dftb-shell-resolved-spin`); the
    /// unimplemented shell-resolved path (per-shell Mulliken spin \(p_{Al}\)
    /// [e] contracted with the full \(W_{ll'}\) matrix) is not computed.
    std::vector<basis_data::DftbSpinConstant> spin_w_ha{};

    /// @brief When true, request DFTB+ `ShellResolvedSpin = Yes` (unimplemented).
    ///
    /// Default path collapses `spin_w_ha` [Ha] to the highest-occupied
    /// atomic orbital (HOAO) diagonal and converts Ha→eV, equivalent to
    /// DFTB+ `ShellResolvedSpin = No`. `true` throws `std::runtime_error`
    /// naming `dftb-shell-resolved-spin` when the spin channel is taken.
    bool shell_resolved_spin = false;

    DFTB2() = default;

    /// @brief Extrapolate the SCC starting charges from previous launches.
    ///
    /// Along a trajectory the converged Mulliken \f$\Delta q\f$ is a smooth
    /// function of time, so the previous steps predict the next one. Kolafa's
    /// always-stable predictor–corrector (`electronic::ASPCPredictor`, the
    /// same one Gaussian SCF and GPW use on the density matrix) is applied to
    /// the \f$[\Delta q, m]\f$ vector, which is the layout `ChargeMixer`
    /// already mixes.
    ///
    /// This changes **where the SCC starts, never where it converges**: the
    /// residual gate is unchanged, so the energy is the same to `scc_tol` and
    /// only the iteration count moves. History resets automatically when the
    /// atom count changes; call `reset_predictor()` after any other geometry
    /// discontinuity.
    ///
    /// Off by default — a single point has no history to extrapolate from.
    ///
    /// @param order Lagrange extrapolation order K (dimensionless, ≥ 0).
    ///              Prediction needs K+1 recorded steps.
    /// @return `*this`, so the call chains into a component registration.
    /// @throws std::invalid_argument when `order < 0`.
    DFTB2& predictor(int order = 3) & {
        dq_predictor_.emplace(order);
        predictor_n_at_ = 0;
        return *this;
    }

    /// @brief Rvalue overload of @ref predictor, for
    ///        `on(phase).add(DFTB2{}.predictor(3))`.
    /// @param order Lagrange extrapolation order K (dimensionless, ≥ 0).
    DFTB2&& predictor(int order = 3) && {
        dq_predictor_.emplace(order);
        predictor_n_at_ = 0;
        return std::move(*this);
    }

    /// @brief Drop the extrapolation history (geometry discontinuity/restart).
    ///        No-op when no predictor is armed.
    void reset_predictor() {
        if (dq_predictor_)
            dq_predictor_->reset();
        predictor_n_at_ = 0;
    }

    /// @brief Component name token `"dftb2"` (dimensionless).
    std::string name() const override { return "dftb2"; }

    /// @brief No-op; DFTB2 allocates no per-run workspace in `build`.
    /// @param ctx Driver build context (unused).
    void build(const BuildContext& ctx) override;

    /// @brief One SCC-DFTB2 evaluation; writes `kEnergy` [eV] to `frame`.
    ///
    /// Optional: `with_forces` writes \(F=-\nabla E\) [eV/Å]; `with_stress`
    /// writes intensive `kStress` [eV/Å³].
    ///
    /// @param ctx Driver launch context (non-owning `frame` + services).
    /// @throws std::invalid_argument when `sk_set` is `Unset`, when
    ///         `multiplicity != 1` and `spin_w_ha` is empty, when the Frame
    ///         is empty, or when a numeric parameter is out of range.
    /// @throws std::runtime_error when `shell_resolved_spin` is true on an
    ///         open-shell run, or when a requested SK / k-mesh / boundary
    ///         combination is not implemented.
    void launch(const LaunchContext& ctx) override;

    /// @brief `QmEngine` name token; same as `name()` (`"dftb2"`).
    std::string qm_engine_name() const override { return name(); }
    /// @brief `QmEngine` prepare; forwards to `build`.
    /// @param qm_ctx Driver build context (unused by DFTB2).
    void prepare(const BuildContext& qm_ctx) override { build(qm_ctx); }
    /// @brief `QmEngine` evaluate; forwards to `launch`.
    /// @param qm_ctx Driver launch context (non-owning `frame` + services).
    void evaluate(const LaunchContext& qm_ctx) override { launch(qm_ctx); }
    // as_embedding() default nullptr

    /// @brief Last total energy E_elec+rep+disp [eV].
    double last_energy() const { return energy_; }

    /// @brief Last D3(BJ) dispersion energy [eV] (0 when `d3` is false).
    double last_e_disp() const { return e_disp_; }

    /// @brief Last band-structure energy \(\sum_i n_i \varepsilon_i\) [eV].
    ///
    /// The same quantity DFTB+ prints as `Band energy` in `detailed.out`,
    /// k-weighted on a mesh and summed over both spin channels when open
    /// shell. It is **not** \(\mathrm{Tr}(\rho H^0)\): the SCC shift is in
    /// \(H\), so this carries \(\sum_A V_A q_A\) on the gross Mulliken
    /// population. Exposed so an oracle residual can be bisected term by term
    /// rather than only on the total.
    double last_e_band() const { return e_band_; }

    /// @brief Last second-order SCC energy \(\tfrac12\sum_{AB}\gamma_{AB}
    ///        \Delta q_A\Delta q_B\) [eV] — DFTB+ `Energy SCC`.
    double last_e_scc() const { return e_scc_; }

    /// @brief Last pair repulsive energy [eV] — DFTB+ `Repulsive energy`.
    ///
    /// `last_energy() - last_e_rep()` is DFTB+'s `Total Electronic energy`
    /// (before any D3 / +U add-on).
    double last_e_rep() const { return e_rep_; }

    /// @brief Last DFTB+U Dudarev correction [eV] (0 when all `u_eff` are 0).
    double last_e_u() const { return e_u_; }

    /// @brief SCC residual max|Δq_out − Δq_in| at exit [e]; below `scc_tol`
    ///        whenever `launch` returned (it throws otherwise).
    double last_scc_residual() const { return scc_residual_; }

    /// @brief SCC iterations the last solve took (dimensionless).
    int last_scc_iterations() const { return scc_iterations_; }

    /// @brief Last intensive Voigt stress [eV/Å³] (zero if stress not requested).
    const std::array<double, 6>& last_stress() const { return last_stress_; }

    /// @brief Per-atom Mulliken charge fluctuation Δq_A [e] after last launch.
    const std::vector<double>& last_mulliken_dq() const { return last_dq_; }

    /// @brief Per-atom Mulliken spin population m_A = Nα−Nβ [e] after last launch.
    const std::vector<double>& last_mulliken_spin() const { return last_spin_; }

    /// @brief Vertical linear-response TD-DFTB (Casida) excitation energies at
    ///        the geometry in `frame`.
    ///
    /// Tight-binding time-dependent DFT response (Niehaus, Suhai, Della Sala,
    /// Lugli, Elstner, Seifert, Frauenheim, *Phys. Rev. B* **63**, 085108
    /// (2001), DOI 10.1103/PhysRevB.63.085108). Mulliken transition charges
    /// \(q^{ia}_A=\tfrac12\sum_{\mu\in A}(c_{\mu i}(Sc_a)_\mu
    /// + c_{\mu a}(Sc_i)_\mu)\) [e] contract the coupling kernel
    /// \(K^{S}_{ia,jb}=\sum_{AB}q^{ia}_A\gamma_{AB}q^{jb}_B\) (singlet, the
    /// same \(\gamma_{AB}\) [eV] as the SCC energy) or
    /// \(K^{T}_{ia,jb}=\sum_{A}q^{ia}_A W_A q^{jb}_A\) (triplet, on-site
    /// magnetization kernel). \(W_A\) [eV] is the highest-occupied atomic
    /// orbital (HOAO) diagonal of `spin_w_ha` (H \(W_{ss}\), C/N/O
    /// \(W_{pp}\)) converted Ha→eV with `units::kHartreeInEv`; `spin_w_ha`
    /// itself is
    /// [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
    /// The Hermitian eigenproblem \(\Omega F=\omega^2 F\) with
    /// \(\Omega_{ia,jb}=\delta_{ij}\delta_{ab}(\varepsilon_a-\varepsilon_i)^2
    /// +2\sqrt{\varepsilon_a-\varepsilon_i}\,K_{ia,jb}\,
    /// \sqrt{\varepsilon_b-\varepsilon_j}\) is solved densely.
    ///
    /// **Stateless / Frame-in.** This runs its own converged restricted SCC at
    /// `frame`'s geometry through the same routine `launch()` uses; it does
    /// **not** read a cached ground state, so it is independent of whether (or
    /// where) `launch()` last ran. Cost is one extra SCC solve.
    ///
    /// **Energies + oscillator strengths** (length form, singlet). No
    /// excited-state gradients. The triplet kernel needs `spin_w_ha` (HOAO
    /// collapse of the published PBE table); empty `spin_w_ha` throws.
    /// Absolute excitation energies are not DFTB+/mio-1-1 numbers for
    /// anything but the structural limits gated in
    /// `regressions/semiempirical/td-dftb-h2/`.
    ///
    /// @param frame          Geometry [Å] + atomic number Z in the type column.
    ///                       Read-only; nothing is written to it.
    /// @param n_states       Number of lowest excitations to return
    ///                       (dimensionless), 1 ≤ n_states ≤ n_pair.
    /// @param channel        Singlet or triplet spin channel.
    /// @param n_occ_window   Highest-lying occupied orbitals to include
    ///                       (dimensionless); 0 = all occupied.
    /// @param n_virt_window  Lowest-lying virtual orbitals to include
    ///                       (dimensionless); 0 = all virtual.
    /// @param coupling_scale Scale λ on the coupling kernel K (dimensionless).
    ///                       1 = physical response; 0 = the non-interacting
    ///                       limit, where ω reduces to \(\varepsilon_a
    ///                       -\varepsilon_i\) exactly.
    /// @return `TdDftbResult` — excitation energies [eV], the matching
    ///         single-particle gaps [eV], Mulliken transition charges [e], and
    ///         the assembled Casida matrix Ω [eV²].
    /// @throws std::runtime_error when `sk_set != MioSketch`, when a k-mesh
    ///         beyond Γ is set, when the reference is not closed shell
    ///         (`multiplicity != 1` or an odd electron count), when
    ///         `shell_resolved_spin` is true on the triplet channel, when
    ///         there is no occupied or no virtual orbital, when the pair
    ///         space exceeds the dense cap of 512, when an occupied→virtual
    ///         gap is non-positive, or when a Casida root ω² is negative (an
    ///         imaginary excitation — instability). No fallback values.
    /// @throws std::invalid_argument when `n_states`, either window, or
    ///         `coupling_scale` is out of range, or when `channel` is
    ///         `triplet` and `spin_w_ha` is empty.
    TdDftbResult td_casida(const Frame& frame, int n_states,
                           TdDftbChannel channel = TdDftbChannel::singlet,
                           int n_occ_window = 0, int n_virt_window = 0,
                           double coupling_scale = 1.0) const;

  private:
    double energy_ = 0;
    double e_band_ = 0;
    double e_scc_ = 0;
    double e_rep_ = 0;
    double e_disp_ = 0;
    double e_u_ = 0;
    std::optional<electronic::ASPCPredictor> dq_predictor_;
    int predictor_n_at_ = 0;
    double scc_residual_ = 0;
    int scc_iterations_ = 0;
    std::array<double, 6> last_stress_{};
    std::vector<double> last_dq_;
    std::vector<double> last_spin_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
