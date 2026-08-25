/// @file cpu/semiempirical/dftb2.h
/// @brief SCC-DFTB2 multi-element method (static in-tree SK tables, CPU).
#pragma once

#include <array>
#include <string>
#include <vector>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/td_dftb2.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief Which static SK parameter set DFTB2 uses.
///
/// Parameters are a public member on the method (no `*Config` bag).
enum class DftbSkSet {
    /// Mio path: H–H uses transcribed mio-1-1 grids; other H/C/N/O pairs use
    /// exponential sketches (+ spin_W).
    MioSketch = 0,
    /// DFTB.org 3ob-3-1 organic CHNO radial tables (static header).
    ThreeOb = 1,
};

/// @brief SCC-DFTB2 energy (+ optional D3 dispersion, nuclear forces, cell
///        stress) for H/C/N/O.
///
/// Self-consistent-charge DFTB2 (Elstner et al., Phys. Rev. B **58**, 7260
/// (1998), DOI 10.1103/PhysRevB.58.7260) with optional **collinear** spin
/// polarization (Köhler / DFTB+ spin channel): dual α/β Fock matrices,
/// Mulliken Δq and spin population m = Nα−Nβ.
///
/// - `sk_set = MioSketch` (default): H–H from static mio-1-1
///   (`basis_data/dftb_mio_hh_sk.h`, DFTB+ pin
///   `regressions/semiempirical/dftb2-dftbplus-energy-pin/`); other pairs from
///   exponential sketch (`dftb_mio_tables.h` + spin_W). **Γ-point PBC:**
///   lattice-summed H⁰/S, screened γ, and repulsive within static SK cutoffs
///   (molecular limit when L ≫ r_cut). **k-mesh beyond Γ:** set
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
    /// @brief SCC residual tolerance on max |Δq| (and |Δm| when open shell).
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

    DFTB2() = default;

    std::string name() const override { return "dftb2"; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    // ── QmEngine (esmm-qmmm-02) — one-line Component forwards ─────────
    std::string qm_engine_name() const override { return name(); }
    void prepare(const BuildContext& qm_ctx) override { build(qm_ctx); }
    void evaluate(const LaunchContext& qm_ctx) override { launch(qm_ctx); }
    // as_embedding() default nullptr

    /// @brief Last total energy E_elec+rep+disp [eV].
    double last_energy() const { return energy_; }

    /// @brief Last D3(BJ) dispersion energy [eV] (0 when `d3` is false).
    double last_e_disp() const { return e_disp_; }

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
    /// magnetization kernel, atom-averaged spin constant \(W_A\) [eV]); the
    /// Hermitian eigenproblem \(\Omega F=\omega^2 F\) with
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
    /// excited-state gradients. **Parameter caveat:** only the H onsite /
    /// Hubbard \(U\) and the H–H Slater–Koster + repulsive grids are
    /// transcribed mio-1-1; every other `MioSketch` pair (and every spin
    /// constant \(W_A\), including H's) is an order-of-magnitude sketch, so
    /// absolute excitation energies are not DFTB+/mio-1-1 numbers for anything
    /// but the structural limits gated in
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
    ///         (`multiplicity != 1` or an odd electron count), when there is
    ///         no occupied or no virtual orbital, when the pair space exceeds
    ///         the dense cap of 512, when an occupied→virtual gap is
    ///         non-positive, or when a Casida root ω² is negative (an
    ///         imaginary excitation — instability). No fallback values.
    /// @throws std::invalid_argument when `n_states`, either window, or
    ///         `coupling_scale` is out of range.
    TdDftbResult td_casida(const Frame& frame, int n_states,
                           TdDftbChannel channel = TdDftbChannel::singlet,
                           int n_occ_window = 0, int n_virt_window = 0,
                           double coupling_scale = 1.0) const;

  private:
    double energy_ = 0;
    double e_disp_ = 0;
    double e_u_ = 0;
    double scc_residual_ = 0;
    int scc_iterations_ = 0;
    std::array<double, 6> last_stress_{};
    std::vector<double> last_dq_;
    std::vector<double> last_spin_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
