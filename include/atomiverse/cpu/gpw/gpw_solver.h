/// @file cpu/gpw/gpw_solver.h
/// @brief Self-contained GPW (Gaussian and Plane Waves) SCF solver.
///
/// Structurally parallel to `cpu::scf::SCFSolver` and `cpu::pw::PW`: a
/// self-contained inner solver for one electronic-structure method. The Kohn-Sham
/// problem is solved in the contracted Gaussian AO basis (small dense generalized
/// eigenproblem H C = S C ε), while the Hartree + XC + long-range local
/// pseudopotential are evaluated on a uniform real-space grid (collocate / FFT
/// Poisson / integrate). It reuses only shared *physics/algebra* from public
/// headers — `cpu::linalg` (generalized eigensolve, density build) and
/// `cpu::xc::pade` (functional) — never another domain's `internal`.
///
/// Scope: closed-shell (RKS, `n_occ`) at Γ **or** on a Monkhorst-Pack k mesh
/// (`k_mesh` / explicit `k_points`; complex Bloch matrices per k, k-summed
/// density), optional **metallic occupation smearing** (Fermi–Dirac /
/// Methfessel–Paxton; multi-k RKS DIIS **and** OT, including OT+DFT+U+smear;
/// Mermin free energy), and unrestricted (UKS, `n_alpha`/`n_beta` with PADE
/// LSD or PBE/PBEsol GGA) at Γ **or** on the same k mesh (collinear dual
/// H_k, spin-resolved density/XC; multi-k UKS + FD/MP smear productized
/// (DIIS **and** OT); multi-k UKS OT+DFT+U product for integer occ **and**
/// for metallic smear — joint free-energy OT+U+smear). Periodic neutral
/// electrostatics via the GTH Gaussian core charge. Analytic forces cover
/// Γ RKS/UKS and multi-k RKS/UKS (`gpw_forces_analytic`); analytic cell
/// stress covers Γ RKS/UKS, multi-k RKS/UKS, and Dudarev DFT+U on Γ and
/// multi-k (`gpw_stress` via fixed-\(P_k^\sigma\) + BZ-averaged \(E_U\)).
#pragma once

#include <array>
#include <complex>
#include <string>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h" // GaussianBasis
#include "atomiverse/cpu/gpw/gth_potential.h" // GthPotential, GthNonlocalChannel
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h" // kD3Cp2kPairCutoffBohr
#include "atomiverse/cpu/gpw/g0w0.h"      // G0W0Result
#include "atomiverse/cpu/gpw/mp2.h"       // Mp2Result
#include "atomiverse/cpu/gpw/rpa.h"       // RpaResult
#include "atomiverse/cpu/linalg/matrix.h" // RealMatrix
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Dudarev DFT+U (CP2K PLUS_U) on one element / angular-momentum shell.
///
/// Parameters are constructor args / members — no `PlusUConfig` bag.
/// @param element Correlated element (e.g. Ni).
/// @param l       Angular momentum of the correlated shell (d → 2).
/// @param u_minus_j_ev Effective U−J [eV] (public surface; converted to Ha inside).
struct PlusU {
    molrs::Element element = molrs::Element::H;
    int l = 2;
    double u_minus_j_ev = 0.0;

    PlusU() = default;
    PlusU(molrs::Element element, int l, double u_minus_j_ev)
        : element(element), l(l), u_minus_j_ev(u_minus_j_ev) {}
};

/// @brief One atom's GPW data: valence charge, basis, and GTH pseudopotential.
struct GpwAtom {
    double z_val = 0.0;        ///< valence charge (electrons it contributes).
    int z = 0;                 ///< atomic number Z (for DFT-D3; 0 = unset).
    double pos[3] = {0, 0, 0}; ///< position [Bohr].
    std::vector<GridBasisFunction>
        basis; ///< contracted GTOs, already centered at pos.
    /// @brief Angular momentum ℓ of each entry in `basis` (same length; for DFT+U).
    std::vector<int> basis_l;
    double rloc = 0.0;        ///< GTH local radius.
    std::vector<double> cloc; ///< GTH local short-range C coefficients.
    std::vector<GthNonlocalChannel>
        nonlocal; ///< GTH nonlocal projector channels.

    GpwAtom() = default;

    /// @brief Construct from a contracted basis + GTH potential.
    ///
    /// Converts each contracted shell of `basis` into Cartesian
    /// GridBasisFunctions centered at `pos`, folding the per-primitive Gaussian
    /// normalization N(alpha,L) = (2 alpha/pi)^{3/4} (4 alpha)^{L/2} into the
    /// coefficient (the L alpha-power is what fixes the spanned space to match
    /// CP2K; the constant factor is irrelevant to the energy). Angular
    /// functions are the (2L+1) pure-spherical real solid harmonics from
    /// real_solid_harmonics(L), for any L.
    /// @param pos   Atom position [Bohr]; z_val is taken from `pot.zion`.
    /// @param basis Contracted Gaussian basis to center at `pos`.
    /// @param pot   GTH pseudopotential (local + nonlocal channels).
    /// @param z     Atomic number Z (needed only for DFT-D3, GpwDft.d3).
    GpwAtom(const double pos[3], const GaussianBasis& basis,
            const GthPotential& pot, int z = 0);
};

/// @brief Exchange-correlation functional choice.
///
/// `PADE` — LDA Padé; `PBE` / `PBEsol` / `RevPBE` / `RPBE` / `BLYP` /
/// `OptB88VdW` — pure GGA (OptB88VdW = optB88-x + PW92-c; pair with `vdw_df`)
/// (shared `evaluate_xc`). RKS and Γ-UKS support all six (spin-GGA via
/// `xc_grid_*_spin`; closed-shell via `xc_grid_*`).
enum class XcKind { PADE, PBE, PBEsol, RevPBE, RPBE, BLYP, OptB88VdW };

/// @brief Metallic occupation smearing kind (product path: multi-k RKS/UKS).
///
/// Mirrors `cpu::pw::Smearing` for API parity with QE/CP2K. Default `none`
/// keeps fixed integer fillings (`n_occ` double-occupied, or `n_alpha`/
/// `n_beta` for UKS). When not `none`, the SCF reports the Mermin free
/// energy \(F=E_{\mathrm{KS}}-TS\) as `e_total` and fills bands via a
/// shared chemical potential \(\mu\) (RKS: band form \(f\in[0,2]\); UKS:
/// spin-orbital form \(f_\sigma\in[0,1]\) over \(\alpha+\beta\)).
enum class Smearing : int {
    none = 0,              ///< Fixed integer occupations.
    fermi_dirac = 1,       ///< Fermi–Dirac (CP2K `METHOD FERMI_DIRAC`).
    methfessel_paxton = 2, ///< Methfessel–Paxton of order `mp_order`.
};

/// @brief GPW SCF result with an energy decomposition.
struct GpwResult {
    double e_total = 0.0;    ///< total energy [Hartree] (free energy \(F\) when smearing).
    double e_kinetic = 0.0;  ///< Tr(P T)  (RKS: closed P; UKS: Tr((Pα+Pβ)T)).
    double e_loc_sr = 0.0;   ///< Tr(P V_loc^sr).
    double e_nonlocal = 0.0; ///< Tr(P V_nl).
    double e_es = 0.0; ///< total electrostatic (Hartree + e-ion-lr + ion-ion).
    double e_xc = 0.0; ///< XC energy.
    double e_disp = 0.0; ///< DFT-D3/D4/TS/MBD/XDM/vdW-DF/DF2/rVV10 dispersion [Hartree] (0 when off).
    /// @brief Fermi level \(\mu\) [Hartree] when metallic smearing is on (0 if fixed occ).
    double e_fermi = 0.0;
    /// @brief Entropy contribution \(-\sigma S\) [Hartree] folded into free energy (0 if fixed).
    double e_entropy = 0.0;
    /// @brief \(\sum_{k,b} w_k f_{kb}\) electron count after occupation fill.
    double n_elec_occ = 0.0;
    /// @brief Hirshfeld volume ratios \(V/V_{\mathrm{free}}\) when
    ///        `GpwDft::hirshfeld` was set (empty for free-atom TS/MBD).
    std::vector<double> volume_ratios;
    double e_dftu =
        0.0; ///< Dudarev DFT+U energy [Hartree] (0 when plus_u empty).
    /// @name k-point sampling payload (populated by multi-k RKS or multi-k UKS;
    /// empty for pure Γ runs). One entry per sampled k.
    /// @{
    std::vector<std::array<double, 3>> k_frac; ///< fractional k coordinates.
    std::vector<double> k_weights;             ///< quadrature weights (Σ = 1).
    /// @brief Band energies per k [Hartree], ascending; one vector per k
    ///        (length = linearly independent basis directions at that k).
    ///        Multi-k UKS: α channel; β bands live in `eps_k_beta`.
    std::vector<std::vector<double>> eps_k;
    /// @brief Multi-k UKS β band energies per k [Hartree] (empty for RKS / Γ).
    std::vector<std::vector<double>> eps_k_beta;
    /// @brief Complex Bloch MO coefficients per k (nb × m_k, columns =
    ///        orbitals, ascending energy). Empty for Γ runs. Multi-k RKS
    ///        and multi-k UKS α. Required for multi-k `GpwDft::mp2`
    ///        (closed-shell and unrestricted).
    std::vector<linalg::CplxMatrix> C_k;
    /// @brief Multi-k UKS β full Bloch MO coefficients per k (nb × m_k;
    ///        empty for RKS / Γ). Paired with `eps_k_beta`; virtual columns
    ///        feed multi-k UMP2.
    std::vector<linalg::CplxMatrix> C_k_beta;
    /// @brief Total sampled k-points after mesh / TR fold / explicit list.
    int n_kpts = 0;
    /// @brief k-points owned by this MPI rank under round-robin
    ///        (`ik = rank; ik < n_kpts; ik += nprocs`). Equals `n_kpts` when
    ///        MPI is inactive or `nprocs == 1`.
    int n_kpts_local = 0;
    /// @brief Occupied MO coefficients per k (k-path only): each entry is a
    ///        row-major complex matrix of shape `nb × n_occ` (RKS) or
    ///        `nb × n_alpha` (UKS α). Empty for pure Γ. Used by
    ///        `gpw_forces_analytic` / `gpw_stress` to rebuild \(P_k\) / \(W_k\).
    std::vector<std::vector<std::complex<double>>> C_occ_k;
    /// @brief Multi-k UKS β occupied MOs per k: row-major `nb × n_beta`
    ///        (empty for RKS / Γ). Paired with `eps_k_beta`.
    std::vector<std::vector<std::complex<double>>> C_occ_k_beta;
    /// @brief Multi-k complex densities \(P_k\) (nb × nb Hermitian) after a
    ///        converged k-path SCF — the XL-BOMD advance payload. RKS: total
    ///        closed-shell \(P_k\) (occ = 2). UKS: α-spin \(P_k^\alpha\)
    ///        (occ = 1). Empty for pure Γ runs.
    std::vector<linalg::CplxMatrix> P_k;
    /// @brief Multi-k UKS β-spin densities \(P_k^\beta\) (nb × nb Hermitian;
    ///        empty for RKS / Γ). Paired with `P_k` for dual-spin XL-BOMD.
    std::vector<linalg::CplxMatrix> P_k_beta;
    /// @}
    double s2 = 0.0; ///< ⟨S²⟩ (Löwdin; RKS → 0; UKS after convergence).
    /// @brief Mulliken spin populations m_A = Nα−Nβ [e] per atom (empty if RKS).
    std::vector<double> mulliken_spin;
    bool converged = false;
    int iters = 0;

    /// @name Converged wavefunction (populated on convergence; empty otherwise).
    /// RKS: `P == build_density_matrix(C, n_occ, ·, 2)` from the closed H.
    /// UKS: `C`/`P`/`eps` are α; `C_beta`/`P_beta`/`eps_beta` are β.
    /// @{
    linalg::RealMatrix C; ///< MO coefficients α (or RKS), columns = orbitals.
    linalg::RealMatrix P; ///< Density matrix α (or RKS closed-shell P).
    std::vector<double> eps;      ///< Orbital energies α (or RKS) [Hartree].
    linalg::RealMatrix C_beta;    ///< UKS β MO coefficients (empty for RKS).
    linalg::RealMatrix P_beta;    ///< UKS β density (empty for RKS).
    std::vector<double> eps_beta; ///< UKS β orbital energies (empty for RKS).
    /// @}
    /// @name Fixed-density KS evaluation payload (Car–Parrinello / linearized-XL path)
    /// Populated when `GpwDft::run` is called with `max_iter == 0` and a
    /// density guess — one collocate→Poisson→XC→integrate, no SCF. Empty
    /// after a normal multi-iteration SCF. RKS: `H_ks` only. UKS: `H_ks` is
    /// \(H_\alpha\), `H_ks_beta` is \(H_\beta\). Multi-k RKS fixed-density
    /// fills `H_ks_k` / `S_k` (size == `n_kpts`) instead of the Γ fields;
    /// multi-k UKS additionally fills `H_ks_k_beta`.
    /// @{
    linalg::RealMatrix H_ks; ///< KS matrix H[P] / \(H_\alpha\) [Hartree] (nb × nb).
    linalg::RealMatrix H_ks_beta; ///< UKS \(H_\beta\) [Hartree] (empty for RKS).
    linalg::RealMatrix S;    ///< Overlap matrix at this geometry (nb × nb).
    /// @brief Multi-k fixed-density KS matrices \(H_k\) [Hartree] (empty for Γ
    ///        / multi-iteration SCF). One Hermitian nb×nb per sampled k.
    ///        RKS: total. Multi-k UKS fixed-density: the α channel
    ///        \(H_k^\alpha\), paired with `H_ks_k_beta`.
    std::vector<linalg::CplxMatrix> H_ks_k;
    /// @brief Multi-k UKS β fixed-density KS matrices \(H_k^\beta\) [Hartree]
    ///        (empty for RKS / Γ / multi-iteration SCF). Size == `n_kpts`,
    ///        paired with `H_ks_k` — the dual-spin Bloch XL-BOMD/CPMD payload.
    std::vector<linalg::CplxMatrix> H_ks_k_beta;
    /// @brief Multi-k overlaps \(S_k\) at the evaluation geometry (empty for Γ
    ///        / multi-iteration SCF). Size == `n_kpts` with `H_ks_k`.
    std::vector<linalg::CplxMatrix> S_k;
    /// @}
};

/// @brief Concrete GPW DFT method — members + build/run
///        (mirrors `cpu::pw::PW` / `cpu::scf::SCFSolver`).
///
/// **NOT a `Driver<...>` subclass** — this is the inner solver behind
/// `GPWEvaluator`. Construction fixes the configuration; `build()` runs
/// the one-time fail-fast validation against a reference atom set (the
/// no-fallback rule: invalid configuration throws instead of degrading);
/// `run()` is re-entrant and rebuilds every geometry-dependent structure
/// per call (grids, lattice images, S/T/Vpp — same policy as `PW`).
/// Move-only, like the sibling solvers.
class GpwDft {
  public:
    // ── Method parameters (were GpwConfig fields) ─────────────────────
    double cell[3] = {0, 0, 0}; ///< orthorhombic box lengths [Bohr].
    double cutoff_ry = 300.0;   ///< density grid cutoff [Ry].
    int ngrids = 1;             ///< CP2K multigrid levels.
    double rel_cutoff = 0.0;    ///< CP2K REL_CUTOFF [Ry].
    int n_occ = 0;              ///< RKS: doubly-occupied orbitals (N_elec / 2).
    /// UKS occupations (valence electrons per spin). When
    /// `n_alpha + n_beta > 0`, the unrestricted path is used and `n_occ` is
    /// ignored. High-spin O₂ (GTH-q6, 12 valence e⁻, multiplicity 3) → (7, 5).
    int n_alpha = 0;
    int n_beta = 0;
    int max_iter = 100;        ///< max SCF iterations.
    double tol = 1e-8;         ///< SCF energy convergence [Hartree].
    double eps_default = 1e-8; ///< Gaussian-tail cutoff.
    double mix = 0.5;          ///< linear density-mixing (diis_depth == 0).
    int diis_depth = 6;        ///< Pulay DIIS history depth.
    double lindep_threshold = 1e-8; ///< S-eigenvalue keep threshold.
    double level_shift = 0.0;       ///< virtual level shift [Hartree].
    XcKind xc =
        XcKind::PADE; ///< LDA (PADE) or GGA (PBE / PBEsol / RevPBE / RPBE / BLYP / OptB88VdW).
    bool nonperiodic = false;       ///< isolated (MT) Poisson solver.
    bool local_pp_on_grid = true;   ///< collocate GTH local SR PP on grid.
    bool d3 =
        false; ///< Grimme DFT-D3 zero damping (PBE). Exclusive vs d3bj/d4/ts/mbd/xdm.
    /// @brief Grimme DFT-D3 Becke–Johnson damping (PBE params, CP2K CN form).
    /// Mutually exclusive with `d3` / `d4` / `ts` / `mbd` / `xdm`.
    bool d3bj = false; ///< Grimme DFT-D3(BJ) dispersion.
    /// @brief Three-body ATM / C9 on top of `d3` or `d3bj` (CP2K
    ///        `CALCULATE_C9_TERM`). Ignored when neither D3 flag is set.
    bool d3_atm = false; ///< DFT-D3 Axilrod–Teller–Muto (C9).
    /// @brief DFT-D3 pair / ABC hard cutoff [Bohr], **doubled** convention:
    ///        CP2K `&PAIR_POTENTIAL R_CUTOFF x` (Å) sums to 2·x (it prints the
    ///        half). Default = CP2K default `R_CUTOFF 20.0` → 75.59 Bohr
    ///        (`kD3Cp2kPairCutoffBohr`). Energy, forces and stress all fold
    ///        the same value. Applies to `d3` / `d3bj` only.
    double d3_rcut_bohr = kD3Cp2kPairCutoffBohr;
    /// @brief Grimme DFT-D4 BJ damping (PBE params; periodic EEQ).
    /// Mutually exclusive with `d3` / `d3bj` / `ts` / `mbd` / `xdm`.
    /// Two-body only unless `d4_atm` is also set.
    bool d4 = false; ///< Grimme DFT-D4 dispersion.
    /// @brief Include Axilrod–Teller–Muto three-body (s9 = 1) with D4.
    ///        Requires `d4`. Default false (two-body product; `gpw-d4-water`).
    bool d4_atm = false;
    /// @brief Tkatchenko–Scheffler pairwise dispersion (PBE s_R=0.94, free-atom
    ///        volumes by default). Mutually exclusive with `d3` / `d3bj` / `d4` /
    ///        `mbd` / `xdm`.
    bool ts = false; ///< Tkatchenko–Scheffler (TS) dispersion.
    /// @brief Free-atom exchange-hole dipole moment (XDM) pairwise dispersion
    ///        (Becke–Johnson; QE NC-PBE a1/a2). Mutually exclusive with
    ///        `d3` / `d3bj` / `d4` / `ts` / `mbd` / `vdw_df` / `vdw_df2` /
    ///        `rvv10`. Geometry-only free-atom foothold (no Hirshfeld).
    bool xdm = false; ///< Free-atom XDM dispersion.
    /// @brief Many-body dispersion MBD@rsSCS (PBE β=0.83, free-atom volumes).
    ///        Default `mbd_k_mesh={0,0,0}` → isolated free-atom MBD@rsSCS.
    ///        Non-zero `mbd_k_mesh` → periodic Ewald + k-mesh (requires
    ///        `!nonperiodic`). Mutually exclusive with `d3` / `d3bj` / `d4` /
    ///        `ts` / `xdm` / `vdw_df` / `vdw_df2` / `rvv10`.
    bool mbd = false; ///< MBD@rsSCS dispersion.
    /// @brief MBD Brillouin-zone mesh (libMBD shift-½). All-zero keeps isolated
    ///        free-atom MBD; any positive component requires all three ≥ 1 and
    ///        a periodic cell. Typical ice / condensed-phase pin: `{2,2,2}`.
    int mbd_k_mesh[3] = {0, 0, 0};
    /// @brief Nonlocal vdW-DF correlation (DRSLL / DF1). Self-consistent
    ///        \(V_{\mathrm{nl}}\) each SCF step; analytic forces via veff;
    ///        \(E_{\mathrm{disp}}\) on the final SC density. CP2K
    ///        `VDW_POTENTIAL NON_LOCAL TYPE DRSLL`. Mutually exclusive with
    ///        `d3` / `d3bj` / `d4` / `ts` / `mbd` / `vdw_df2` / `rvv10`.
    ///        Periodic, Γ closed-shell RKS (single-grid or REL_CUTOFF
    ///        multigrid `ngrids > 1`); k path / UKS reject fail-fast.
    bool vdw_df = false; ///< vdW-DF (DRSLL) nonlocal correlation.
    /// @brief Nonlocal vdW-DF2 (LMKLL / DF2). Same SC product path as `vdw_df`;
    ///        Z_ab = −1.887. CP2K `NON_LOCAL TYPE LMKLL`. Same Γ RKS scope
    ///        (single-grid or multigrid) as `vdw_df`.
    bool vdw_df2 = false; ///< vdW-DF2 (LMKLL) nonlocal correlation.
    /// @brief Nonlocal rVV10 (Sabatini et al.). SC product path; CP2K
    ///        `NON_LOCAL TYPE RVV10` (b=6.3, C=0.0093). Same Γ RKS scope
    ///        (single-grid or multigrid) as `vdw_df`.
    bool rvv10 = false; ///< rVV10 nonlocal correlation.
    /// @brief Scale TS/MBD free-atom parameters by Hirshfeld \(V/V_{\mathrm{free}}\)
    ///        from the final SCF total valence density (post-SCF,
    ///        density-dependent). Requires `ts` or `mbd`. RKS and UKS (Γ and
    ///        multi-k); UKS uses \(\rho=\rho_\alpha+\rho_\beta\). Free-atom
    ///        path remains the default when false.
    bool hirshfeld = false; ///< Hirshfeld volume ratios for TS/MBD.
    /// @brief OT minimizer instead of diagonalization (Γ RKS / Γ UKS,
    ///        including Γ-UKS + Dudarev DFT+U via `plus_u`, multi-k
    ///        closed-shell RKS, multi-k RKS + metallic FD/MP smearing,
    ///        multi-k collinear UKS (integer **and** FD/MP smear),
    ///        multi-k RKS + Dudarev DFT+U (integer and with metallic
    ///        smear), and multi-k UKS + Dudarev DFT+U (integer **and**
    ///        with metallic smear — free-energy PR+ path).
    bool use_ot = false;
    /// @brief Monkhorst–Pack k mesh (CP2K `&KPOINTS SCHEME MONKHORST-PACK`
    ///        convention: κ_d = (2r − N_d − 1)/(2N_d), so even N is a
    ///        Γ-avoiding shifted mesh). `{1,1,1}` keeps the Γ-point path.
    ///        The k-point path is periodic, single-grid, `local_pp_on_grid`
    ///        only — RKS (`n_occ`) or collinear UKS (`n_alpha`/`n_beta`);
    ///        multi-k Dudarev DFT+U (`plus_u`) is supported for SCF energy
    ///        (DIIS and OT, RKS and UKS; with or without metallic smear).
    ///        Multi-k RKS/UKS OT (`use_ot`) is product including all
    ///        OT+DFT+U and OT+DFT+U+smear combos; Γ-UKS OT+DFT+U is
    ///        product on the Γ path. Multigrid / nonperiodic throw at `build()`.
    int k_mesh[3] = {1, 1, 1};
    /// @brief Fold ±k pairs by time reversal (exact for collinear RKS/UKS
    ///        without SOC; weights doubled on the retained partner).
    bool k_time_reversal = true;
    /// @brief Explicit fractional k list (CP2K `SCHEME GENERAL`); non-empty
    ///        overrides `k_mesh`. Same length as `k_point_weights`
    ///        (weights must sum to 1; no implicit normalization).
    std::vector<std::array<double, 3>> k_points;
    std::vector<double> k_point_weights; ///< weights for `k_points` [Σ = 1].
    /// @brief Dudarev PLUS_U entries (empty = off). U−J in eV on each entry.
    std::vector<PlusU> plus_u;
    /// @brief MP2 virtual-space cap for `mp2()` (0 = all kept virtuals).
    int mp2_n_virt = 0;
    /// @brief Product RI-GPW path for `mp2()` (plane-wave resolution of the
    ///        pair-density identity; Del Ben 2012 / CP2K MP2_GPW). `false`
    ///        keeps the Poisson real-space foothold tensor (DIRECT_CANONICAL
    ///        class). Isolated (`nonperiodic`) cells always use Poisson.
    bool mp2_ri = true;
    /// @brief Grimme opposite-spin scale for `mp2()` (1.0 = plain MP2;
    ///        SCS-MP2 1.2 — same member name as the molecular methods).
    double mp2_os_scale = 1.0;
    /// @brief Grimme same-spin scale for `mp2()` (1.0 = plain MP2; SCS 1/3).
    double mp2_ss_scale = 1.0;
    /// @brief Optional per-atom AFM/FM spin seed for the initial UKS density
    ///        (units-free sign/magnitude; size 0 or n_atoms). Applied only to
    ///        the cold-start densities, not as an external potential.
    std::vector<double> atom_spin_guess;
    /// @brief Metallic occupation smearing (default: fixed integer fillings).
    ///        Product path: multi-k RKS (DIIS **and** OT, with or without
    ///        `plus_u`) and multi-k collinear UKS (DIIS **and** OT). Multi-k
    ///        UKS OT + DFT+U + smear remains a residual.
    Smearing smearing = Smearing::none;
    /// @brief Smearing width [Ry] (QE `degauss` / CP2K \(k_B T_e\)). Required
    ///        `> 0` when `smearing != none`. Converted to Ha inside the SCF.
    double degauss = 0.02;
    /// @brief Methfessel–Paxton order \(N\) (used when `smearing == methfessel_paxton`).
    int mp_order = 1;
    /// @brief Extra empty bands beyond `n_occ` (RKS) or \(\max(n_\alpha,n_\beta)\)
    ///        (UKS) — CP2K `ADDED_MOS` / PW `extra_bands`. GPW diagonalizes the
    ///        full AO spectrum; this is API parity.
    int extra_bands = 0;

    GpwDft() = default;
    GpwDft(const GpwDft&) = delete;
    GpwDft& operator=(const GpwDft&) = delete;
    GpwDft(GpwDft&&) = default;
    GpwDft& operator=(GpwDft&&) = default;

    /// @brief One-time fail-fast validation against a reference system.
    ///
    /// @param reference Atoms the solver will be run on (used for
    ///        structural validation only; `run()` takes the live set).
    /// @throws std::invalid_argument when the atom set is empty, when
    ///         neither RKS (`n_occ >= 1`) nor UKS (`n_alpha+n_beta >= 1`) is
    ///         set, or a periodic cell edge is non-positive.
    void build(const std::vector<GpwAtom>& reference);

    /// @return true when this method is configured for UKS (`n_alpha+n_beta>0`).
    bool is_uks() const { return n_alpha + n_beta > 0; }

    /// @brief Run the GPW SCF (RKS or UKS) at the given geometry.
    ///
    /// Re-entrant: geometry-dependent structures are rebuilt per call.
    /// @param atoms Atom set at the current geometry [Bohr].
    /// @param guess_C Optional warm-start MO coefficients from an earlier run.
    /// @param guess_P Optional initial-density override (n_ao × n_ao) —
    ///        XL-BOMD auxiliary density or SGCP ASPC prediction. RKS:
    ///        total closed-shell \(P\). UKS: \(P_\alpha\) (requires
    ///        non-null `guess_P_beta`). On the DIIS path the density
    ///        replaces the C0-derived \(P\). On the **OT** path the
    ///        density is purified to natural orbitals that seed the OT
    ///        manifold (RKS: total \(P\) → \(C_0\); UKS: dual-spin
    ///        \(P_\alpha/P_\beta\) → \(C_a/C_b\) — SGCP OT corrector).
    ///        **Γ path only** for real `guess_P` — multi-k uses
    ///        `guess_P_k`. Shape mismatch throws inside the solve.
    /// @param guess_P_beta Optional UKS \(P_\beta\) initial density
    ///        (n_ao × n_ao). Must be null for RKS. For UKS both spin
    ///        guesses are required together (no silent cold start of one
    ///        channel).
    /// @param guess_P_k Optional multi-k initial densities (one complex
    ///        Hermitian matrix per sampled k, size == `n_kpts` after mesh /
    ///        TR fold). RKS: total \(P_k\). UKS: α-spin \(P_k^\alpha\)
    ///        (requires matching `guess_P_k_beta`). On the RKS **OT** path
    ///        each \(P_k\) is purified to natural orbitals that seed the
    ///        per-k OT manifold (complex mirror of the Γ `guess_P` warm
    ///        start — multi-k SGCP OT corrector). Real `guess_P` with a
    ///        multi-k mesh throws (no silent Γ guess on the k path).
    /// @param guess_P_k_beta Optional multi-k UKS β densities \(P_k^\beta\)
    ///        (size == `n_kpts`; UKS-only — RKS + non-null throws).
    /// @return The owned result of this run (valid until the next `run()`).
    /// @throws std::logic_error when called before `build()`.
    /// @throws std::invalid_argument for UKS with only one spin guess,
    ///         RKS + non-null `guess_P_beta` / `guess_P_k_beta`, multi-k +
    ///         real-only density guess, multi-k UKS partial k-spin guess,
    ///         or shape / count mismatch.
    GpwResult& run(const std::vector<GpwAtom>& atoms,
                   const linalg::RealMatrix* guess_C = nullptr,
                   const linalg::RealMatrix* guess_P = nullptr,
                   const linalg::RealMatrix* guess_P_beta = nullptr,
                   const std::vector<linalg::CplxMatrix>* guess_P_k = nullptr,
                   const std::vector<linalg::CplxMatrix>* guess_P_k_beta =
                       nullptr);

    /// @brief Result of the most recent `run()` (default-constructed —
    ///        `converged == false` — before the first run).
    const GpwResult& last_result() const { return result_; }

    /// @brief Standalone Γ overlap \(S\) at the given geometry — basis
    ///        assembly only, no collocation / Poisson / XC.
    ///
    /// Bit-identical to the `GpwResult::S` a fixed-density `run()` produces at
    /// the same geometry: the same flattened basis, the same Gaussian-tail
    /// radii, the same pair-image counts and the same summation order (one
    /// shared assembly routine, not a re-implementation). Car–Parrinello uses
    /// it to align the propagated coefficients against \(S\) **before** the
    /// step's single KS evaluation, so the step evaluates the KS matrix once
    /// instead of twice.
    ///
    /// @param atoms Atom set at the current geometry [Bohr] (as for `run`).
    /// @return Overlap matrix \(S\) [nb × nb], dimensionless.
    /// @throws std::logic_error when called before `build()`.
    /// @throws std::invalid_argument when this method is configured for the
    ///         k-point path (non-Γ `k_mesh`, explicit `k_points`, or metallic
    ///         `smearing`) — that path has no real Γ overlap; use
    ///         `overlap_k()`.
    linalg::RealMatrix overlap(const std::vector<GpwAtom>& atoms) const;

    /// @brief Standalone multi-k Bloch overlaps \(S_k\) at the given geometry.
    ///
    /// The k-point twin of `overlap()`: Hermitian nb × nb per sampled k, in
    /// the same k order and bit-identical to the `GpwResult::S_k` a
    /// fixed-density multi-k `run()` produces (same shared phased assembly).
    ///
    /// @param atoms Atom set at the current geometry [Bohr] (as for `run`).
    /// @return One \(S_k\) [nb × nb] per sampled k (size == `n_kpts`).
    /// @throws std::logic_error when called before `build()`.
    /// @throws std::invalid_argument when this method is configured for the Γ
    ///         path (no k mesh / list / smearing) — use `overlap()`.
    std::vector<linalg::CplxMatrix>
    overlap_k(const std::vector<GpwAtom>& atoms) const;

    /// @brief GPW MP2 correlation on the converged orbitals of the last
    ///        `run()` — RKS Γ/multi-k or UKS Γ/multi-k (UMP2).
    ///
    /// **Γ RKS** (`n_kpts == 0`, closed-shell): product RI-GPW (`mp2_ri`,
    /// default true, periodic) or Poisson foothold (`mp2_ri = false` /
    /// isolated).
    ///
    /// **Γ UKS** (`n_alpha`/`n_beta`, no k-mesh): unrestricted MP2
    /// (same-spin αα/ββ + opposite-spin αβ; molecular / PySCF UMP2
    /// convention) on the dual α/β MO sets.
    ///
    /// **Multi-k RKS** (`n_kpts >= 1` with `C_k` populated): momentum-
    /// conserving RI-GPW sum over sampled k (Del Ben / CP2K `MP2_GPW` +
    /// `&KPOINTS`). Equal-weight meshes
    /// (\(E=\frac1{N_k}\sum_{k_i k_j k_a}\)); pair densities are complex,
    /// Coulomb via linear real Poisson on Re/Im. Requires `mp2_ri` product
    /// (Poisson multi-k foothold not offered). Prefer
    /// `k_time_reversal = false` so weights stay equal.
    ///
    /// **Multi-k UKS** (`n_alpha`/`n_beta` with `C_k` + `C_k_beta`): same
    /// momentum-conserving RI-GPW path with unrestricted doubles (SS αα/ββ
    /// + OS αβ). Requires `mp2_ri` and a periodic cell.
    ///
    /// Virtual space capped by `mp2_n_virt` (per spin for UMP2); `e_corr`
    /// applies `mp2_os_scale` / `mp2_ss_scale`.
    /// @param atoms Same geometry [Bohr] the last `run()` converged on.
    /// @return The owned MP2 result (valid until the next `mp2()`).
    /// @throws std::logic_error before `build()` or a converged `run()`.
    /// @throws std::invalid_argument for `use_ot` (no canonical virtuals),
    ///         when multi-k `C_k` / `C_k_beta` is missing, when multi-k
    ///         and `!mp2_ri`, when no virtuals remain, or for a vanishing
    ///         min HOMO–LUMO gap (< 1e-6 Ha).
    const Mp2Result& mp2(const std::vector<GpwAtom>& atoms);

    /// @brief GPW direct RPA (dRPA) correlation on the converged orbitals of
    ///        the last `run()` — Γ/multi-k RKS, or **Γ/multi-k UKS (URPA)**.
    ///
    /// **Γ RKS** (`n_kpts == 0`): Coulomb-only singlet Casida
    /// (\(\alpha_{\mathrm{hf}}=0\)) from the same \((ia|jb)\) path as
    /// `mp2()` (product RI-GPW when `mp2_ri` and periodic; else Poisson
    /// foothold), then Furche
    /// \(E_c=\tfrac12(\sum\Omega_s-\mathrm{Tr}\,A)\).
    ///
    /// **Multi-k RKS** (`n_kpts >= 1` with `C_k` populated): **finite-\(q\)
    /// inter-\(k\)** RI-GPW dRPA — for each transfer \(q\) on the closed MP
    /// mesh, Casida from transitions \(i_k\to a_{k+q}\) with Coulomb at
    /// \(q\), Furche-averaged
    /// \(E_c=\sum_q w_q\tfrac12(\sum_s\Omega_{q,s}-\mathrm{Tr}\,A_q)\).
    /// Requires `mp2_ri` product and a periodic cell. Prefer
    /// `k_time_reversal = false` so weights stay equal.
    ///
    /// **Γ UKS (URPA):** spin-conserving dRPA over α→α ⊕ β→β (Coulomb
    /// kernel, spins explicit — no closed-shell factor 2). Same-spin
    /// tensors via `mo_eri_ovov{,_ri}`; cross-spin via
    /// `mo_eri_ovov_ab{,_ri}`; Furche \(E_c\) from
    /// `internal::direct_rpa_uhf`. When \(C^\alpha\approx C^\beta\) and
    /// \(n_\alpha=n_\beta\), \(E_c\) dual-gates closed-shell dRPA.
    ///
    /// **Multi-k UKS (URPA):** finite-\(q\) inter-\(k\) spin-conserving
    /// dRPA on \(C_k\) / \(C_k^\beta\) (`internal::direct_rpa_uhf_energy_kpoints`).
    /// Requires `mp2_ri` and a periodic cell. Dual-gates multi-k RKS dRPA
    /// when \(C^\alpha\approx C^\beta\). `RpaResult::n_occ` / `n_virt` are
    /// spin-orbital totals (\(n_\alpha+n_\beta\), …); `omega_ha` length is
    /// the composite OV space \(n_{\mathrm{ov}}^\alpha+n_{\mathrm{ov}}^\beta\).
    ///
    /// Virtual space capped by `mp2_n_virt`. Total energy is dRPA@KS
    /// (\(E_{\mathrm{SCF}}+E_c\); no EXX replacement).
    /// @param atoms Same geometry [Bohr] the last `run()` converged on.
    /// @return The owned RPA result (valid until the next `direct_rpa()`).
    /// @throws std::logic_error before `build()` or a converged `run()`,
    ///         or when UKS β orbitals are missing.
    /// @throws std::invalid_argument for `use_ot`, when multi-k `C_k` /
    ///         `C_k_beta` is missing, when multi-k and `!mp2_ri`, when no
    ///         virtuals remain, or for a vanishing HOMO–LUMO gap (< 1e-6 Ha).
    /// @throws std::runtime_error when the dRPA reduction is ill-conditioned.
    const RpaResult& direct_rpa(const std::vector<GpwAtom>& atoms);

    /// @brief Spectral G0W0 quasiparticle energies on the converged KS
    ///        orbitals of the last `run()` — Γ/multi-k RKS, or **Γ/multi-k
    ///        UKS (UGW)**.
    ///
    /// **RKS:** dRPA screening reuses the same dielectric path as multi-k
    /// `direct_rpa` / `mp2` (product RI-GPW when `mp2_ri` and periodic;
    /// multi-k requires `mp2_ri = true`). Correlation self-energy is the
    /// spectral sum over finite-\(q\) RPA poles. Exact-exchange diagonals
    /// use truncated (MT) Coulomb. Local \(v_{\mathrm{xc}}\) matches the SCF.
    ///
    /// **UKS (Γ and multi-k):** spin-conserving dRPA over α→α ⊕ β→β
    /// (Coulomb kernel, spins explicit), same-spin spectral
    /// \(\Sigma_c^\sigma\), per-spin MT \(\Sigma_x\), spin-resolved
    /// \(v_{\mathrm{xc}}^\sigma\). Multi-k uses finite-\(q\) inter-\(k\)
    /// \(W\) on `C_k` / `C_k_beta` (requires `mp2_ri = true`). Primary
    /// report is the **α-channel** spectrum / min-direct gap (dual-gates
    /// RKS G0W0 when \(n_\alpha=n_\beta\) and orbitals match).
    ///
    /// Virtual space for screening capped by `mp2_n_virt` (0 = all kept
    /// virtuals); QP levels cover every MO of the reported spectrum.
    /// @param atoms Same geometry [Bohr] the last `run()` converged on.
    /// @return The owned G0W0 result (valid until the next `g0w0()`).
    /// @throws std::logic_error before `build()` or a converged `run()`,
    ///         or when UKS β orbitals are missing.
    /// @throws std::invalid_argument for `use_ot`, when multi-k `C_k` /
    ///         `C_k_beta` is missing, when multi-k and `!mp2_ri`, when no
    ///         virtuals remain, or for a vanishing HOMO–LUMO gap
    ///         (< 1e-6 Ha).
    /// @throws std::runtime_error when the dRPA reduction is ill-conditioned.
    const G0W0Result& g0w0(const std::vector<GpwAtom>& atoms);

  private:
    GpwResult result_;     ///< Owned result buffer (last run).
    Mp2Result mp2_result_; ///< Owned MP2 result buffer (last mp2()).
    RpaResult rpa_result_; ///< Owned RPA result buffer (last direct_rpa()).
    G0W0Result g0w0_result_; ///< Owned G0W0 result buffer (last g0w0()).
    bool built_ = false;   ///< Whether build() validated the configuration.
};

/// @brief Finite-difference forces F_a = -dE/dR_a via central differences.
///
/// Runs a `GpwDft` at +/- h displacements of each atom along each axis
/// (6*N_atoms SCF solves). Correct but expensive; product path is
/// `gpw_forces_analytic`. Returns forces [Hartree/Bohr], one xyz triple per atom.
/// @param h Displacement step [Bohr].
std::vector<std::array<double, 3>>
gpw_forces_fd(const std::vector<GpwAtom>& atoms, const GpwDft& method,
              double h = 1e-3);

/// @brief Analytic cell stress σ [Hartree/Bohr³], Voigt order xx,yy,zz,xy,xz,yz.
///
/// Product path: one SCF, then Hellmann–Feynman fixed-\(P\) central FD of the
/// non-self-consistent total energy under orthorhombic diagonal strain at fixed
/// fractional coordinates (GPW analogue of `cpu::pw::pw_stress`). Supports
/// Gamma RKS, Gamma UKS (spin-resolved fixed-\(P_\sigma\)), Dudarev DFT+U
/// on Γ and multi-k (`plus_u`, \(E_U\) / BZ-averaged \(E_U\) on the fixed-\(P\)
/// Lagrangian), closed-shell multi-k RKS (k-summed fixed-\(P_k\)), multi-k
/// collinear UKS (fixed \(\{P_k^\sigma,W_k^\sigma\}\) + spin XC), and
/// geometry-only dispersion (`d3` / `d3bj` / `d4` / `ts` / `xdm` / `mbd`)
/// rebuilt under strain. Off-diagonal shear is 0 (orthorhombic cell only).
/// Multi-SCF oracle: `gpw_stress_fd`.
///
/// @param atoms Geometry [Bohr] + basis/PP (as for `GpwDft::run`).
/// @param method SCF configuration; `cell` must match the atom box [Bohr].
/// @param h Dimensionless strain step for the fixed-\(P\) energy FD (default 2e-3).
/// @param axes Per-axis mask (x,y,z); a masked axis skips samples and reports
///        exactly 0 — for variable-cell relaxation of constrained box edges.
/// @param guess_C Optional warm-start MO coefficients for the reference SCF.
/// @return Voigt stress [Hartree/Bohr³].
std::array<double, 6> gpw_stress(const std::vector<GpwAtom>& atoms,
                                 const GpwDft& method, double h = 2e-3,
                                 const std::array<bool, 3>& axes = {true, true,
                                                                    true},
                                 const linalg::RealMatrix* guess_C = nullptr);

/// @brief Analytic cell stress σ [Hartree/Bohr³] at a **supplied** state —
///        no reference SCF.
///
/// Same fixed-\(P\) Hellmann–Feynman central FD as `gpw_stress`, but the
/// density payload is handed in instead of re-converged: Γ needs
/// `state.P` / `state.C` / `state.eps` (+ β fields for UKS); multi-k needs
/// `state.C_occ_k` / `state.eps_k` (+ β for UKS). `state.converged` must be
/// true (a stationary payload — a converged SCF result, or the zero-SCF
/// linearized XL-BOMD shadow payload, whose warm step marks the shadow
/// functional stationary). This is the cell analogue of
/// `gpw_forces_at_density`: the stress of the energy surface at the density
/// you already have, used by `GPWEvaluator` for the shadow-density stress
/// on linearized XL warm steps.
///
/// @param atoms Geometry [Bohr] + basis/PP (as for `GpwDft::run`).
/// @param method SCF configuration; `cell` must match the atom box [Bohr].
/// @param state Stationary payload (see above); `converged` must be true.
/// @param h Dimensionless strain step for the fixed-\(P\) energy FD (default 2e-3).
/// @param axes Per-axis mask (x,y,z); a masked axis skips samples and reports
///        exactly 0.
/// @return Voigt stress [Hartree/Bohr³], order xx,yy,zz,xy,xz,yz.
std::array<double, 6> gpw_stress(const std::vector<GpwAtom>& atoms,
                                 const GpwDft& method, const GpwResult& state,
                                 double h = 2e-3,
                                 const std::array<bool, 3>& axes = {true, true,
                                                                    true});

/// @brief Multi-SCF strain-FD cell stress σ [Hartree/Bohr³] (oracle / twin).
///
/// Fully re-converges the SCF at each strained cell (±h per diagonal axis).
/// Same sign convention and orthorhombic scope as `gpw_stress`. Use only for
/// dual-gate tests and finite-difference twins — not the product hot path.
///
/// @param atoms Geometry [Bohr] + basis/PP (as for `GpwDft::run`).
/// @param method SCF configuration; `cell` must match the atom box [Bohr].
/// @param h Dimensionless strain step (default 2e-3).
/// @param axes Per-axis mask (x,y,z); masked axes report exactly 0.
/// @param guess_C Optional warm-start MO coefficients for every strained SCF.
/// @return Voigt stress [Hartree/Bohr³].
std::array<double, 6>
gpw_stress_fd(const std::vector<GpwAtom>& atoms, const GpwDft& method,
              double h = 2e-3,
              const std::array<bool, 3>& axes = {true, true, true},
              const linalg::RealMatrix* guess_C = nullptr);

} // namespace gpw
} // namespace cpu
} // namespace ATV
