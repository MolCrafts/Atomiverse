/// @file cpu/gpw/bands_dos.h
/// @brief Band-structure and density-of-states post-processing on GPW.
///
/// **Bands** — after multi-k closed-shell SCF on `cpu::gpw::GpwDft`, non-
/// self-consistent diagonalization of
/// \(H_k = T_k + V_\mathrm{nl}^k + V^k[V_\mathrm{eff}]\) on a high-symmetry
/// fractional path yields \(\varepsilon_n(\mathbf{k})\) (Hartree). Twin:
/// CP2K Quickstep `&PRINT &BAND_STRUCTURE` on GTH-GPW.
///
/// **Dos** — Gaussian-broadened \(D(E)\) from multi-k Bloch eigenvalues on
/// the SCF mesh. Twin: CP2K Quickstep `&PRINT &DOS` / PDOS class on GTH-GPW.
///
/// Parameters are members of `Bands` / `Dos` — no public `*Config`, no
/// factories.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Result of a GPW band-structure (NSCF path) calculation.
struct BandsResult {
    /// Fractional k-points used for the path (same order as eigenvalues).
    std::vector<std::array<double, 3>> kfrac;
    /// Eigenvalues \(\varepsilon_n(\mathbf{k})\) [Hartree], layout `[ik][ib]`.
    /// RKS: closed-shell channel. UKS: α channel (β in `eigenvalues_beta_ha`).
    std::vector<std::vector<double>> eigenvalues_ha;
    /// UKS β eigenvalues [Hartree], layout `[ik][ib]` (empty for RKS).
    std::vector<std::vector<double>> eigenvalues_beta_ha;
    /// 1 = restricted (RKS); 2 = collinear spin-polarized (UKS).
    int nspin = 1;
    double e_vbm_ha = 0;   ///< Valence-band maximum [Hartree] (fixed-occ).
    double e_cbm_ha = 0;   ///< Conduction-band minimum [Hartree] (fixed-occ).
    double gap_ha = 0;     ///< \(E_\mathrm{cbm}-E_\mathrm{vbm}\) [Hartree] (0 if metallic).
    int vbm_k = 0;         ///< k-index of VBM on the path.
    int cbm_k = 0;         ///< k-index of CBM on the path.
    int vbm_band = 0;      ///< Band index of VBM (0-based).
    int cbm_band = 0;      ///< Band index of CBM (0-based).
    /// @brief SCF Fermi level [Hartree] when metallic smear is on (0 if fixed occ).
    double e_fermi_ha = 0;
    bool scf_converged = false; ///< Whether the preceding SCF converged.
    /// @brief SCF free energy \(F\) (smear) or total energy [Hartree].
    double energy_ha = 0;
};

/// @brief High-symmetry-path band structure (NSCF on frozen GPW \(V_\mathrm{eff}\)).
///
/// SCF uses `method` members (`cutoff_ry`, `k_mesh` / `k_points`, `xc`,
/// `n_occ` / UKS `n_alpha`/`n_beta`, optional `smearing`). Path k-points are
/// `kfrac` (filled by `set_diamond_path` when empty at `compute` time).
///
/// Product paths:
/// - closed-shell RKS, fixed integer occupations (`reg_gpw-bands-si`)
/// - collinear UKS multi-k, frozen dual \(V_{\mathrm{eff}}^\sigma\)
///   (`reg_gpw-bands-uks`)
/// - metallic FD/MP smear, frozen \(V_{\mathrm{eff}}\) + report \(E_F\)
///   (`reg_gpw-bands-metallic`)
///
/// Multigrid, nonperiodic, DFT+U, and dispersion flags throw with a how-to
/// (no silent degradation).
class Bands {
  public:
    /// @brief Number of bands to report per path k (lowest eigenvalues).
    ///        Must cover occupied spectrum so the gap / path is defined. GPW
    ///        diagonalizes the full AO spectrum; this only caps the report.
    int n_bands = 12;
    /// @brief Fractional (crystal) k-list. Empty → `set_diamond_path(4)`.
    std::vector<std::array<double, 3>> kfrac;

    /// @brief FCC/diamond high-symmetry path Γ–X–W–L–Γ–K (fractional of the
    ///        simulation cell reciprocal lattice — same SPECIAL_POINT list as
    ///        `cpu::pw::Bands` and CP2K `&BAND_STRUCTURE` twins).
    /// @param n_seg Number of samples per segment (including endpoints once).
    void set_diamond_path(int n_seg = 4);

    /// @brief Multi-k SCF + NSCF bands for the geometry in `atoms`.
    ///
    /// @param method GPW DFT method (`build()` already done; k-mesh / list
    ///               used for the SCF density; not mutated).
    /// @param atoms  Atom set at the current geometry [Bohr] (same as `build`).
    /// @return Owned `BandsResult` with eigenvalues [Hartree] and gap / \(E_F\).
    /// @throws std::logic_error when `method` was not `build()`'d.
    /// @throws std::invalid_argument on multigrid / nonperiodic / analytic local
    ///         PP / DFT+U / dispersion / empty path / insufficient n_bands.
    BandsResult compute(GpwDft& method, const std::vector<GpwAtom>& atoms) const;
};

/// @brief Result of a Gaussian-broadened multi-k density of states (GPW).
struct DosResult {
    /// Bin centres [Hartree].
    std::vector<double> energy_ha;
    /// DOS \(D(E)\) [states / Ha / cell]. Restricted: each band carries
    /// spin degeneracy 2. Collinear: total \(D = D_\alpha + D_\beta\) with
    /// spin-orbital weight 1 (when UKS path is used).
    std::vector<double> dos;
    /// Spin-up \(D_\alpha(E)\) when `nspin == 2`; empty when restricted.
    std::vector<double> dos_alpha;
    /// Spin-down \(D_\beta(E)\) when `nspin == 2`; empty when restricted.
    std::vector<double> dos_beta;
    /// 1 = restricted (RKS); 2 = collinear spin-polarized (UKS).
    int nspin = 1;
    /// Occupied DOS integral \(\int_{-\infty}^{E_\ast} D(E)\,dE\)
    /// (target \(\approx n_e\)). Fixed occupations: \(E_\ast=E_\mathrm{vbm}\);
    /// metallic smearing: \(E_\ast=E_F\) (`e_fermi_ha`).
    double n_elec_to_vbm = 0;
    double e_vbm_ha = 0;   ///< Valence-band maximum [Ha] (fixed-occ / gapped).
    double e_cbm_ha = 0;   ///< Conduction-band minimum [Ha].
    double gap_ha = 0;     ///< \(E_\mathrm{cbm}-E_\mathrm{vbm}\) [Ha] (0 if no empty).
    double e_fermi_ha = 0; ///< Fermi level [Ha] (0 if fixed occupations).
    bool scf_converged = false;
    /// @brief SCF free energy \(F\) (smear) or total energy [Ha].
    double scf_energy_ha = 0;
    /// @brief Number of k-points in the DOS mesh (after TR fold if on).
    int n_kpts = 0;
    /// @brief Bands used per k (after optional `Dos::n_bands` cap).
    int n_bands = 0;
};

/// @brief Gaussian-broadened DOS from multi-k GPW eigenvalues.
///
/// SCF uses `method` members (`cutoff_ry`, `xc`, `n_occ` / UKS, …). This
/// object sets `method.k_mesh = {k_mesh,k_mesh,k_mesh}` for the SCF that
/// supplies \(\varepsilon_{nk}\) (same mesh used for the DOS histogram —
/// CP2K `&PRINT &DOS` class on the SCF k set; no denser NSCF mesh).
///
/// Fixed occupations (`method.smearing == none`): integer-fill SCF;
/// occupied integral is to VBM. Metallic smearing: free-energy SCF;
/// integral is to SCF \(E_F\) (product pin: `reg_gpw-dos-metallic`).
///
/// **Scope:** closed-shell RKS multi-k (fixed-occ `reg_gpw-dos-si`; metallic
/// FD `reg_gpw-dos-metallic`) and collinear UKS multi-k when `method.is_uks()`
/// fills per-spin channels (g=1; `reg_gpw-dos-uks`).
class Dos {
  public:
    /// @brief DOS / SCF k-mesh per dimension (MP \(k_\mathrm{mesh}^3\)).
    ///        Must be \(\ge 1\); multi-k capability requires at least one
    ///        mesh component \(\ne 1\) (or an explicit `method.k_points`
    ///        list left alone when `use_method_k_list`).
    int k_mesh = 2;
    /// @brief Bands per k used in the DOS sum (0 = all AO eigenvalues from
    ///        the multi-k diagonalization). When > 0, only the lowest
    ///        `n_bands` states per k enter \(D(E)\).
    int n_bands = 0;
    /// @brief Gaussian broadening width [Hartree].
    double sigma_ha = 0.025;
    /// @brief Number of energy bins for \(D(E)\).
    int n_bins = 200;
    /// @brief Energy window [Ha]; if `e_max_ha <= e_min_ha`, auto from evals.
    double e_min_ha = 0;
    double e_max_ha = 0;
    /// @brief When true, do not overwrite `method.k_mesh` / `k_points` —
    ///        use the mesh already configured on the method (explicit lists
    ///        and TR fold preserved). Default false (set cubic MP mesh).
    bool use_method_k_list = false;

    /// @brief Multi-k SCF + Gaussian DOS for the geometry in `atoms`.
    ///
    /// @param method GPW DFT method (`build()` already done). Unless
    ///               `use_method_k_list`, `k_mesh` is set to
    ///               `{k_mesh,k_mesh,k_mesh}` and explicit `k_points` cleared.
    /// @param atoms  Atom set at the current geometry [Bohr].
    /// @return Owned `DosResult` (energies [Ha]).
    /// @throws std::invalid_argument on bad parameters or non multi-k result.
    /// @throws std::logic_error if `method` was not built.
    DosResult compute(GpwDft& method, const std::vector<GpwAtom>& atoms) const;
};

/// @brief Locate VBM/CBM from a spectrum with `n_occ` occupied bands.
///
/// Pure helper for unit tests and the post-processor. Eigenvalues must be
/// sorted ascending per k-point. Units are those of the input (Ha or Ry).
///
/// @param evals  `[ik][ib]` eigenvalues (sorted ascending in `ib`).
/// @param n_occ  Number of occupied bands (\(n_e/2\) closed shell).
/// @param e_vbm  Output valence-band maximum.
/// @param e_cbm  Output conduction-band minimum.
/// @param vbm_k  Output k-index of VBM.
/// @param cbm_k  Output k-index of CBM.
/// @param vbm_b  Output band index of VBM.
/// @param cbm_b  Output band index of CBM.
/// @return gap = e_cbm − e_vbm.
double gap_from_spectrum(const std::vector<std::vector<double>>& evals,
                         int n_occ, double& e_vbm, double& e_cbm, int& vbm_k,
                         int& cbm_k, int& vbm_b, int& cbm_b);

/// @brief Gaussian DOS on a uniform energy grid (unit-testable free math).
///
/// \(D(E_i)=\sum_{n k} w_{nk}\, (g/\sqrt{2\pi}\sigma)\,
/// \exp(-(E_i-\varepsilon)^2/(2\sigma^2))\). Default \(g=2\) is closed-shell
/// spin degeneracy; collinear spin-orbitals use \(g=1\).
///
/// @param evals            Flat list of eigenvalues [energy units].
/// @param weights          Per-eigenvalue k-weight (same length as evals).
/// @param e_min            Grid start.
/// @param e_max            Grid end.
/// @param n_bins           Number of bins.
/// @param sigma            Gaussian width (same energy units).
/// @param energy           Output bin centres.
/// @param dos              Output \(D(E)\).
/// @param spin_degeneracy  Band spin factor \(g\) (default 2; spin channels use 1).
void gaussian_dos(const std::vector<double>& evals,
                  const std::vector<double>& weights, double e_min,
                  double e_max, int n_bins, double sigma,
                  std::vector<double>& energy, std::vector<double>& dos,
                  double spin_degeneracy = 2.0);

/// @brief Trapezoid \(\int_{-\infty}^{E_\mathrm{cut}} D(E)\,dE\) on a uniform grid.
///
/// @param energy  Bin centres (ascending, uniform spacing).
/// @param dos     \(D(E)\) at each bin.
/// @param e_cut   Upper integration limit (same units as `energy`).
/// @return Integrated electron count (states / cell).
double integrate_dos_to(const std::vector<double>& energy,
                        const std::vector<double>& dos, double e_cut);

} // namespace gpw
} // namespace cpu
} // namespace ATV
