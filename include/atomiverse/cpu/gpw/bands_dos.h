/// @file cpu/gpw/bands_dos.h
/// @brief Density-of-states post-processing on the GPW multi-k mesh (CPU).
///
/// After multi-k GPW SCF (`GpwDft` with Monkhorst–Pack / explicit k), the
/// Bloch eigenvalues \(\varepsilon_{nk}\) on the sampled mesh are
/// Gaussian-broadened into a density of states. This is the GPW counterpart
/// of `cpu::pw::Dos` (and of CP2K Quickstep `&PRINT &DOS` / PDOS class on a
/// GTH-GPW multi-k SCF). Parameters are members of `Dos` — no public
/// `*Config` bag, no factories.
///
/// @code
///   cpu::gpw::GpwDft dft;
///   dft.cell[0] = dft.cell[1] = a / std::sqrt(2.0); // BCT Si [Bohr]
///   dft.cell[2] = a;
///   dft.cutoff_ry = 150.0;
///   dft.n_occ = 8;
///   dft.xc = cpu::gpw::XcKind::PBE;
///   dft.build(atoms);
///
///   cpu::gpw::Dos dos;
///   dos.k_mesh = 2;
///   dos.sigma_ha = 0.025;   // ~0.68 eV
///   auto dr = dos.compute(dft, atoms);
///   // dr.dos, dr.n_elec_to_vbm, dr.gap_ha, dr.scf_energy_ha
/// @endcode
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h"

namespace ATV {
namespace cpu {
namespace gpw {

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
/// integral is to SCF \(E_F\).
///
/// **Scope (this product):** closed-shell RKS multi-k. UKS multi-k is
/// accepted when `method.is_uks()` and fills per-spin channels (g=1).
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
/// Pure helper for unit tests and the DOS post-processor. Eigenvalues must
/// be sorted ascending per k-point. Units are those of the input (Ha).
///
/// @param evals  `[ik][ib]` eigenvalues (sorted ascending in `ib`).
/// @param n_occ  Number of occupied bands (\(n_e/2\) for closed shell).
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
