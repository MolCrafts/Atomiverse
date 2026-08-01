/// @file cpu/gpw/bands_dos.h
/// @brief Band-structure post-processing on frozen GPW \(V_\mathrm{eff}\).
///
/// After multi-k closed-shell SCF on `cpu::gpw::GpwDft`, non-self-consistent
/// diagonalization of \(H_k = T_k + V_\mathrm{nl}^k + V^k[V_\mathrm{eff}]\) on
/// a high-symmetry fractional path yields \(\varepsilon_n(\mathbf{k})\)
/// (Hartree). Parameters are members of `Bands` — no public `*Config`, no
/// factories. Twin class: CP2K Quickstep `&PRINT &BAND_STRUCTURE` on GTH-GPW.
///
/// @code
///   cpu::gpw::GpwDft dft;
///   dft.cell[0] = dft.cell[1] = a / std::sqrt(2.0);
///   dft.cell[2] = a;
///   dft.cutoff_ry = 150.0;
///   dft.n_occ = 8;
///   dft.k_mesh[0] = dft.k_mesh[1] = dft.k_mesh[2] = 2;
///   dft.xc = cpu::gpw::XcKind::PBE;
///   dft.build(atoms);
///
///   cpu::gpw::Bands bands;
///   bands.n_bands = 12;
///   bands.set_diamond_path(4);
///   auto br = bands.compute(dft, atoms);  // gap_ha, eigenvalues_ha
/// @endcode
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
    /// Restricted closed-shell channel only (product of this slug).
    std::vector<std::vector<double>> eigenvalues_ha;
    /// 1 = restricted (closed-shell). Spin product is a follow-on.
    int nspin = 1;
    double e_vbm_ha = 0;   ///< Valence-band maximum [Hartree].
    double e_cbm_ha = 0;   ///< Conduction-band minimum [Hartree].
    double gap_ha = 0;     ///< \(E_\mathrm{cbm}-E_\mathrm{vbm}\) [Hartree].
    int vbm_k = 0;         ///< k-index of VBM on the path.
    int cbm_k = 0;         ///< k-index of CBM on the path.
    int vbm_band = 0;      ///< Band index of VBM (0-based).
    int cbm_band = 0;      ///< Band index of CBM (0-based).
    bool scf_converged = false; ///< Whether the preceding SCF converged.
    double energy_ha = 0;  ///< SCF total energy [Hartree].
};

/// @brief High-symmetry-path band structure (NSCF on frozen GPW \(V_\mathrm{eff}\)).
///
/// SCF uses `method` members (`cutoff_ry`, `k_mesh` / `k_points`, `xc`,
/// `n_occ`, …). Path k-points are `kfrac` (filled by `set_diamond_path` when
/// empty at `compute` time). Product path: closed-shell RKS, fixed integer
/// occupations, single-grid periodic orthorhombic with local PP on the grid.
///
/// Collinear spin, metallic smear, multigrid, nonperiodic, DFT+U, and
/// dispersion flags throw with a how-to (no silent degradation).
class Bands {
  public:
    /// @brief Number of bands to report per path k (lowest eigenvalues).
    ///        Must be \(\ge n_\mathrm{occ}\) so the gap is defined. GPW
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
    /// @return Owned `BandsResult` with eigenvalues [Hartree] and gap.
    /// @throws std::logic_error when `method` was not `build()`'d.
    /// @throws std::invalid_argument on UKS / smear / multigrid / nonperiodic /
    ///         analytic local PP / DFT+U / dispersion / empty path / n_bands.
    BandsResult compute(GpwDft& method, const std::vector<GpwAtom>& atoms) const;
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

} // namespace gpw
} // namespace cpu
} // namespace ATV
