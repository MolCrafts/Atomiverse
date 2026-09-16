/// @file cpu/vibration/phonon_dispersion.h
/// @brief Fourier phonon dispersion \(\omega(\mathbf{q})\) from supercell IFCs.
///
/// Takes the Cartesian force-constant matrix of an \(n_a\times n_b\times n_c\)
/// supercell (as produced by `cpu::vib::Phonon`, atom order =
/// `expand_supercell`) and builds the mass-weighted dynamical matrix of the
/// **unit cell** at arbitrary fractional \(\mathbf{q}\):
///
/// \[
///   D_{\alpha\beta}(\kappa\kappa';\mathbf{q})
///     = \frac{1}{\sqrt{m_\kappa m_{\kappa'}}}
///       \sum_{\ell'}
///       \Phi_{\alpha\beta}(0\kappa,\,\ell'\kappa')
///       \,e^{i\mathbf{q}\cdot\mathbf{R}_{\ell'}} .
/// \]
///
/// \(\Phi\) is converted eV/Å² → Hartree/Bohr² with the same NIST factors as
/// `MassWeighting` / `PhononDiagonalizer`; eigenvalues become wavenumbers
/// [cm⁻¹]. Parameters are members — no public `*Config` bag.
///
/// @code
///   cpu::vib::Phonon phonon(/*na=*/2, /*nb=*/2, /*nc=*/2);
///   phonon.set_neighbor(...);
///   phonon.on(ATV::vib::Phase::force).add(cpu::StillingerWeber{pot});
///   phonon.build();
///   auto pr = phonon.run(unit);
///
///   cpu::vib::PhononDispersion path;
///   path.set_diamond_path(3);
///   auto r = path.compute(pr.hessian, unit, pr.na, pr.nb, pr.nc);
///   // r.frequencies_cm[iq][ibranch]  [cm^-1]
/// @endcode
#pragma once

#include <array>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Phonon dispersion spectrum on a list of fractional q-points.
///
/// Frame purity: \(\omega(\mathbf{q})\) lives here, never on Frame.
struct PhononDispersionResult {
    /// @brief Fractional q-points (crystal coordinates of the unit cell).
    std::vector<std::array<double, 3>> qfrac;

    /// @brief Wavenumbers [cm⁻¹], layout `[iq][ibranch]`, each q ascending.
    ///        Imaginary modes (\(\lambda < 0\)) are reported negative.
    std::vector<std::vector<double>> frequencies_cm;

    /// @brief Atoms in the unit cell (\(N\)); number of branches is \(3N\).
    int n_atoms_unit = 0;

    /// @brief Branches per q-point (\(= 3 \times n_\mathrm{atoms\_unit}\)).
    int n_branches = 0;

    /// @brief Supercell factors used to build the IFCs.
    int na = 1;
    int nb = 1;
    int nc = 1;
};

/// @brief Fourier \(\omega(\mathbf{q})\) from a supercell Cartesian Hessian.
///
/// `compute` does **not** run forces — the caller supplies Φ from
/// `PhononResult::hessian` (or any equivalent matrix in the same atom
/// order). Empty `qfrac` at call time fills the diamond path
/// (`set_diamond_path(4)`).
class PhononDispersion {
  public:
    /// @brief Fractional q-list. Empty → `set_diamond_path(4)` inside `compute`.
    std::vector<std::array<double, 3>> qfrac;

    /// @brief FCC/diamond high-symmetry path Γ–X–W–L–Γ–K.
    ///
    /// Waypoints match `cpu::pw::Bands::set_diamond_path` (fractional of the
    /// primitive reciprocal lattice).
    ///
    /// @param n_seg Samples per segment (including shared endpoints once).
    void set_diamond_path(int n_seg = 4);

    /// @brief \(\omega(\mathbf{q})\) for every point in `qfrac`.
    ///
    /// @param hessian_sc Cartesian force constants [eV/Å²], shape
    ///                   `[3 N_sc, 3 N_sc]`, \(N_\mathrm{sc}=N\cdot n_a n_b n_c\).
    /// @param unit       Unit-cell Frame (atomic numbers via `kAtom_type`,
    ///                   lattice via `box`; not mutated).
    /// @param na         Supercell repeats along \(\mathbf{a}\) (≥1).
    /// @param nb         Supercell repeats along \(\mathbf{b}\) (≥1).
    /// @param nc         Supercell repeats along \(\mathbf{c}\) (≥1).
    /// @return Owned spectrum; frequencies [cm⁻¹].
    /// @throws std::invalid_argument on shape / factor / atom-count mismatch.
    PhononDispersionResult compute(const xt::xtensor<double, 2>& hessian_sc,
                                   const cpu::Frame& unit, int na, int nb,
                                   int nc) const;

    /// @brief Frequencies [cm⁻¹] at a single fractional q (ascending).
    ///
    /// @param hessian_sc Cartesian Φ [eV/Å²], shape `[3 N_sc, 3 N_sc]`.
    /// @param unit       Unit-cell Frame.
    /// @param na,nb,nc   Supercell factors of `hessian_sc`.
    /// @param q_frac     Fractional wavevector.
    /// @return Length-\(3N\) wavenumbers [cm⁻¹], ascending.
    std::vector<double>
    frequencies_at_q(const xt::xtensor<double, 2>& hessian_sc,
                     const cpu::Frame& unit, int na, int nb, int nc,
                     const std::array<double, 3>& q_frac) const;
};

} // namespace vib
} // namespace cpu
} // namespace ATV
