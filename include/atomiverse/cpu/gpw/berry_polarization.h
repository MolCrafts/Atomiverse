/// @file cpu/gpw/berry_polarization.h
/// @brief Berry-phase (King–Smith–Vanderbilt) formal polarization (CPU GPW).
///
/// After multi-k SCF on strings of k-points along each reciprocal lattice
/// vector (AO-basis Bloch MOs from `GpwDft`), the electronic polarization is
/// the string-averaged Berry phase of the occupied manifold
/// \(\varphi_i = \operatorname{Im}\ln\prod_j \det M^{(j)}\) with
/// \(M^{(j)}=C^\dagger(k_j)\,S^{(k_j,k_{j+1})}\,C(k_{j+1})\) and phased AO
/// overlaps \(S\) that carry \(e^{-i\mathbf b\cdot\mathbf r}\)
/// (King-Smith & Vanderbilt, Phys. Rev. B 47, 1651 (1993); Resta,
/// Rev. Mod. Phys. 66, 899 (1994)). Closing links use the short reciprocal
/// step \(b = 2\pi/(L_i J)\) with wrapped k-labels (same convention as
/// multi-k MLWF).
///
/// **Closed shell** (`!is_uks()`): spin factor \(f=2\),
/// \(p^\mathrm{el}=f\varphi/2\pi\), total wrapped with quantum 2.
/// **Collinear spin** (`is_uks()`): each spin channel has its own occupied
/// manifold; \(p^\mathrm{el}=(\varphi_\uparrow+\varphi_\downarrow)/2\pi\),
/// total wrapped with quantum 1. Metallic smearing is rejected (insulator
/// required). Parameters are members of `BerryPolarization` — no public
/// `*Config` bag, no factories.
///
/// Product: closed-shell or collinear UKS insulator, orthorhombic cell
/// (GPW convention). Twin: CP2K Quickstep `&MOMENTS PERIODIC`.
///
/// @code
///   cpu::gpw::GpwDft dft;
///   dft.cell[0] = dft.cell[1] = a / std::sqrt(2.0); // BCT Si [Bohr]
///   dft.cell[2] = a;
///   dft.cutoff_ry = 100.0;
///   dft.n_occ = 8;
///   dft.xc = cpu::gpw::XcKind::PADE;
///   dft.build(atoms);
///
///   cpu::gpw::BerryPolarization berry;   // members: k_perp, k_string, …
///   auto r = berry.compute(dft, atoms);
///   // r.p_frac[i]  — total formal polarization along a_i [e, fractional]
///   // r.dipole_e_ang — Cartesian cell dipole P·a  [e Å]
/// @endcode
#pragma once

#include <array>
#include <complex>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Result of a Berry-phase polarization calculation on GPW.
///
/// Fractional components are along the orthorhombic lattice vectors
/// \(\mathbf a_i\), in units of the elementary charge \(e\). Closed-shell:
/// spin factor 2, `p_frac` wrapped into \((-1,1]\) (quantum 2). Collinear
/// spin: factor 1 per channel, `p_frac` wrapped into \((-0.5,0.5]\)
/// (quantum 1).
struct BerryPolarizationResult {
    /// String-averaged electronic Berry phase per direction [rad].
    /// Closed-shell: single-manifold \(\varphi\). Collinear: mean of the
    /// two spin phases \((\varphi_\uparrow+\varphi_\downarrow)/2\).
    std::array<double, 3> phase_el{};
    /// Electronic polarization [e, fractional]: \(2\varphi/2\pi\) closed-
    /// shell, \((\varphi_\uparrow+\varphi_\downarrow)/2\pi\) collinear.
    std::array<double, 3> p_el_frac{};
    /// Ionic polarization \(\sum_a Z^\mathrm{val}_a \tau_{a,i}\)
    /// (\(\tau\) wrapped into [0,1)) [e, fractional].
    std::array<double, 3> p_ion_frac{};
    /// Total formal polarization, wrapped into the quantum window [e, fractional].
    std::array<double, 3> p_frac{};
    /// Cartesian cell dipole \(\sum_i p_i\,a_i\,\hat e_i\) [e Å] for the
    /// orthorhombic cell (meaningful when all three directions are computed).
    std::array<double, 3> dipole_e_ang{};
    /// Which directions were computed (mirrors `BerryPolarization::directions`).
    std::array<bool, 3> computed{};
    /// Whether every SCF that fed a computed direction converged.
    bool scf_converged = false;
    /// Last SCF total energy [Hartree] (final direction computed).
    double energy_ha = 0.0;
};

/// @brief Discretized Berry-phase polarization on the GPW method
///        (string-mesh multi-k SCF + occupied overlap determinants).
///
/// For each requested lattice direction the method configures an explicit
/// string k-mesh (string-major layout; TR folding off), runs `GpwDft::run`,
/// and averages \(\operatorname{Im}\ln\det M\) over the \(N_\perp^2\)
/// strings. Occupied-only determinants make the phase gauge-invariant under
/// band mixing at interior string points.
class BerryPolarization {
  public:
    /// @brief Transverse mesh per direction (\(N_\perp^2\) strings,
    ///        dimensionless).
    int k_perp = 2;
    /// @brief k-points per string \(J\) (dimensionless, >= 2).
    int k_string = 4;
    /// @brief Lattice directions to compute (default: all three).
    std::array<bool, 3> directions{true, true, true};

    /// @brief Fractional k-points of all strings along direction `dir`.
    ///
    /// String-major layout: `k_perp*k_perp` blocks of `k_string`
    /// consecutive points; within a string the `dir` component is
    /// \(j/J\), transverse components \(i/N_\perp\); all in [0,1).
    ///
    /// @param dir Lattice direction in [0,2].
    /// @return `k_perp^2 * k_string` fractional (crystal) k-points.
    std::vector<std::array<double, 3>> string_kpoints(int dir) const;

    /// @brief Multi-k SCF strings + Berry phase for the geometry in `atoms`.
    ///
    /// For each requested direction: sets `method.k_points` to the string
    /// mesh (equal weights, TR off), runs `method.run(atoms)`, then forms
    /// link determinants from `last_result()` occupied Bloch MOs.
    ///
    /// @param method GPW DFT method (`build()` already done; k-mesh / list
    ///               and TR flag are overwritten per direction).
    /// @param atoms  Atom set at the current geometry [Bohr] (same as `build`).
    /// @return Owned `BerryPolarizationResult` (fractional [e], dipole [e Å]).
    /// @throws std::invalid_argument on metallic smearing, non-positive
    ///         mesh, empty occupied manifold, or non-periodic cell.
    /// @throws std::logic_error when `method` was not `build()`'d.
    BerryPolarizationResult compute(GpwDft& method,
                                    const std::vector<GpwAtom>& atoms) const;
};

/// @brief \(\operatorname{Im}\ln\prod_j d_j\) of a closed string of link
///        determinants (pure math; unit-testable without SCF).
///
/// Each determinant is normalized to its modulus before the product, so
/// the result is the phase \(\in(-\pi,\pi]\) of the unimodular product.
///
/// @param dets Link determinants of one closed string (non-empty).
/// @return Berry phase of the string [rad] in \((-\pi,\pi]\).
/// @throws std::invalid_argument on an empty string or a link determinant
///         with modulus <= 1e-12 (metallic / broken overlap).
double phase_from_link_dets(const std::vector<std::complex<double>>& dets);

/// @brief Wrap `p` into the centered interval \((-q/2,\, q/2]\).
/// @param p       Value to wrap (any unit).
/// @param quantum Period \(q\) (same unit, must be > 0).
/// @return Wrapped value in \((-q/2, q/2]\).
double wrap_centered(double p, double quantum);

} // namespace gpw
} // namespace cpu
} // namespace ATV
