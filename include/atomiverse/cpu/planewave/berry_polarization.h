/// @file cpu/planewave/berry_polarization.h
/// @brief Berry-phase (King–Smith–Vanderbilt) formal polarization (CPU PW).
///
/// After a self-consistent density from `cpu::pw::PW` members, occupied
/// Bloch states are diagonalized non-self-consistently on strings of
/// k-points along each reciprocal lattice vector and the electronic
/// polarization is the string-averaged Berry phase
/// \(\varphi_i = \operatorname{Im}\ln\prod_j \det S^{(j)}\)
/// (King-Smith & Vanderbilt, Phys. Rev. B 47, 1651 (1993); Resta,
/// Rev. Mod. Phys. 66, 899 (1994)).
///
/// **Closed shell** (`spin_polarized == false`): spin factor \(f=2\),
/// \(p^\mathrm{el}=f\varphi/2\pi\), total wrapped with quantum 2.
/// **Collinear spin** (`spin_polarized == true`): each spin channel has
/// its own occupied manifold; \(p^\mathrm{el}=(\varphi_\uparrow+
/// \varphi_\downarrow)/2\pi\) (factor 1 per channel), total wrapped with
/// quantum 1. Metallic smearing is rejected (insulator required).
/// Parameters are members of `BerryPolarization` — no public `*Config`
/// bag, no factories.
///
/// @code
///   cpu::pw::PW pw;
///   pw.upf_path = "Si.upf";
///   pw.ecut_ry = 15.0;
///   pw.k_mesh = 2;
///   pw.xc = cpu::scf::XCType::LSDA;
///   // optional collinear: pw.spin_polarized = true; pw.multiplicity = 1;
///
///   cpu::pw::BerryPolarization berry;   // members: k_perp, k_string
///   auto r = berry.compute(pw, frame);
///   // r.p_frac[i]  — total formal polarization along a_i [e, fractional]
///   // r.dipole_e_ang — Cartesian cell dipole P*V [e Å]
/// @endcode
#pragma once

#include <array>
#include <complex>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {

/// @brief Result of a Berry-phase polarization calculation.
///
/// Fractional components are along the lattice vectors \(\mathbf a_i\), in
/// units of the elementary charge \(e\). Closed-shell: spin factor 2,
/// `p_frac` wrapped into \((-1,1]\) (quantum 2). Collinear spin: factor 1
/// per channel, `p_frac` wrapped into \((-0.5,0.5]\) (quantum 1).
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
    /// Total formal polarization, wrapped into \((-1, 1]\) [e, fractional].
    std::array<double, 3> p_frac{};
    /// Cartesian cell dipole \(\sum_i p_i\,\mathbf a_i\) [e Å]; sums the
    /// computed directions only (meaningful when all three are computed).
    std::array<double, 3> dipole_e_ang{};
    /// Which directions were computed (mirrors
    /// `BerryPolarization::directions`).
    std::array<bool, 3> computed{};
    bool scf_converged = false; ///< Whether the preceding SCF converged.
    double energy_ry = 0;       ///< SCF total energy [Ry].
};

/// @brief Discretized Berry-phase polarization on the PW method
///        (NSCF strings on the frozen SCF \(V_\mathrm{eff}\)).
///
/// Occupied-only determinants make the phase gauge-invariant under any
/// band mixing at interior string points; the closing link
/// \(k_{J-1}\to k_0+\mathbf b_i\) uses the periodic-gauge coefficient
/// shift \(c_{n,k_0+\mathbf b}(\mathbf G)=c_{n,k_0}(\mathbf G+\mathbf b)\).
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

    /// @brief SCF + Berry-phase strings for the geometry in `frame`.
    /// @param method Plane-wave method (members only; not mutated).
    /// @param frame  Atom positions [Å], types (Z), simulation cell.
    /// @return Owned `BerryPolarizationResult` (fractional [e], dipole [e Å]).
    BerryPolarizationResult compute(const PW& method,
                                    const cpu::Frame& frame) const;
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

} // namespace pw
} // namespace cpu
} // namespace ATV
