/// @file cpu/longrange/multipole_expansion.h
/// @brief Multipole and local expansions for open-boundary Coulomb (FMM).
///
/// Backend-agnostic operator math (no Frame / Component / MPI). Built on
/// factorial-scaled complex solid harmonics in the White & Head-Gordon
/// convention (J. Chem. Phys. **101**, 6593 (1994),
/// DOI: 10.1063/1.468354), so the M2M/L2L addition theorems are exact within
/// the truncation order. Real solid-harmonic Cartesian forms from
/// `core/real_harmonics.h` are available as an independent angular basis; the
/// FMM operators use the factorial-scaled complex form for translation.
///
/// Coulomb kernel is **dimensionless** Σ q_i q_j / r_ij. Conversion e²/Å → eV
/// (`core/units.h`) is applied by the HF/CFMM wiring (chain link 21), not here.
/// Coefficients and energies use AccReal (always double).
#pragma once

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/vec3.h"

namespace ATV {
namespace longrange {

/// @brief Flat index for complex solid-harmonic coefficient (n, m).
///
/// Layout: for n = 0..p, m = −n..n, index = n² + n + m. Length = (p+1)².
/// @param n Degree (n ≥ 0).
/// @param m Order (|m| ≤ n).
/// @return Flat index in a length-(p+1)² coefficient array.
inline int solid_index(int n, int m) { return n * n + n + m; }

/// @brief Number of complex solid-harmonic coefficients for truncation order p.
/// @param order Truncation order p ≥ 0.
/// @return (p+1)².
inline int solid_ncoef(int order) { return (order + 1) * (order + 1); }

/// @brief Factorial-scaled complex regular solid harmonic R_n^m(r).
///
/// R_n^m = r^n / (n+|m|)! · P_n^{|m|}(cos θ) · e^{i m φ}, with Ferrers
/// associated Legendre P (no Condon–Shortley (−1)^m on P_m^m) and the negative-m
/// extension R_n^{−m} = (−1)^m conj(R_n^m) · (n−m)!/(n+m)! adjusted so the
/// addition theorem is a plain product sum (White & Head-Gordon 1994).
///
/// @param n Degree (n ≥ 0).
/// @param m Order (|m| ≤ n).
/// @param r Position relative to expansion centre [length units of the kernel].
/// @return R_n^m(r) [dimensionless for the FMM kernel convention].
std::complex<AccReal> regular_solid(int n, int m, const Vec3& r);

/// @brief Factorial-scaled complex irregular solid harmonic I_n^m(r).
///
/// I_n^m = (n−|m|)! · r^{−(n+1)} · P_n^{|m|}(cos θ) · e^{i m φ}, with the
/// same P convention as #regular_solid. For |r|→0 the result is undefined
/// (caller must keep evaluation points outside the source ball).
///
/// @param n Degree (n ≥ 0).
/// @param m Order (|m| ≤ n).
/// @param r Position relative to expansion centre (r ≠ 0).
/// @return I_n^m(r).
std::complex<AccReal> irregular_solid(int n, int m, const Vec3& r);

/// @brief Outer multipole expansion of a charge cluster about a centre.
///
/// Coefficients M_n^m satisfy
///   Φ(r) = Σ_{n,m} M_n^m · I_n^m(r − centre)
/// for |r − centre| larger than the cluster radius. Built by #accumulate (P2M)
/// and/or M2M translation; evaluated by #evaluate (M2P).
class MultipoleExpansion {
  public:
    /// @param order  Truncation order p ≥ 0 (coefficients through degree p).
    /// @param center Expansion centre [same length unit as positions].
    /// @throws std::invalid_argument if order < 0.
    MultipoleExpansion(int order, Vec3 center);

    /// @brief Truncation order p.
    int order() const { return order_; }

    /// @brief Expansion centre.
    const Vec3& center() const { return center_; }

    /// @brief Mutable centre (tree rebuild / M2M destination).
    Vec3& center() { return center_; }

    /// @brief Coefficient array length (p+1)².
    int ncoef() const { return solid_ncoef(order_); }

    /// @brief Complex multipole coefficients M_n^m (layout #solid_index).
    const std::vector<std::complex<AccReal>>& coeffs() const { return c_; }
    std::vector<std::complex<AccReal>>& coeffs() { return c_; }

    /// @brief Zero all coefficients (keep centre and order).
    void clear();

    /// @brief P2M: add one point charge to this multipole.
    ///
    /// M_n^m += q · R_n^{−m}(pos − centre)  (conjugate-index convention so
    /// the Coulomb kernel is real for real charges).
    ///
    /// @param q   Charge [dimensionless kernel charge].
    /// @param pos Absolute position [length].
    void accumulate(AccReal q, const Vec3& pos);

    /// @brief M2P: evaluate the multipole potential at an absolute point.
    ///
    /// Φ = Σ_{n,m} M_n^m · I_n^m(point − centre). Requires |point − centre| > 0.
    ///
    /// @param point Absolute evaluation point [length].
    /// @return Potential Φ [dimensionless; multiply by units::coulomb for eV].
    AccReal evaluate(const Vec3& point) const;

  private:
    int order_;
    Vec3 center_;
    std::vector<std::complex<AccReal>> c_;
};

/// @brief Local (Taylor) expansion of a smooth potential about a centre.
///
/// Coefficients L_n^m satisfy
///   Φ(r) = Σ_{n,m} L_n^m · R_n^m(r − centre)
/// inside a ball free of sources. Filled by M2L/L2L; evaluated by #evaluate
/// (L2P).
class LocalExpansion {
  public:
    /// @param order  Truncation order p ≥ 0.
    /// @param center Expansion centre [length].
    /// @throws std::invalid_argument if order < 0.
    LocalExpansion(int order, Vec3 center);

    int order() const { return order_; }
    const Vec3& center() const { return center_; }
    Vec3& center() { return center_; }
    int ncoef() const { return solid_ncoef(order_); }

    const std::vector<std::complex<AccReal>>& coeffs() const { return c_; }
    std::vector<std::complex<AccReal>>& coeffs() { return c_; }

    /// @brief Zero all coefficients.
    void clear();

    /// @brief L2P: evaluate the local expansion at an absolute point.
    ///
    /// Φ = Σ_{n,m} L_n^m · R_n^m(point − centre).
    ///
    /// @param point Absolute evaluation point [length].
    /// @return Potential Φ [dimensionless].
    AccReal evaluate(const Vec3& point) const;

  private:
    int order_;
    Vec3 center_;
    std::vector<std::complex<AccReal>> c_;
};

} // namespace longrange
} // namespace ATV
