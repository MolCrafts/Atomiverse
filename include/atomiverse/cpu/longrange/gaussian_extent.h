/// @file cpu/longrange/gaussian_extent.h
/// @brief Extent classification for continuous Gaussian charge distributions (CFMM).
///
/// Backend-agnostic (no Frame / Component / MPI). A product Gaussian with total
/// exponent ζ is assigned a finite extent
///   r_ext = √(2/ζ) · erfc⁻¹(ε_thr)   [Bohr]
/// so that the neglected charge outside the ball of radius r_ext is bounded by
/// the multipole threshold ε_thr. Two distributions are **well-separated**
/// (eligible for multipole interaction) only when
///   |R_A − R_B| > r_ext(A) + r_ext(B).
///
/// Reference: White, Johnson, Gill, Head-Gordon, *Chem. Phys. Lett.* **230**, 8
/// (1994), DOI: 10.1016/0009-2614(94)01128-1. Consumed by HF CFMM-J (chain link
/// 21); does not modify the FMM tree of link 20.
#pragma once

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/vec3.h"

namespace ATV {
namespace longrange {

namespace detail {

/// @brief Inverse complementary error function erfc⁻¹(y) for y ∈ (0, 2).
///
/// Rational starter + Halley refinement of erfc. Accurate to ~1e-14 relative
/// on (1e-16, 1.9) — enough for extent thresholds used in CFMM (ε ≥ 1e-14).
inline AccReal erfc_inv(AccReal y) {
    if (!(y > AccReal(0) && y < AccReal(2)))
        throw std::invalid_argument("erfc_inv: argument must lie in (0, 2)");
    if (y == AccReal(1))
        return AccReal(0);

    // Map to erf⁻¹ domain: erfc⁻¹(y) = erf⁻¹(1 − y).
    const AccReal z = AccReal(1) - y;
    // Beasley–Springer–Moro style seed for |z| not near 1, else asymptotic.
    AccReal x;
    const AccReal az = std::fabs(z);
    if (az <= AccReal(0.7)) {
        // erf⁻¹ rational approx (Winitzki-like).
        constexpr AccReal kPi = AccReal(3.14159265358979323846);
        const AccReal a = AccReal(0.147);
        const AccReal t =
            AccReal(2) / (kPi * a) + std::log(AccReal(1) - z * z) / AccReal(2);
        const AccReal u = std::log(AccReal(1) - z * z) / a;
        x = (z > 0 ? AccReal(1) : AccReal(-1))
            * std::sqrt(std::sqrt(t * t - u) - t);
    } else {
        // Asymptotic: erfc(x) ~ exp(-x²)/(x√π) ⇒ x² ≈ -log(y · x · √π)
        constexpr AccReal kPi = AccReal(3.14159265358979323846);
        AccReal yy = (y < AccReal(1)) ? y : (AccReal(2) - y);
        AccReal w = -std::log(yy);
        x = std::sqrt(w);
        for (int k = 0; k < 4; ++k)
            x = std::sqrt(w - std::log(x * std::sqrt(kPi)));
        if (y > AccReal(1))
            x = -x;
    }

    // Halley refinement on f(x) = erfc(x) - y.
    constexpr AccReal kPi = AccReal(3.14159265358979323846);
    for (int it = 0; it < 6; ++it) {
        const AccReal err = std::erfc(x) - y;
        // d(erfc)/dx = -2/√π exp(-x²), d²(erfc)/dx² = 4x/√π exp(-x²)
        const AccReal e = std::exp(-x * x);
        const AccReal d1 = -AccReal(2) / std::sqrt(kPi) * e;
        const AccReal d2 = AccReal(4) * x / std::sqrt(kPi) * e;
        const AccReal step = err / (d1 - err * d2 / (AccReal(2) * d1));
        x -= step;
        if (std::fabs(step) < AccReal(1e-16) * (AccReal(1) + std::fabs(x)))
            break;
    }
    return x;
}

} // namespace detail

/// @brief Extent of a product Gaussian with total exponent ζ [Bohr⁻²].
///
/// Value type: constructed with (ζ, ε_thr), immediately computes r_ext.
/// Formula (WJGH 1994): r_ext = √(2/ζ) · erfc⁻¹(ε_thr) [Bohr].
class GaussianExtent {
  public:
    /// @brief Construct and evaluate the extent.
    /// @param zeta      Total product exponent ζ = ζ_a + ζ_b [Bohr⁻²]; must be > 0.
    /// @param threshold Multipole penetration threshold ε_thr ∈ (0, 1]
    ///                  (dimensionless charge fraction outside the ball).
    /// @throws std::invalid_argument if zeta ≤ 0 or threshold not in (0, 1].
    GaussianExtent(AccReal zeta, AccReal threshold)
        : zeta_(zeta), threshold_(threshold) {
        if (!(zeta_ > AccReal(0)))
            throw std::invalid_argument(
                "GaussianExtent: zeta must be > 0 [Bohr^-2]");
        if (!(threshold_ > AccReal(0) && threshold_ <= AccReal(1)))
            throw std::invalid_argument(
                "GaussianExtent: threshold must lie in (0, 1]");
        // r_ext = √(2/ζ) · erfc⁻¹(ε). For ε = 1, erfc⁻¹(1) = 0 ⇒ r_ext = 0.
        r_ext_ = std::sqrt(AccReal(2) / zeta_) * detail::erfc_inv(threshold_);
    }

    /// @brief Total product exponent ζ [Bohr⁻²].
    AccReal zeta() const { return zeta_; }

    /// @brief Penetration threshold ε_thr (dimensionless).
    AccReal threshold() const { return threshold_; }

    /// @brief Extent radius r_ext [Bohr].
    AccReal r_ext() const { return r_ext_; }

    /// @brief True iff the two balls are strictly well-separated for multipoles.
    ///
    /// Well-separated ⇔ |c_a − c_b| > r_ext(a) + r_ext(b). Equality falls to
    /// the near-field (direct ERI) side so multipole penetration stays below
    /// threshold.
    ///
    /// @param a   Extent of distribution A.
    /// @param ca  Centre of A [Bohr].
    /// @param b   Extent of distribution B.
    /// @param cb  Centre of B [Bohr].
    /// @return Whether multipole (far-field) interaction is permitted.
    static bool well_separated(const GaussianExtent& a, const Vec3& ca,
                               const GaussianExtent& b, const Vec3& cb) {
        const AccReal r = (ca - cb).norm();
        return r > (a.r_ext() + b.r_ext());
    }

  private:
    AccReal zeta_;
    AccReal threshold_;
    AccReal r_ext_;
};

} // namespace longrange
} // namespace ATV
