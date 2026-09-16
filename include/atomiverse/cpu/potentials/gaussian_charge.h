/// @file cpu/potentials/gaussian_charge.h
/// @brief Gaussian electrode charge kernels (MetalWalls / Reed form) — pure math.
///
/// Normalized 3D Gaussian atomic charges
/// \f[
///   \rho_i(\mathbf{r}) = q_i \frac{\eta_i^3}{\pi^{3/2}}
///     e^{-\eta_i^2 |\mathbf{r}-\mathbf{R}_i|^2}
/// \f]
/// interact via the closed-form Green function
/// \f[
///   G_{ij}(r) = \frac{\mathrm{erf}(\eta_{ij} r)}{r},\quad
///   \eta_{ij} = \frac{\eta_i\eta_j}{\sqrt{\eta_i^2+\eta_j^2}},
/// \f]
/// with self term \f$G_{ii} = \eta_i\sqrt{2/\pi}\f$ (Reed, Lanning, Madden,
/// *J. Chem. Phys.* **126**, 084704 (2007); MetalWalls JOSS 2020).
///
/// Public surface units: length [Å], \f$\eta\f$ [Å⁻¹], charge [e].
/// Energy prefactor is caller-supplied (`units::coulomb` for eV).
///
/// Header-only pure math — no Frame, no Component. Consumed by
/// `PairGaussianCoul` and (later) constant-potential BO private \(A\).
#pragma once

#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {

/// @brief Closed-form Coulomb kernels for Gaussian (and point) atomic charges.
///
/// Value type with static methods only — no mutable state. \(\eta\le 0\) on an
/// atom means a classical **point** charge (limit \(\eta\to\infty\)).
class GaussianChargeKernel {
  public:
    /// @brief Effective pair width \(\eta_{ij}\) [Å⁻¹].
    ///
    /// @param eta_i Width of site \(i\) [Å⁻¹]; \(\le 0\) ⇒ point.
    /// @param eta_j Width of site \(j\) [Å⁻¹]; \(\le 0\) ⇒ point.
    /// @return \(\eta_{ij}\); \(+\infty\) (as a large sentinel path) is not
    ///         returned — use `is_point_pair` / `greens` instead.
    /// @throws std::invalid_argument if both are point (no finite \(\eta_{ij}\)).
    static double eta_eff(double eta_i, double eta_j) {
        const bool pi = eta_i <= 0.0;
        const bool pj = eta_j <= 0.0;
        if (pi && pj)
            throw std::invalid_argument(
                "GaussianChargeKernel::eta_eff: both sites are point charges");
        if (pi)
            return eta_j;
        if (pj)
            return eta_i;
        return eta_i * eta_j / std::sqrt(eta_i * eta_i + eta_j * eta_j);
    }

    /// @brief True if both sites are point charges.
    static bool both_point(double eta_i, double eta_j) {
        return eta_i <= 0.0 && eta_j <= 0.0;
    }

    /// @brief True if site is a Gaussian (finite \(\eta>0\)).
    static bool is_gaussian(double eta) { return eta > 0.0; }

    /// @brief Pair Green function \(G_{ij}(r)\) [Å⁻¹].
    ///
    /// Point–point: \(1/r\). Gaussian–any: \(\mathrm{erf}(\eta_{\mathrm{eff}} r)/r\).
    ///
    /// @param r     Separation [Å]; must be > 0.
    /// @param eta_i Width of \(i\) [Å⁻¹]; \(\le 0\) ⇒ point.
    /// @param eta_j Width of \(j\) [Å⁻¹]; \(\le 0\) ⇒ point.
    /// @return \(G_{ij}(r)\) [1/Å].
    /// @throws std::invalid_argument if \(r\le 0\).
    static double greens(double r, double eta_i, double eta_j) {
        if (!(r > 0.0))
            throw std::invalid_argument(
                "GaussianChargeKernel::greens: r must be > 0 [Å]");
        if (both_point(eta_i, eta_j))
            return 1.0 / r;
        const double e = eta_eff(eta_i, eta_j);
        return std::erf(e * r) / r;
    }

    /// @brief Self Green function \(G_{ii}=\eta\sqrt{2/\pi}\) [Å⁻¹].
    ///
    /// Equals \(\lim_{r\to 0}\mathrm{erf}(\eta_{\mathrm{eff}} r)/r\) with
    /// \(\eta_{\mathrm{eff}}=\eta/\sqrt{2}\). Point charges have no finite self
    /// term — throws if \(\eta\le 0\).
    ///
    /// @param eta Width [Å⁻¹]; must be > 0.
    /// @return \(G_{ii}\) [1/Å].
    static double self_greens(double eta) {
        if (!(eta > 0.0))
            throw std::invalid_argument(
                "GaussianChargeKernel::self_greens: eta must be > 0 [Å^-1]");
        constexpr double kSqrt2OverPi = 0.79788456080286535588; // sqrt(2/pi)
        return eta * kSqrt2OverPi;
    }

    /// @brief Radial derivative \(dG/dr\) [Å⁻²] of the pair Green function.
    ///
    /// Point–point: \(-1/r^2\).
    /// Gaussian:
    /// \f[
    ///   \frac{dG}{dr}
    ///   = \frac{2\eta_{\mathrm{eff}}}{\sqrt{\pi}}
    ///     \frac{e^{-(\eta_{\mathrm{eff}} r)^2}}{r}
    ///     - \frac{\mathrm{erf}(\eta_{\mathrm{eff}} r)}{r^2}.
    /// \f]
    ///
    /// @param r     Separation [Å]; must be > 0.
    /// @param eta_i Width of \(i\) [Å⁻¹].
    /// @param eta_j Width of \(j\) [Å⁻¹].
    /// @return \(dG/dr\) [1/Å²].
    static double dgreens_dr(double r, double eta_i, double eta_j) {
        if (!(r > 0.0))
            throw std::invalid_argument(
                "GaussianChargeKernel::dgreens_dr: r must be > 0 [Å]");
        if (both_point(eta_i, eta_j))
            return -1.0 / (r * r);
        const double e = eta_eff(eta_i, eta_j);
        constexpr double kTwoOverSqrtPi = 1.12837916709551257390; // 2/sqrt(pi)
        const double er = e * r;
        const double inv_r = 1.0 / r;
        return kTwoOverSqrtPi * e * std::exp(-er * er) * inv_r
               - std::erf(er) * inv_r * inv_r;
    }

    /// @brief Pair Coulomb energy contribution [eV] (no self).
    ///
    /// \(U_{ij} = \texttt{coulomb}\, q_i q_j\, G_{ij}(r)\).
    ///
    /// @param coulomb Prefactor [eV·Å/e²] (`units::coulomb` for ATV metal units).
    /// @param q_i Charge on \(i\) [e].
    /// @param q_j Charge on \(j\) [e].
    /// @param r Separation [Å].
    /// @param eta_i Width of \(i\) [Å⁻¹].
    /// @param eta_j Width of \(j\) [Å⁻¹].
    /// @return Pair energy [eV].
    static double pair_energy(double coulomb, double q_i, double q_j, double r,
                              double eta_i, double eta_j) {
        return coulomb * q_i * q_j * greens(r, eta_i, eta_j);
    }

    /// @brief Gaussian self-energy \(\frac12\,\texttt{coulomb}\,q^2 G_{ii}\) [eV].
    ///
    /// @param coulomb Prefactor [eV·Å/e²].
    /// @param q Charge [e].
    /// @param eta Width [Å⁻¹]; must be > 0.
    /// @return Self energy [eV].
    static double self_energy(double coulomb, double q, double eta) {
        return 0.5 * coulomb * q * q * self_greens(eta);
    }
};

} // namespace cpu
} // namespace ATV
