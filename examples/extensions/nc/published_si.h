#pragma once

#include "atomiverse/electronic/norm_conserving_potential.h"
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

namespace nc_example {

// Krack, Theor. Chem. Acc.114 (2005)145, Si GTH-PBE-q4.
// Values transcribed from CP2K GTH_POTENTIALS, not a built-in ATV provider.
// This class owns immutable physical radial data. There is no execution state,
// parser, Fourier transform, electronic loop, or per-point heap allocation.
class PublishedSi final : public ATV::electronic::NormConservingPotential {
  public:
    using Real = ATV::AccReal;
    static constexpr Real rloc = .44;
    static constexpr Real c1 = -6.26928833;
    static constexpr std::size_t points = 4097;
    static constexpr std::size_t projector_points = 2049;
    static constexpr Real spacing = Real(12) / Real(points - 1);
    static constexpr std::array<Real, 9> coupling_ha = {
        8.95174150, -2.70627082, 0, -2.70627082, 3.49378060,
        0,          0,           0, 2.43127673};

    PublishedSi() : radius_(points), jacobian_(points, spacing) {
        for (auto& beta : beta_)
            beta.resize(projector_points);
        for (std::size_t j = 0; j < points; ++j) {
            const Real r = spacing * Real(j);
            radius_[j] = r;
            if (j >= projector_points)
                continue;
            for (int p = 0; p < 3; ++p) {
                const int l = p == 2 ? 1 : 0;
                const int i = p == 1 ? 2 : 1; // GTH's one-based radial index.
                const Real a = p == 2 ? Real(.49794218) : Real(.43563383);
                const Real nu = Real(l) + Real(4 * i - 1) / Real(2);
                const Real norm =
                    std::sqrt(Real(2))
                    / (std::pow(a, nu) * std::sqrt(std::tgamma(nu)));
                // Public beta=r*p_l. The normalized radial p_l has dimension
                // Bohr^-3/2; D is in Ha, with no automatic normalization.
                beta_[p][j] = norm * std::pow(r, l + 2 * i - 1)
                              * std::exp(-r * r / (2 * a * a));
            }
        }
        for (int p = 0; p < 3; ++p)
            views_[p] = {p == 2 ? 1 : 0, beta_[p]};
    }
    PublishedSi(const PublishedSi&) = delete; // Borrowed spans stay with owner.
    PublishedSi& operator=(const PublishedSi&) = delete;
    PublishedSi(PublishedSi&&) = delete;
    PublishedSi& operator=(PublishedSi&&) = delete;

    int atomic_number() const noexcept override { return 14; }
    int valence() const noexcept override { return 4; }
    ATV::electronic::NcProjectorSetView projectors() const override {
        return {radius_, jacobian_, views_, coupling_ha, {}, {}};
    }

  protected:
    void evaluate_local(std::span<const Real> radius, std::span<Real> value,
                        std::span<Real> derivative) const override {
        const Real root = std::sqrt(Real(2) / std::numbers::pi_v<Real>);
        for (std::size_t i = 0; i < radius.size(); ++i) {
            const Real r = radius[i], x = r / rloc, x2 = x * x;
            const Real gaussian = std::exp(-x2 / 2);
            Real coulomb, slope;
            if (x < Real(1e-3)) {
                // Taylor limits of erf(x/sqrt2)/r and its derivative;
                // both remain finite and accurate at the smooth origin.
                coulomb = -4 * root / rloc
                          * (1 - x2 / 6 + x2 * x2 / 40 - x2 * x2 * x2 / 336);
                slope = 4 * root / (rloc * rloc) * x
                        * (Real(1) / 3 - x2 / 10 + x2 * x2 / 56);
            } else {
                const Real erf = std::erf(x / std::sqrt(Real(2)));
                coulomb = -4 * erf / r;
                slope = 4 * (erf / (r * r) - root * gaussian / (rloc * r));
            }
            value[i] = coulomb + c1 * gaussian;
            derivative[i] = slope - c1 * x / rloc * gaussian;
        }
    }

  private:
    std::vector<Real> radius_, jacobian_;
    std::array<std::vector<Real>, 3> beta_;
    std::array<ATV::electronic::NcProjectorView, 3> views_;
};
} // namespace nc_example
