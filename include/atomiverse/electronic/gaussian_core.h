/// Backend-neutral GPW Gaussian-core short-range correction. Atomic units.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ATV::gaussian_core {
struct Site {
    double position[3]; // Bohr; arbitrary unwrapped coordinates allowed.
    double charge;      // GTH valence charge, not atomic number.
    double sigma;       // Gaussian standard deviation = GTH rloc, Bohr.
};
struct Pair {
    int atom_a, atom_b;
    double displacement[3]; // R_a - R_b - lattice image, Bohr.
    double charge_product;
    double gamma; // 1 / sqrt(2*(sigma_a^2+sigma_b^2)), inverse Bohr.
};
struct PairValue {
    double energy;       // Hartree.
    double force_over_r; // Ha/Bohr^2; F_a = this * displacement is Ha/Bohr.
};
#if defined(__CUDACC__)
__host__ __device__
#endif
    inline PairValue evaluate_pair(const Pair& pair) {
    const double r2 = pair.displacement[0] * pair.displacement[0]
                      + pair.displacement[1] * pair.displacement[1]
                      + pair.displacement[2] * pair.displacement[2];
    const double r = ::sqrt(r2), x = pair.gamma * r;
    const double energy = pair.charge_product * ::erfc(x) / r;
    constexpr double two_over_root_pi = 1.12837916709551257390;
    return {energy, (energy
                     + two_over_root_pi * pair.charge_product * pair.gamma
                           * ::exp(-x * x))
                        / r2};
}
struct TailBounds {
    double energy = 0; // Absolute Hartree bound for all omitted pairs/images.
    double force = 0;  // Bound on each atom's omitted force norm, Ha/Bohr.
    double virial = 0; // Bound on omitted tensor Frobenius norm, Hartree.
};
class Model {
    std::vector<Pair> pairs_;
    TailBounds tail_;
    friend Model make_model(const std::vector<Site>&,
                            const std::array<double, 3>&, bool);

  public:
    const std::vector<Pair>& pairs() const { return pairs_; }
    TailBounds tail_bounds() const { return tail_; }
};

namespace detail {
// For nearest-image base displacement |d_k| <= L_k/2, the image shell n has
// |d_k-L_k*n| >= L_k*(|n|-1/2). Bound each one-dimensional Gaussian sum by
// the first omitted term plus its monotone integral. The union of the three
// outside-cube sets is bounded by sum_k tail_k * product_{j!=k} total_j.
inline TailBounds image_tail(double gamma, double q,
                             const std::array<double, 3>& cell,
                             const std::array<int, 3>& images) {
    constexpr double root_pi = 1.77245385090551602730;
    double total[3], tail[3], nearest = std::numeric_limits<double>::infinity();
    for (int k = 0; k < 3; ++k) {
        const double y = gamma * cell[k];
        const double t = y * (images[k] + .5);
        total[k] =
            1 + 2 * std::exp(-.25 * y * y) + root_pi / y * std::erfc(.5 * y);
        tail[k] = 2 * std::exp(-t * t) + root_pi / y * std::erfc(t);
        nearest = std::min(nearest, cell[k] * (images[k] + .5));
    }
    double gaussian_tail = 0;
    for (int k = 0; k < 3; ++k)
        gaussian_tail += tail[k] * total[(k + 1) % 3] * total[(k + 2) % 3];
    const double invr = 1 / nearest;
    const double prefactor = q / root_pi * gaussian_tail;
    const double invgr = invr / gamma;
    return {prefactor * invgr * invr,
            prefactor * (invgr * invr * invr + 2 * gamma * invr),
            prefactor * (invgr * invr + 2 * gamma)};
}
inline bool positive_half(int x, int y, int z) {
    return x > 0 || (x == 0 && (y > 0 || (y == 0 && z > 0)));
}
} // namespace detail

// Only the presently admitted GPW orthorhombic cell is accepted. Periodic:
// A<B, every retained image; A=B, one positive half lattice excluding origin.
// Isolated/MT: A<B at their actual unwrapped separation, with no image wrap.
// The fixed accuracy budgets are primitive summation errors, not SCF/oracle
// comparison tolerances. They are deliberately independent of eps_default.
inline Model make_model(const std::vector<Site>& sites,
                        const std::array<double, 3>& cell, bool periodic) {
    constexpr double energy_budget = 1e-14;
    constexpr double force_budget = 1e-14;
    constexpr double virial_budget = 1e-14;
    if (sites.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Gaussian-core site count overflow");
    if (periodic)
        for (double length : cell)
            if (!(length > 0) || !std::isfinite(length))
                throw std::invalid_argument(
                    "Gaussian-core invalid periodic cell");
    for (const auto& site : sites) {
        if (!(site.sigma > 0) || !std::isfinite(site.sigma)
            || !(site.charge >= 0) || !std::isfinite(site.charge))
            throw std::invalid_argument("Gaussian-core invalid charge/width");
        for (double x : site.position)
            if (!std::isfinite(x))
                throw std::invalid_argument("Gaussian-core nonfinite position");
    }
    Model model;
    const double n = static_cast<double>(sites.size());
    const double pair_budget_count = std::max(1.0, n * (n + 1) / 2);
    for (std::size_t a = 0; a < sites.size(); ++a)
        for (std::size_t b = a; b < sites.size(); ++b) {
            if ((!periodic && a == b) || sites[a].charge == 0
                || sites[b].charge == 0)
                continue;
            Pair base{static_cast<int>(a),
                      static_cast<int>(b),
                      {},
                      sites[a].charge * sites[b].charge,
                      1
                          / (std::sqrt(2.0)
                             * std::hypot(sites[a].sigma, sites[b].sigma))};
            if (!(base.gamma > 0) || !std::isfinite(base.gamma)
                || !std::isfinite(base.charge_product))
                throw std::overflow_error("Gaussian-core parameter overflow");
            for (int k = 0; k < 3; ++k) {
                // Wrap each position before subtracting: do not let large,
                // independent lattice translations invalidate the tail bound.
                base.displacement[k] =
                    periodic ? std::remainder(
                                   std::remainder(sites[a].position[k], cell[k])
                                       - std::remainder(sites[b].position[k],
                                                        cell[k]),
                                   cell[k])
                             : sites[a].position[k] - sites[b].position[k];
                if (!std::isfinite(base.displacement[k]))
                    throw std::overflow_error(
                        "Gaussian-core separation overflow");
            }
            if (a != b && base.displacement[0] == 0 && base.displacement[1] == 0
                && base.displacement[2] == 0)
                throw std::invalid_argument(
                    "Gaussian-core coincident charged sites");
            std::array<int, 3> images{};
            TailBounds omitted{};
            if (periodic) {
                for (double x = 6;; x += 1) {
                    if (x > 64)
                        throw std::overflow_error(
                            "Gaussian-core tail bound cannot be represented");
                    for (int k = 0; k < 3; ++k) {
                        const double count = std::max(
                            0.0, std::ceil(x / (base.gamma * cell[k]) - .5));
                        if (!std::isfinite(count)
                            || count > std::numeric_limits<int>::max() / 4)
                            throw std::overflow_error(
                                "Gaussian-core image count overflow");
                        images[k] = static_cast<int>(count);
                    }
                    omitted = detail::image_tail(
                        base.gamma, base.charge_product, cell, images);
                    if (!std::isfinite(omitted.energy)
                        || !std::isfinite(omitted.force)
                        || !std::isfinite(omitted.virial))
                        throw std::overflow_error(
                            "Gaussian-core nonfinite tail bound");
                    if (omitted.energy <= energy_budget / pair_budget_count
                        && omitted.force <= force_budget / pair_budget_count
                        && omitted.virial <= virial_budget / pair_budget_count)
                        break;
                }
                model.tail_.energy += omitted.energy;
                model.tail_.force += omitted.force;
                model.tail_.virial += omitted.virial;
            }
            for (int x = -images[0]; x <= images[0]; ++x)
                for (int y = -images[1]; y <= images[1]; ++y)
                    for (int z = -images[2]; z <= images[2]; ++z) {
                        if (a == b && !detail::positive_half(x, y, z))
                            continue;
                        const int image[3] = {x, y, z};
                        Pair pair = base;
                        for (int k = 0; k < 3; ++k)
                            if (periodic)
                                pair.displacement[k] -= cell[k] * image[k];
                        double r2 = 0;
                        for (double d : pair.displacement)
                            r2 += d * d;
                        if (!(r2 > 0) || !std::isfinite(r2))
                            throw std::overflow_error(
                                "Gaussian-core distance outside numeric range");
                        // Conservative representability bounds avoid doing
                        // the pair's scientific radial evaluation on the cold
                        // geometry path; each backend executes evaluate_pair.
                        const double e_bound =
                            pair.charge_product / std::sqrt(r2);
                        const double c_bound =
                            (e_bound
                             + 1.12837916709551257390 * pair.charge_product
                                   * pair.gamma)
                            / r2;
                        if (!std::isfinite(e_bound) || !std::isfinite(c_bound))
                            throw std::overflow_error(
                                "Gaussian-core pair outside numeric range");
                        model.pairs_.push_back(pair);
                    }
        }
    return model;
}
} // namespace ATV::gaussian_core
