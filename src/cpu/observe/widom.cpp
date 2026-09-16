/// @file src/cpu/observe/widom.cpp
/// @brief Widom test-particle insertion μ_ex estimator.

#include "atomiverse/cpu/observe/widom.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

namespace {
/// @brief Cut-and-shifted LJ pair energy (PairLJCut convention) [eV].
double lj_pair(double r2, double eps, double sig6, double v_shift,
               double rc_sq) {
    if (r2 >= rc_sq || r2 <= 0.0)
        return 0.0;
    const double r2inv = 1.0 / r2;
    const double r6inv = r2inv * r2inv * r2inv * sig6;
    return 4.0 * eps * (r6inv * r6inv - r6inv) - v_shift;
}
} // namespace

Widom::Widom(double temperature_K, double epsilon_ev, double sigma_a,
             double cutoff_a, int n_trials_in, std::uint64_t seed_in)
    : temperature(temperature_K), n_trials(n_trials_in), seed(seed_in),
      potential(WidomPotential::lj_cut), epsilon(epsilon_ev), sigma(sigma_a),
      cutoff(cutoff_a) {
    validate_params_();
    reset();
}

Widom::Widom(double temperature_K, double diameter_a, int n_trials_in,
             std::uint64_t seed_in, WidomPotential tag)
    : temperature(temperature_K), n_trials(n_trials_in), seed(seed_in),
      potential(tag), epsilon(0.0), sigma(diameter_a), cutoff(diameter_a) {
    if (tag != WidomPotential::hard_sphere)
        throw std::invalid_argument(
            "Widom HS ctor: potential tag must be hard_sphere");
    validate_params_();
    reset();
}

void Widom::validate_params_() const {
    if (!(temperature > 0.0))
        throw std::invalid_argument("Widom: temperature must be > 0 K");
    if (n_trials <= 0)
        throw std::invalid_argument("Widom: n_trials must be > 0");
    if (!(sigma > 0.0))
        throw std::invalid_argument("Widom: sigma must be > 0 Å");
    if (potential == WidomPotential::lj_cut) {
        if (!(epsilon > 0.0))
            throw std::invalid_argument(
                "Widom: epsilon must be > 0 eV for lj_cut");
        if (!(cutoff > 0.0))
            throw std::invalid_argument(
                "Widom: cutoff must be > 0 Å for lj_cut");
    }
}

void Widom::reset() {
    rng_.seed(seed);
    n_insertions_ = 0;
    sum_boltz_ = 0.0;
    if (potential == WidomPotential::lj_cut) {
        const double sig6 = sigma * sigma * sigma * sigma * sigma * sigma;
        const double rc_sq = cutoff * cutoff;
        const double r2inv = 1.0 / rc_sq;
        const double r6inv = r2inv * r2inv * r2inv * sig6;
        v_shift_ = 4.0 * epsilon * (r6inv * r6inv - r6inv);
    } else {
        v_shift_ = 0.0;
    }
}

void Widom::build(const BuildContext&) {
    validate_params_();
    reset();
}

void Widom::launch(const LaunchContext& ctx) { accumulate_frame_(ctx.frame); }

void Widom::sample(const Frame& frame) {
    validate_params_();
    accumulate_frame_(frame);
}

double Widom::delta_u_(const Frame& frame, double tx, double ty,
                       double tz) const {
    const int n = frame.nlocal();
    const auto x = frame.span(kAtom_x);
    const auto y = frame.span(kAtom_y);
    const auto z = frame.span(kAtom_z);
    const Box& box = frame.box;

    if (potential == WidomPotential::hard_sphere) {
        const double sig2 = sigma * sigma;
        for (int i = 0; i < n; ++i) {
            double dx = tx - x[i];
            double dy = ty - y[i];
            double dz = tz - z[i];
            box.min_image(dx, dy, dz);
            if (dx * dx + dy * dy + dz * dz < sig2)
                return std::numeric_limits<double>::infinity();
        }
        return 0.0;
    }

    // lj_cut
    const double sig6 = sigma * sigma * sigma * sigma * sigma * sigma;
    const double rc_sq = cutoff * cutoff;
    double du = 0.0;
    for (int i = 0; i < n; ++i) {
        double dx = tx - x[i];
        double dy = ty - y[i];
        double dz = tz - z[i];
        box.min_image(dx, dy, dz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        du += lj_pair(r2, epsilon, sig6, v_shift_, rc_sq);
    }
    return du;
}

void Widom::accumulate_frame_(const Frame& frame) {
    const double Lx = frame.box.length(0);
    const double Ly = frame.box.length(1);
    const double Lz = frame.box.length(2);
    if (!(Lx > 0.0 && Ly > 0.0 && Lz > 0.0))
        throw std::invalid_argument("Widom: box edges must be > 0");

    const double kT = units::kB_ev_per_k * temperature;
    const double beta = 1.0 / kT;
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Origin of the orthorhombic cell for fractional → Cartesian (min corner).
    // Box stores edge lengths; positions are typically in [0, L) for cubic.
    // Use absolute coordinates consistent with host atoms (same lab frame):
    // sample uniformly in [0, L) along each axis (cubic / orthorhombic).
    for (int t = 0; t < n_trials; ++t) {
        const double tx = U01(rng_) * Lx;
        const double ty = U01(rng_) * Ly;
        const double tz = U01(rng_) * Lz;
        const double du = delta_u_(frame, tx, ty, tz);
        double w = 0.0;
        if (std::isfinite(du)) {
            // Clamp extreme repulsion for LJ to avoid underflow noise.
            const double x = -beta * du;
            w = (x < -700.0) ? 0.0 : std::exp(x);
        }
        sum_boltz_ += w;
        ++n_insertions_;
    }
}

double Widom::boltzmann_avg() const {
    if (n_insertions_ <= 0)
        return 0.0;
    return sum_boltz_ / static_cast<double>(n_insertions_);
}

double Widom::mu_ex() const {
    const double avg = boltzmann_avg();
    if (!(avg > 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    const double kT = units::kB_ev_per_k * temperature;
    return -kT * std::log(avg);
}

} // namespace cpu
} // namespace ATV
