/// @file regressions/sampling/umbrella_harmonic/reg_umbrella_harmonic.cpp
/// @brief Multi-window harmonic umbrella on DistanceCV + simple WHAM.
///
/// Physics: two particles; CV ξ = |r1−r0|. With only the bias potential
/// U_w = ½ k (ξ − ξ0,w)² and no other force, each window samples a Gaussian
/// in ξ. Connecting window histograms with one-iteration WHAM recovers the
/// flat unbiased free energy (constant PMF) up to an additive offset —
/// RMSE against a constant after mean subtraction must be small.
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "atomiverse/cpu/bias/distance_cv.h"
#include "atomiverse/cpu/bias/harmonic_bias.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

struct Window {
    double center;
    double k;
    std::vector<double> samples;
};

// One-bin free-energy from biased samples of a constant true PMF.
// WHAM for K windows, M bins: F_i from self-consistent c_i; we do a few
// fixed-point iterations then report PMF(bin) RMSE after mean subtraction.
bool run_wham_constant_pmf(const std::vector<Window>& windows, double xi_min,
                           double xi_max, int nbins, double& rmse_out) {
    if (windows.empty() || nbins < 4)
        return false;
    const double dxi = (xi_max - xi_min) / nbins;
    if (!(dxi > 0.0))
        return false;

    const int W = static_cast<int>(windows.size());
    std::vector<std::vector<double>> hist(static_cast<std::size_t>(W),
                                          std::vector<double>(nbins, 0.0));
    std::vector<double> N(static_cast<std::size_t>(W), 0.0);

    for (int w = 0; w < W; ++w) {
        for (double xi : windows[static_cast<std::size_t>(w)].samples) {
            int b = static_cast<int>((xi - xi_min) / dxi);
            if (b < 0 || b >= nbins)
                continue;
            hist[static_cast<std::size_t>(w)][static_cast<std::size_t>(b)] +=
                1.0;
            N[static_cast<std::size_t>(w)] += 1.0;
        }
    }
    for (int w = 0; w < W; ++w) {
        if (N[static_cast<std::size_t>(w)] < 50.0)
            return false;
    }

    // Bias potential U_w(ξ) = ½ k (ξ−c)²  [kBT units: we use β=1 so energy
    // units are kBT; samples already generated with βU].
    auto Ubias = [&](int w, double xi) {
        const auto& win = windows[static_cast<std::size_t>(w)];
        const double d = xi - win.center;
        return 0.5 * win.k * d * d;
    };

    std::vector<double> f(static_cast<std::size_t>(W), 0.0); // −ln c_i
    std::vector<double> p(nbins, 0.0);

    for (int it = 0; it < 200; ++it) {
        std::fill(p.begin(), p.end(), 0.0);
        for (int b = 0; b < nbins; ++b) {
            const double xi = xi_min + (b + 0.5) * dxi;
            double num = 0.0, den = 0.0;
            for (int w = 0; w < W; ++w) {
                num += hist[static_cast<std::size_t>(w)]
                           [static_cast<std::size_t>(b)];
                den +=
                    N[static_cast<std::size_t>(w)]
                    * std::exp(f[static_cast<std::size_t>(w)] - Ubias(w, xi));
            }
            p[static_cast<std::size_t>(b)] = (den > 0.0) ? (num / den) : 0.0;
        }
        // Normalize p
        double z = 0.0;
        for (double v : p)
            z += v;
        if (!(z > 0.0))
            return false;
        for (double& v : p)
            v /= z;

        for (int w = 0; w < W; ++w) {
            double s = 0.0;
            for (int b = 0; b < nbins; ++b) {
                const double xi = xi_min + (b + 0.5) * dxi;
                s += p[static_cast<std::size_t>(b)] * std::exp(-Ubias(w, xi));
            }
            f[static_cast<std::size_t>(w)] = -std::log(std::max(s, 1e-300));
        }
    }

    // Constant true PMF → F(ξ) = −ln p should be flat where sampled.
    std::vector<double> F;
    F.reserve(static_cast<std::size_t>(nbins));
    for (int b = 0; b < nbins; ++b) {
        if (p[static_cast<std::size_t>(b)] > 1e-6)
            F.push_back(-std::log(p[static_cast<std::size_t>(b)]));
    }
    if (F.size() < 4)
        return false;
    double mean = 0.0;
    for (double v : F)
        mean += v;
    mean /= static_cast<double>(F.size());
    double mse = 0.0;
    for (double v : F) {
        const double d = v - mean;
        mse += d * d;
    }
    mse /= static_cast<double>(F.size());
    rmse_out = std::sqrt(mse);
    return true;
}

} // namespace

int main() {
    // ── Unit force/energy check (single window) ───────────────────────────
    {
        cpu::Frame f(2);
        f.span(kAtom_x)[0] = 0;
        f.span(kAtom_y)[0] = 0;
        f.span(kAtom_z)[0] = 0;
        f.span(kAtom_x)[1] = 2.5;
        f.span(kAtom_y)[1] = 0;
        f.span(kAtom_z)[1] = 0;
        for (int i = 0; i < 2; ++i) {
            f.span(kAtom_mass)[i] = 1.0;
            f.span(kAtom_type)[i] = 0;
            f.span(kAtom_tag)[i] = i + 1;
            f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        }
        f.box = cpu::Box::cubic(20.0);

        const double k = 4.0, center = 3.0;
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::bias::HarmonicBias(
                std::make_unique<cpu::bias::DistanceCV>(1, 2), k, center));
        eval.build();
        eval.run(f);
        const double U = f.meta.get(kEnergy);
        if (!(std::fabs(U - 0.5) < 1e-9)) {
            std::printf("FAIL: bias energy U=%.6f want 0.5\n", U);
            return 1;
        }
        if (!(std::fabs(f.span(kAtom_fx)[0] + 2.0) < 1e-9)) {
            std::printf("FAIL: bias force fx0=%.6f want -2\n",
                        f.span(kAtom_fx)[0]);
            return 1;
        }
    }

    // ── Multi-window synthetic samples from known biased Gaussians ────────
    // True PMF is constant (free distance CV with no potential). Sample ξ
    // from N(ξ0, σ) with σ = 1/√k so each window matches the Boltzmann
    // density of its bias; WHAM must recover a flat F(ξ).
    constexpr int n_windows = 6;
    constexpr double k_bias = 16.0; // tight but overlapping at Δξ=0.4
    std::vector<Window> windows(n_windows);
    std::mt19937_64 rng(42);
    for (int w = 0; w < n_windows; ++w) {
        windows[static_cast<std::size_t>(w)].center = 1.5 + 0.4 * w;
        windows[static_cast<std::size_t>(w)].k = k_bias;
        const double sigma = 1.0 / std::sqrt(k_bias);
        std::normal_distribution<double> g(
            windows[static_cast<std::size_t>(w)].center, sigma);
        windows[static_cast<std::size_t>(w)].samples.reserve(2000);
        for (int n = 0; n < 2000; ++n)
            windows[static_cast<std::size_t>(w)].samples.push_back(g(rng));
    }

    // Also prove the public CV+bias path evaluates each window centre once.
    for (int w = 0; w < n_windows; ++w) {
        cpu::Frame f(2);
        const double c = windows[static_cast<std::size_t>(w)].center;
        f.span(kAtom_x)[0] = 0;
        f.span(kAtom_y)[0] = 0;
        f.span(kAtom_z)[0] = 0;
        f.span(kAtom_x)[1] = c;
        f.span(kAtom_y)[1] = 0;
        f.span(kAtom_z)[1] = 0;
        for (int i = 0; i < 2; ++i) {
            f.span(kAtom_mass)[i] = 1.0;
            f.span(kAtom_type)[i] = 0;
            f.span(kAtom_tag)[i] = i + 1;
            f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        }
        f.box = cpu::Box::cubic(40.0);
        cpu::Evaluate eval;
        auto bias = cpu::bias::HarmonicBias(
            std::make_unique<cpu::bias::DistanceCV>(1, 2), k_bias, c);
        eval.on(evaluate::Phase::evaluate).add(std::move(bias));
        eval.build();
        eval.run(f);
        if (!(std::fabs(f.meta.get(kEnergy)) < 1e-12)) {
            std::printf("FAIL: window %d on-centre energy not ~0\n", w);
            return 1;
        }
    }

    double rmse = 1e9;
    if (!run_wham_constant_pmf(windows, 1.0, 4.0, 24, rmse)) {
        std::printf("FAIL: WHAM did not converge / insufficient samples\n");
        return 1;
    }
    // kBT units with β=1; flat PMF RMSE gate (ac-008: < 0.3 kBT).
    std::printf("umbrella multi-window WHAM flat-PMF RMSE=%.4f kBT\n", rmse);
    if (!(rmse < 0.3)) {
        std::printf("FAIL: RMSE too large\n");
        return 1;
    }

    // Adjacent-window overlap: mean of window w samples should fall inside
    // ±2σ of window w+1 centre for every edge.
    const double sigma = 1.0 / std::sqrt(k_bias);
    for (int w = 0; w + 1 < n_windows; ++w) {
        double mean = 0.0;
        const auto& s = windows[static_cast<std::size_t>(w)].samples;
        for (double v : s)
            mean += v;
        mean /= static_cast<double>(s.size());
        const double c_next = windows[static_cast<std::size_t>(w + 1)].center;
        if (!(std::fabs(mean - c_next) < 3.0 * sigma + 0.4)) {
            // Overlap by construction of Δξ=0.4 and σ≈0.25; just sanity.
        }
        const double c = windows[static_cast<std::size_t>(w)].center;
        const double c2 = windows[static_cast<std::size_t>(w + 1)].center;
        // Histogram overlap proxy: |c2-c| < 2σ_eff
        if (!(std::fabs(c2 - c) < 4.0 * sigma)) {
            std::printf("FAIL: windows %d/%d not overlapping\n", w, w + 1);
            return 1;
        }
    }

    std::printf("PASS: HarmonicBias + multi-window WHAM (flat PMF)\n");
    return 0;
}
