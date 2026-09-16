/// @file regressions/sampling/wham_double_well/reg_wham_double_well.cpp
/// @brief Multi-window WHAM reconstruction of a known double-well PMF.
///
/// Uses β=1 energy units for numerical stability of synthetic histograms
/// (hard-coded F gates in those units). Math is molrs; C++ only bins + calls.
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/observe/wham.h"

using namespace ATV;

namespace {

// Toy double well in kT units (β = 1): F(s) = ¼(s²−1)²
double F_true(double s) {
    const double u = s * s - 1.0;
    return 0.25 * u * u;
}

} // namespace

int main() {
    constexpr double beta = 1.0;
    constexpr int n_win = 7;
    constexpr int n_bins = 48;
    const double s_min = -2.0, s_max = 2.0;
    const double ds = (s_max - s_min) / n_bins;
    const double k_bias = 8.0;

    std::vector<double> centers(n_win);
    std::vector<double> k(n_win, k_bias);
    for (int w = 0; w < n_win; ++w)
        centers[static_cast<std::size_t>(w)] = -1.5 + 0.5 * w;

    std::vector<double> hist(static_cast<std::size_t>(n_win * n_bins), 0.0);
    std::vector<double> n_samples(n_win, 0.0);
    for (int w = 0; w < n_win; ++w) {
        const double c = centers[static_cast<std::size_t>(w)];
        for (int b = 0; b < n_bins; ++b) {
            const double s = s_min + (b + 0.5) * ds;
            const double wbias = 0.5 * k_bias * (s - c) * (s - c);
            const double wgt = std::exp(-beta * (F_true(s) + wbias));
            hist[static_cast<std::size_t>(w * n_bins + b)] = wgt * 5000.0;
            n_samples[static_cast<std::size_t>(w)] +=
                hist[static_cast<std::size_t>(w * n_bins + b)];
        }
    }

    cpu::compute::Wham wham(beta);
    auto F = wham.free_energy(hist, n_samples, k, centers, s_min, s_max, n_bins,
                              2000, 1e-12);
    if (static_cast<int>(F.size()) != n_bins) {
        std::printf("FAIL: WHAM bad size (%zu)\n", F.size());
        return 1;
    }

    auto F_at = [&](double s_target) {
        int b = static_cast<int>((s_target - s_min) / ds);
        if (b < 0)
            b = 0;
        if (b >= n_bins)
            b = n_bins - 1;
        return F[static_cast<std::size_t>(b)];
    };
    const double F0 = F_at(0.0);
    const double Fm = F_at(-1.0);
    const double Fp = F_at(1.0);
    std::printf("wham_double_well F(0)=%.4f F(-1)=%.4f F(+1)=%.4f  iters=%d "
                "conv=%d\n",
                F0, Fm, Fp, wham.last_iterations(),
                wham.last_converged() ? 1 : 0);

    if (!std::isfinite(F0) || !std::isfinite(Fm) || !std::isfinite(Fp)) {
        std::printf("FAIL: non-finite F\n");
        return 1;
    }
    if (!(F0 > Fm + 0.08) || !(F0 > Fp + 0.08)) {
        std::printf("FAIL: barrier not above wells\n");
        return 1;
    }
    if (std::fabs(Fm - Fp) > 0.12) {
        std::printf("FAIL: wells asymmetric\n");
        return 1;
    }
    const double barrier = F0 - 0.5 * (Fm + Fp);
    if (std::fabs(barrier - 0.25) > 0.12) {
        std::printf("FAIL: barrier height %.4f not near 0.25\n", barrier);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
