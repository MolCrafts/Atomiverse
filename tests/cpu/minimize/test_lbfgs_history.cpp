/// @file tests/cpu/minimize/test_lbfgs_history.cpp
/// @brief Unit: LbfgsHistory curvature bookkeeping and the two-loop recursion.
///
/// The reference is Nocedal & Wright, Numerical Optimization 2e, Algorithm 7.4
/// with the eq. 7.20 initial scaling, written out plainly below over the pairs
/// the history is expected to hold.

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/minimize/lbfgs_history.h"

using ATV::cpu::minimizer::LbfgsHistory;

namespace {

using Vec = std::vector<double>;

double identity(double v) { return v; }

double dot(const Vec& a, const Vec& b) {
    double s = 0;
    for (std::size_t j = 0; j < a.size(); ++j)
        s += a[j] * b[j];
    return s;
}

/// N&W Alg. 7.4: `-H g` from pairs ordered oldest first.
Vec reference_direction(const std::vector<Vec>& s, const std::vector<Vec>& y,
                        const Vec& g) {
    const int m = static_cast<int>(s.size());
    Vec q = g;
    Vec alpha(m);
    for (int k = m - 1; k >= 0; --k) {
        alpha[k] = dot(s[k], q) / dot(y[k], s[k]);
        for (std::size_t j = 0; j < q.size(); ++j)
            q[j] -= alpha[k] * y[k][j];
    }
    const double gamma =
        m > 0 ? dot(s[m - 1], y[m - 1]) / dot(y[m - 1], y[m - 1]) : 1.0;
    for (double& v : q)
        v *= gamma;
    for (int k = 0; k < m; ++k) {
        const double beta = dot(y[k], q) / dot(y[k], s[k]);
        for (std::size_t j = 0; j < q.size(); ++j)
            q[j] += s[k][j] * (alpha[k] - beta);
    }
    for (double& v : q)
        v = -v;
    return q;
}

Vec direction_of(LbfgsHistory& h, const Vec& g) {
    Vec d(g.size(), 0.0);
    h.direction(g.data(), d.data(), static_cast<int>(g.size()));
    return d;
}

} // namespace

TEST(LbfgsHistory, EmptyHistoryIsSteepestDescent) {
    LbfgsHistory h(3, 4, identity);
    const Vec g = {1.5, -2.0, 0.25, 7.0};
    const Vec d = direction_of(h, g);
    for (std::size_t j = 0; j < g.size(); ++j)
        EXPECT_EQ(d[j], -g[j]);
}

TEST(LbfgsHistory, RejectedPairLeavesAFullHistoryIntact) {
    // memory = 1: the one slot is full after the first accepted pair. A second
    // pair with negative curvature (yᵀs < 0) must be dropped without touching
    // the stored one.
    LbfgsHistory h(1, 2, identity);
    const Vec x0 = {0, 0}, g0 = {1, 1};
    const Vec x1 = {1, 0}, g1 = {3, 1}; // s = (1, 0), y = (2, 0): yᵀs = 2
    const Vec x2 = {1, 1}, g2 = {3, 0}; // s = (0, 1), y = (0,-1): yᵀs = -1
    h.save(x0.data(), g0.data(), 2);
    h.push(x1.data(), g1.data(), 2);
    h.save(x1.data(), g1.data(), 2);
    ASSERT_EQ(h.stored(), 1);
    h.push(x2.data(), g2.data(), 2);
    EXPECT_EQ(h.stored(), 1);

    const Vec g = {1, 2};
    const Vec expected = reference_direction({{1, 0}}, {{2, 0}}, g);
    const Vec d = direction_of(h, g);
    for (std::size_t j = 0; j < g.size(); ++j)
        EXPECT_NEAR(d[j], expected[j], 1e-15) << "component " << j;
}

TEST(LbfgsHistory, TwoLoopMatchesTextbookRecursionOverARollingWindow) {
    // Gradient of a convex quadratic, g = A x + b with A SPD, so every pair
    // satisfies the curvature condition. Seven points through memory = 3 roll
    // the ring buffer twice over.
    constexpr int kDim = 7;
    auto grad = [](const Vec& x) {
        Vec g(kDim);
        for (int i = 0; i < kDim; ++i) {
            g[i] = (2.0 + i) * x[i] + 0.3 - 0.1 * i;
            if (i > 0)
                g[i] += 0.4 * x[i - 1];
            if (i + 1 < kDim)
                g[i] += 0.4 * x[i + 1];
        }
        return g;
    };
    std::vector<Vec> xs;
    for (int k = 0; k < 7; ++k) {
        Vec x(kDim);
        for (int i = 0; i < kDim; ++i)
            x[i] = std::sin(1.3 * k + 0.7 * i) / (1.0 + k);
        xs.push_back(x);
    }

    LbfgsHistory h(3, kDim, identity);
    std::vector<Vec> s_all, y_all;
    for (std::size_t k = 0; k < xs.size(); ++k) {
        const Vec g = grad(xs[k]);
        h.push(xs[k].data(), g.data(), kDim);
        if (k > 0) {
            const Vec gp = grad(xs[k - 1]);
            Vec s(kDim), y(kDim);
            for (int i = 0; i < kDim; ++i) {
                s[i] = xs[k][i] - xs[k - 1][i];
                y[i] = g[i] - gp[i];
            }
            ASSERT_GT(dot(y, s), 0.0);
            s_all.push_back(s);
            y_all.push_back(y);
        }
        const std::size_t keep = std::min<std::size_t>(3, s_all.size());
        const std::vector<Vec> s_win(s_all.end() - keep, s_all.end());
        const std::vector<Vec> y_win(y_all.end() - keep, y_all.end());
        ASSERT_EQ(h.stored(), static_cast<int>(keep));

        const Vec expected = reference_direction(s_win, y_win, g);
        const Vec d = direction_of(h, g);
        for (int i = 0; i < kDim; ++i)
            EXPECT_NEAR(d[i], expected[i],
                        1e-13 * (1.0 + std::abs(expected[i])))
                << "point " << k << " component " << i;
        h.save(xs[k].data(), g.data(), kDim);
    }
}
