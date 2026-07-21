/// @file cpu/minimizer/lbfgs_history.cpp
/// @brief Curvature history + Nocedal's two-loop recursion (N&W, Numerical
///        Optimization 2nd ed., Algorithm 7.4).

#include "atomiverse/cpu/minimizer/lbfgs_history.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace minimizer {

namespace {

/// Local (this-rank) inner product; the caller reduces it.
double dot_local(const double* a, const double* b, int dim) {
    double s = 0;
#pragma omp parallel for schedule(static) reduction(+ : s)
    for (int j = 0; j < dim; ++j)
        s += a[j] * b[j];
    return s;
}

} // namespace

LbfgsHistory::LbfgsHistory(int memory, int max_dim, Reducer sum)
    : memory_(memory), max_dim_(max_dim), sum_(std::move(sum)) {
    if (memory < 1)
        throw std::invalid_argument("LbfgsHistory: memory must be >= 1");
    if (max_dim < 1)
        throw std::invalid_argument("LbfgsHistory: max_dim must be >= 1");
    if (!sum_)
        throw std::invalid_argument(
            "LbfgsHistory: no Reducer — pass the identity for a "
            "serial/complete vector, "
            "or an all-reduce under domain decomposition. It is not defaulted "
            "because a "
            "silently-serial reduction gives every rank a different search "
            "direction.");

    s_history_.assign(memory_, std::vector<double>(max_dim_, 0.0));
    y_history_.assign(memory_, std::vector<double>(max_dim_, 0.0));
    rho_history_.assign(memory_, 0.0);
    x_prev_.assign(max_dim_, 0.0);
    g_prev_.assign(max_dim_, 0.0);
    q_ws_.assign(max_dim_, 0.0);
    r_ws_.assign(max_dim_, 0.0);
    alpha_ws_.assign(memory_, 0.0);
}

void LbfgsHistory::grow(int max_dim) {
    if (max_dim <= max_dim_)
        return;
    max_dim_ = max_dim;
    for (auto& h : s_history_)
        h.resize(max_dim_, 0.0);
    for (auto& h : y_history_)
        h.resize(max_dim_, 0.0);
    x_prev_.resize(max_dim_, 0.0);
    g_prev_.resize(max_dim_, 0.0);
    q_ws_.resize(max_dim_, 0.0);
    r_ws_.resize(max_dim_, 0.0);
}

void LbfgsHistory::reset() {
    n_stored_ = 0;
    oldest_ = 0;
    has_prev_ = false;
}

void LbfgsHistory::push(const double* x, const double* g, int dim) {
    if (!has_prev_)
        return;

    const int idx = (n_stored_ < memory_) ? n_stored_ : oldest_;
    auto& s = s_history_[idx];
    auto& y = y_history_[idx];

    double ys = 0, ss = 0, yy = 0;
#pragma omp parallel for schedule(static) reduction(+ : ys, ss, yy)
    for (int j = 0; j < dim; ++j) {
        s[j] = x[j] - x_prev_[j];
        y[j] = g[j] - g_prev_[j];
        ys += y[j] * s[j];
        ss += s[j] * s[j];
        yy += y[j] * y[j];
    }
    ys = sum_(ys);
    const double snorm = std::sqrt(sum_(ss));
    const double ynorm = std::sqrt(sum_(yy));

    // Curvature condition yᵀs > eps·‖s‖·‖y‖ keeps the implicit H positive definite.
    // A rejected pair is simply not stored — the recursion then runs on the older
    // pairs, which is the correct degradation, not an error.
    if (ys > 1e-12 * snorm * ynorm && ys > 0.0) {
        rho_history_[idx] = 1.0 / ys;
        if (n_stored_ < memory_)
            ++n_stored_;
        else
            oldest_ = (oldest_ + 1) % memory_;
    }
}

void LbfgsHistory::direction(const double* g, double* dir_out, int dim) {
    std::copy(g, g + dim, q_ws_.begin());

    for (int k = n_stored_ - 1; k >= 0; --k) {
        const int idx = (oldest_ + k) % memory_;
        const double a =
            rho_history_[idx]
            * sum_(dot_local(s_history_[idx].data(), q_ws_.data(), dim));
        alpha_ws_[k] = a;
        const auto& yk = y_history_[idx];
#pragma omp parallel for schedule(static)
        for (int j = 0; j < dim; ++j)
            q_ws_[j] -= a * yk[j];
    }

    // Initial Hessian scaling gamma = sᵀy / yᵀy (N&W eq. 7.20) — the single most
    // important detail for L-BFGS's practical convergence rate.
    double gamma = 1.0;
    if (n_stored_ > 0) {
        const int newest = (oldest_ + n_stored_ - 1) % memory_;
        const double yy = sum_(dot_local(y_history_[newest].data(),
                                         y_history_[newest].data(), dim));
        if (yy > 1e-30)
            gamma = 1.0 / (rho_history_[newest] * yy);
    }

#pragma omp parallel for schedule(static)
    for (int j = 0; j < dim; ++j)
        r_ws_[j] = gamma * q_ws_[j];

    for (int k = 0; k < n_stored_; ++k) {
        const int idx = (oldest_ + k) % memory_;
        const double b =
            rho_history_[idx]
            * sum_(dot_local(y_history_[idx].data(), r_ws_.data(), dim));
        const auto& sk = s_history_[idx];
        const double coeff = alpha_ws_[k] - b;
#pragma omp parallel for schedule(static)
        for (int j = 0; j < dim; ++j)
            r_ws_[j] += sk[j] * coeff;
    }

    // dir = -H g: the descent direction.
#pragma omp parallel for schedule(static)
    for (int j = 0; j < dim; ++j)
        dir_out[j] = -r_ws_[j];
}

void LbfgsHistory::save(const double* x, const double* g, int dim) {
    std::copy(x, x + dim, x_prev_.begin());
    std::copy(g, g + dim, g_prev_.begin());
    has_prev_ = true;
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
