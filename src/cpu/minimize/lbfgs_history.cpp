/// @file cpu/minimize/lbfgs_history.cpp
/// @brief Curvature history + Nocedal's two-loop recursion (N&W, Numerical
///        Optimization 2nd ed., Algorithm 7.4).

#include "atomiverse/cpu/minimize/lbfgs_history.h"
#include "atomiverse/cpu/minimize/vector_loop.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace minimizer {

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
    yy_history_.assign(memory_, 0.0);
    s_spare_.assign(max_dim_, 0.0);
    y_spare_.assign(max_dim_, 0.0);
    x_prev_.assign(max_dim_, 0.0);
    g_prev_.assign(max_dim_, 0.0);
    q_ws_.assign(max_dim_, 0.0);
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
    s_spare_.resize(max_dim_, 0.0);
    y_spare_.resize(max_dim_, 0.0);
    x_prev_.resize(max_dim_, 0.0);
    g_prev_.resize(max_dim_, 0.0);
    q_ws_.resize(max_dim_, 0.0);
}

void LbfgsHistory::reset() {
    n_stored_ = 0;
    oldest_ = 0;
    has_prev_ = false;
}

void LbfgsHistory::push(const double* x, const double* g, int dim) {
    if (!has_prev_)
        return;

    double* __restrict s = s_spare_.data();
    double* __restrict y = y_spare_.data();
    const double* __restrict xp = x_prev_.data();
    const double* __restrict gp = g_prev_.data();
    double ys = 0, ss = 0, yy = 0;
#pragma omp parallel for simd schedule(static)                                 \
    reduction(+ : ys, ss, yy) if (parallel : dim >= kParallelMinElements)
    for (int j = 0; j < dim; ++j) {
        s[j] = x[j] - xp[j];
        y[j] = g[j] - gp[j];
        ys += y[j] * s[j];
        ss += s[j] * s[j];
        yy += y[j] * y[j];
    }
    ys = sum_(ys);
    const double snorm = std::sqrt(sum_(ss));
    yy = sum_(yy);
    const double ynorm = std::sqrt(yy);

    // Curvature condition yᵀs > eps·‖s‖·‖y‖ keeps the implicit H positive definite.
    // A rejected pair is simply not stored — the recursion then runs on the older
    // pairs, which is the correct degradation, not an error. It was differenced
    // into the spare buffers, so the ring (full or not) is untouched.
    if (ys > 1e-12 * snorm * ynorm && ys > 0.0) {
        const int idx = (n_stored_ < memory_) ? n_stored_ : oldest_;
        s_history_[idx].swap(s_spare_);
        y_history_[idx].swap(y_spare_);
        rho_history_[idx] = 1.0 / ys;
        yy_history_[idx] = yy;
        if (n_stored_ < memory_)
            ++n_stored_;
        else
            oldest_ = (oldest_ + 1) % memory_;
    }
}

void LbfgsHistory::direction(const double* g, double* dir_out, int dim) {
    // N&W Algorithm 7.4 with every vector update fused into the pass that
    // takes the next inner product: q -= a_k y_k carries s_{k-1}·q, the
    // gamma scaling carries y_0·r, and r += c_k s_k carries y_{k+1}·r (the
    // last one writes dir = -r). Each element sees the same operations in the
    // same order as the unfused loops; only the inner-product partial sums
    // are vectorised. 2m + 1 passes over `dim` instead of 4m + 4.
    const int m = n_stored_;
    if (m == 0) {
        // gamma = 1: dir = -g.
#pragma omp parallel for simd if (parallel : dim >= kParallelMinElements)      \
    schedule(static)
        for (int j = 0; j < dim; ++j)
            dir_out[j] = -g[j];
        return;
    }
    auto slot = [&](int k) { return (oldest_ + k) % memory_; };

    // Initial Hessian scaling gamma = sᵀy / yᵀy (N&W eq. 7.20) — the single most
    // important detail for L-BFGS's practical convergence rate. yᵀy of the
    // newest pair was reduced when push() stored it.
    const int newest = slot(m - 1);
    double gamma = 1.0;
    if (yy_history_[newest] > 1e-30)
        gamma = 1.0 / (rho_history_[newest] * yy_history_[newest]);

    double* __restrict q = q_ws_.data();
    double part = dot_local(s_history_[newest].data(), g, dim);
    const double* src = g; // q before this pass's update (g on the first)
    for (int k = m - 1; k >= 0; --k) {
        const int idx = slot(k);
        const double a = rho_history_[idx] * sum_(part);
        alpha_ws_[k] = a;
        const double* __restrict yk = y_history_[idx].data();
        // Next inner product: s_{k-1}·q, or y_0·r with r = gamma q after k=0.
        const double* __restrict next =
            (k > 0) ? s_history_[slot(k - 1)].data() : yk;
        const double* in = src; // aliases q after the first pass (in place)
        double acc = 0;
        if (k > 0) {
#pragma omp parallel for simd schedule(static)                                 \
    reduction(+ : acc) if (parallel : dim >= kParallelMinElements)
            for (int j = 0; j < dim; ++j) {
                const double qj = in[j] - a * yk[j];
                q[j] = qj;
                acc += next[j] * qj;
            }
        } else {
#pragma omp parallel for simd schedule(static)                                 \
    reduction(+ : acc) if (parallel : dim >= kParallelMinElements)
            for (int j = 0; j < dim; ++j) {
                const double qj = in[j] - a * yk[j];
                const double rj = gamma * qj;
                q[j] = rj;
                acc += next[j] * rj;
            }
        }
        part = acc;
        src = q;
    }

    // q now holds r = gamma q; part = y_0·r (local).
    for (int k = 0; k < m; ++k) {
        const int idx = slot(k);
        const double b = rho_history_[idx] * sum_(part);
        const double coeff = alpha_ws_[k] - b;
        const double* __restrict sk = s_history_[idx].data();
        if (k + 1 < m) {
            const double* __restrict next = y_history_[slot(k + 1)].data();
            double acc = 0;
#pragma omp parallel for simd schedule(static)                                 \
    reduction(+ : acc) if (parallel : dim >= kParallelMinElements)
            for (int j = 0; j < dim; ++j) {
                const double rj = q[j] + sk[j] * coeff;
                q[j] = rj;
                acc += next[j] * rj;
            }
            part = acc;
        } else {
            // dir = -H g: the descent direction.
#pragma omp parallel for simd if (parallel : dim >= kParallelMinElements)      \
    schedule(static)
            for (int j = 0; j < dim; ++j) {
                const double rj = q[j] + sk[j] * coeff;
                dir_out[j] = -rj;
            }
        }
    }
}

void LbfgsHistory::save(const double* x, const double* g, int dim) {
    std::copy(x, x + dim, x_prev_.begin());
    std::copy(g, g + dim, g_prev_.begin());
    has_prev_ = true;
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
