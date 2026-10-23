/// @file src/cpu/semiempirical/scc_mixer.cpp
/// Anderson/Pulay charge mixing shared by the SCC methods.
#include "atomiverse/cpu/semiempirical/internal/scc_mixer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

ChargeMixer::ChargeMixer(double beta, int depth)
    : beta_(beta), depth_(static_cast<std::size_t>(depth)) {
    if (depth < 1)
        throw std::invalid_argument("ChargeMixer: depth must be >= 1");
    reset(0, beta);
}

void ChargeMixer::reset(std::size_t size, double beta) {
    if (!(beta > 0.0) || beta > 1.0)
        throw std::invalid_argument("ChargeMixer: beta must be in (0, 1]");
    beta_ = beta;
    first_ = count_ = 0;
    x_.resize(depth_);
    f_.resize(depth_);
    for (std::size_t j = 0; j < depth_; ++j) {
        x_[j].resize(size);
        f_[j].resize(size);
    }
    mixed_.resize(size);
    gram_.resize((depth_ + 1) * (depth_ + 1));
    rhs_.resize(depth_ + 1);
    coefficients_.resize(depth_ + 1);
}

bool ChargeMixer::pulay_coefficients() {
    const std::size_t m = count_;
    const std::size_t dim = m + 1;

    // B = [ <f_i,f_j>  -1 ; -1  0 ], rhs = (0,...,0,-1) ⇒ Σ c_j = 1.
    auto& B = gram_;
    auto& rhs = rhs_;
    std::fill_n(B.begin(), dim * dim, 0.0);
    std::fill_n(rhs.begin(), dim, 0.0);
    double scale = 0.0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            double dot = 0.0;
            for (std::size_t k = 0; k < mixed_.size(); ++k)
                dot +=
                    f_[(first_ + i) % depth_][k] * f_[(first_ + j) % depth_][k];
            B[i * dim + j] = dot;
            scale = std::max(scale, std::abs(dot));
        }
    if (!(scale > 0.0))
        return false;
    for (std::size_t i = 0; i < m; ++i) {
        B[i * dim + m] = -1.0;
        B[m * dim + i] = -1.0;
    }
    rhs[m] = -1.0;

    // Gaussian elimination with partial pivoting.
    for (std::size_t col = 0; col < dim; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < dim; ++r)
            if (std::abs(B[r * dim + col]) > std::abs(B[piv * dim + col]))
                piv = r;
        if (std::abs(B[piv * dim + col]) < 1e-14 * std::max(scale, 1.0))
            return false;
        if (piv != col) {
            for (std::size_t c = 0; c < dim; ++c)
                std::swap(B[piv * dim + c], B[col * dim + c]);
            std::swap(rhs[piv], rhs[col]);
        }
        const double d = B[col * dim + col];
        for (std::size_t r = col + 1; r < dim; ++r) {
            const double factor = B[r * dim + col] / d;
            if (factor == 0.0)
                continue;
            for (std::size_t c = col; c < dim; ++c)
                B[r * dim + c] -= factor * B[col * dim + c];
            rhs[r] -= factor * rhs[col];
        }
    }
    auto& sol = coefficients_;
    std::fill_n(sol.begin(), dim, 0.0);
    for (std::size_t ii = dim; ii-- > 0;) {
        double s = rhs[ii];
        for (std::size_t c = ii + 1; c < dim; ++c)
            s -= B[ii * dim + c] * sol[c];
        sol[ii] = s / B[ii * dim + ii];
        if (!std::isfinite(sol[ii]))
            return false;
    }
    return true;
}

const std::vector<double>& ChargeMixer::next(const std::vector<double>& x_in,
                                             const std::vector<double>& x_out) {
    if (x_in.empty() || x_in.size() != x_out.size())
        throw std::invalid_argument(
            "ChargeMixer: empty or size-mismatched population vectors");

    if (mixed_.size() != x_in.size())
        reset(x_in.size(), beta_);
    if (count_ == depth_) {
        first_ = (first_ + 1) % depth_;
        --count_;
    }
    const std::size_t row = (first_ + count_) % depth_;
    for (std::size_t k = 0; k < x_in.size(); ++k) {
        x_[row][k] = x_in[k];
        f_[row][k] = x_out[k] - x_in[k];
    }
    ++count_;

    // Shrink the history until the DIIS system is solvable; a single retained
    // iterate degenerates to plain linear damping, which always is.
    while (count_ > 1 && !pulay_coefficients()) {
        first_ = (first_ + 1) % depth_;
        --count_;
    }
    if (count_ == 1)
        coefficients_[0] = 1.0;

    std::fill(mixed_.begin(), mixed_.end(), 0.0);
    for (std::size_t j = 0; j < count_; ++j) {
        const std::size_t retained = (first_ + j) % depth_;
        for (std::size_t k = 0; k < mixed_.size(); ++k)
            mixed_[k] +=
                coefficients_[j] * (x_[retained][k] + beta_ * f_[retained][k]);
    }
    return mixed_;
}

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
