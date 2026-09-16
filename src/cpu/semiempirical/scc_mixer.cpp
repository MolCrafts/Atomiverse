/// @file src/cpu/semiempirical/scc_mixer.cpp
/// Anderson/Pulay charge mixing shared by the SCC methods.
#include "atomiverse/cpu/semiempirical/internal/scc_mixer.h"

#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

ChargeMixer::ChargeMixer(double beta, int depth)
    : beta_(beta), depth_(static_cast<std::size_t>(depth)) {
    if (!(beta > 0.0) || beta > 1.0)
        throw std::invalid_argument("ChargeMixer: beta must be in (0, 1]");
    if (depth < 1)
        throw std::invalid_argument("ChargeMixer: depth must be >= 1");
}

bool ChargeMixer::pulay_coefficients(std::vector<double>& out) const {
    const std::size_t m = f_.size();
    const std::size_t dim = m + 1;

    // B = [ <f_i,f_j>  -1 ; -1  0 ],  rhs = (0,...,0,-1)  ⇒  Σ c_j = 1.
    std::vector<double> B(dim * dim, 0.0);
    std::vector<double> rhs(dim, 0.0);
    double scale = 0.0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            double dot = 0.0;
            for (std::size_t k = 0; k < f_[i].size(); ++k)
                dot += f_[i][k] * f_[j][k];
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
    std::vector<double> sol(dim, 0.0);
    for (std::size_t ii = dim; ii-- > 0;) {
        double s = rhs[ii];
        for (std::size_t c = ii + 1; c < dim; ++c)
            s -= B[ii * dim + c] * sol[c];
        sol[ii] = s / B[ii * dim + ii];
        if (!std::isfinite(sol[ii]))
            return false;
    }
    out.assign(sol.begin(), sol.begin() + static_cast<std::ptrdiff_t>(m));
    return true;
}

std::vector<double> ChargeMixer::next(const std::vector<double>& x_in,
                                      const std::vector<double>& x_out) {
    if (x_in.empty() || x_in.size() != x_out.size())
        throw std::invalid_argument(
            "ChargeMixer: empty or size-mismatched population vectors");

    std::vector<double> residual(x_in.size());
    for (std::size_t k = 0; k < x_in.size(); ++k)
        residual[k] = x_out[k] - x_in[k];
    x_.push_back(x_in);
    f_.push_back(std::move(residual));
    if (x_.size() > depth_) {
        x_.erase(x_.begin());
        f_.erase(f_.begin());
    }

    // Shrink the history until the DIIS system is solvable; a single retained
    // iterate degenerates to plain linear damping, which always is.
    std::vector<double> c;
    while (x_.size() > 1 && !pulay_coefficients(c)) {
        x_.erase(x_.begin());
        f_.erase(f_.begin());
    }
    if (x_.size() == 1)
        c.assign(1, 1.0);

    std::vector<double> mixed(x_in.size(), 0.0);
    for (std::size_t j = 0; j < x_.size(); ++j)
        for (std::size_t k = 0; k < mixed.size(); ++k)
            mixed[k] += c[j] * (x_[j][k] + beta_ * f_[j][k]);
    return mixed;
}

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
