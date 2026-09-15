/// @file cpu/scf/internal/stability.cpp
/// @brief StabilityAnalysis — dense and matrix-free Davidson lowest-root
///        solves of the (A + B) channel blocks.

#include "atomiverse/cpu/scf/internal/stability.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

double StabilityAnalysis::lowest(Channel ch) const { return solve_(ch).value; }

RealVector StabilityAnalysis::lowest_vector(Channel ch) const {
    return solve_(ch).vector;
}

double StabilityAnalysis::lowest_dense(Channel ch) const {
    return solve_dense_(ch).value;
}

double StabilityAnalysis::lowest_davidson(Channel ch) const {
    return solve_davidson_(ch).value;
}

StabilityAnalysis::Root StabilityAnalysis::solve_(Channel ch) const {
    return (hessian_.n_ov() <= dense_threshold) ? solve_dense_(ch)
                                                : solve_davidson_(ch);
}

StabilityAnalysis::Root StabilityAnalysis::solve_dense_(Channel ch) const {
    const RealMatrix M = hessian_.dense_plus(ch);
    auto [w, U] = xt::linalg::eigh(M); // ascending eigenvalues
    Root root;
    root.value = w(0);
    root.vector = xt::zeros<double>({M.shape()[0]});
    for (std::size_t r = 0; r < M.shape()[0]; ++r)
        root.vector(r) = U(r, 0);
    return root;
}

StabilityAnalysis::Root StabilityAnalysis::solve_davidson_(Channel ch) const {
    const int n = hessian_.n_ov();
    const RealVector diag = hessian_.diagonal_plus(ch);

    // Deterministic start: the coordinate vector of the smallest diagonal
    // element (the standard Davidson seed; ties break to the lowest index).
    int seed = 0;
    for (int i = 1; i < n; ++i)
        if (diag(i) < diag(seed))
            seed = i;

    const int max_subspace = std::min(n, 40);
    const int max_iter = 500;
    const double res_tol = 1e-9;    // ‖r‖₂ threshold [Hartree]
    const double theta_tol = 1e-13; // Ritz-value stagnation [Hartree]

    std::vector<RealVector> V, W; // subspace vectors and their σ products
    RealVector b = xt::zeros<double>({static_cast<std::size_t>(n)});
    b(seed) = 1.0;
    V.push_back(b);
    W.push_back(hessian_.sigma_plus(b, ch));

    double theta_old = 0.0;
    bool have_old = false;
    for (int iter = 0; iter < max_iter; ++iter) {
        const int m = static_cast<int>(V.size());
        RealMatrix H = xt::zeros<double>(
            {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
        for (int p = 0; p < m; ++p)
            for (int q = 0; q < m; ++q) {
                double s = 0.0;
                for (int r = 0; r < n; ++r)
                    s += V[static_cast<std::size_t>(p)](r)
                         * W[static_cast<std::size_t>(q)](r);
                H(p, q) = s;
            }
        // Symmetrize against round-off before the subspace diagonalization.
        for (int p = 0; p < m; ++p)
            for (int q = p + 1; q < m; ++q) {
                const double s = 0.5 * (H(p, q) + H(q, p));
                H(p, q) = s;
                H(q, p) = s;
            }
        auto [w, S] = xt::linalg::eigh(H);
        const double theta = w(0);

        RealVector x = xt::zeros<double>({static_cast<std::size_t>(n)});
        RealVector Ax = xt::zeros<double>({static_cast<std::size_t>(n)});
        for (int p = 0; p < m; ++p) {
            const double c = S(p, 0);
            for (int r = 0; r < n; ++r) {
                x(r) += c * V[static_cast<std::size_t>(p)](r);
                Ax(r) += c * W[static_cast<std::size_t>(p)](r);
            }
        }
        RealVector rvec = Ax - theta * x;
        double rnorm = 0.0;
        for (int r = 0; r < n; ++r)
            rnorm += rvec(r) * rvec(r);
        rnorm = std::sqrt(rnorm);

        const bool stagnant =
            have_old && std::abs(theta - theta_old) < theta_tol;
        if (rnorm < res_tol || (stagnant && rnorm < 1e-7) || m == n) {
            // Normalize the Ritz vector (V is orthonormal, so ‖x‖ ≈ 1).
            double xn = 0.0;
            for (int r = 0; r < n; ++r)
                xn += x(r) * x(r);
            xn = std::sqrt(xn);
            for (int r = 0; r < n; ++r)
                x(r) /= xn;
            return {theta, x};
        }
        theta_old = theta;
        have_old = true;

        // Diagonal (Davidson) preconditioner with a clamped denominator.
        RealVector t = xt::zeros<double>({static_cast<std::size_t>(n)});
        for (int r = 0; r < n; ++r) {
            double denom = theta - diag(r);
            if (std::abs(denom) < 1e-8)
                denom = (denom < 0.0) ? -1e-8 : 1e-8;
            t(r) = rvec(r) / denom;
        }

        // Restart on a full subspace: collapse to the current Ritz vector.
        if (m >= max_subspace) {
            RealVector Ax_kept = Ax; // σ·x is a linear combo of stored W
            V.clear();
            W.clear();
            double xn = 0.0;
            for (int r = 0; r < n; ++r)
                xn += x(r) * x(r);
            xn = std::sqrt(xn);
            for (int r = 0; r < n; ++r) {
                x(r) /= xn;
                Ax_kept(r) /= xn;
            }
            V.push_back(x);
            W.push_back(Ax_kept);
        }

        // Orthogonalize the correction against the subspace (two MGS passes).
        for (int pass = 0; pass < 2; ++pass)
            for (const RealVector& v : V) {
                double proj = 0.0;
                for (int r = 0; r < n; ++r)
                    proj += v(r) * t(r);
                for (int r = 0; r < n; ++r)
                    t(r) -= proj * v(r);
            }
        double tn = 0.0;
        for (int r = 0; r < n; ++r)
            tn += t(r) * t(r);
        tn = std::sqrt(tn);
        if (tn < 1e-12) {
            // No new direction exists at working precision — the Ritz pair
            // is converged to the representable limit.
            double xn = 0.0;
            for (int r = 0; r < n; ++r)
                xn += x(r) * x(r);
            xn = std::sqrt(xn);
            for (int r = 0; r < n; ++r)
                x(r) /= xn;
            return {theta, x};
        }
        for (int r = 0; r < n; ++r)
            t(r) /= tn;
        V.push_back(t);
        W.push_back(hessian_.sigma_plus(t, ch));
    }
    throw std::runtime_error(
        "StabilityAnalysis: Davidson lowest-root solve did not converge in "
        + std::to_string(max_iter) + " iterations (n_ov=" + std::to_string(n)
        + ") — no fallback");
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
