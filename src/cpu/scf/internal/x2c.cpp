/// @file cpu/scf/internal/x2c.cpp
/// @brief One-step spin-free x2c-1e decoupling (X2cWorkspace).

#include "atomiverse/cpu/scf/internal/x2c.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/linalg/eigensolve.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {
namespace {

/// @brief R = S^{−1/2} (S^{−1/2} S̃ S^{−1/2})^{−1/2} S^{1/2}
///        (PySCF `x2c._get_r`, eigenvalue threshold 1e-14).
RealMatrix get_r(const RealMatrix& S, const RealMatrix& S_tilde) {
    constexpr double kTau = 1e-14;
    auto [s_eig, V] = xt::linalg::eigh(S);
    const int n = static_cast<int>(S.shape(0));

    int first = 0;
    while (first < n && s_eig(first) <= kTau)
        ++first;
    const int m = n - first;
    if (m == 0)
        throw std::runtime_error(
            "X2cWorkspace: every overlap eigenvalue below 1e-14 — no "
            "basis left for the renormalisation matrix R");

    // r_mid = s^{−1/2} Vᵀ S̃ V s^{−1/2} over kept directions (m × m).
    RealMatrix St_proj = xt::zeros<double>({m, m});
    for (int a = 0; a < m; ++a) {
        const int ia = first + a;
        const double inv_sa = 1.0 / std::sqrt(s_eig(ia));
        for (int b = 0; b < m; ++b) {
            const int ib = first + b;
            const double inv_sb = 1.0 / std::sqrt(s_eig(ib));
            double acc = 0.0;
            for (int p = 0; p < n; ++p) {
                double Sp = 0.0;
                for (int q = 0; q < n; ++q)
                    Sp += S_tilde(p, q) * V(q, ib);
                acc += V(p, ia) * Sp;
            }
            St_proj(a, b) = inv_sa * acc * inv_sb;
        }
    }

    auto [w1, v1] = xt::linalg::eigh(St_proj);
    RealMatrix r_mid = xt::zeros<double>({m, m});
    for (int k = 0; k < m; ++k) {
        if (w1(k) <= kTau)
            continue;
        const double inv_sqrt_w = 1.0 / std::sqrt(w1(k));
        for (int i = 0; i < m; ++i) {
            const double vi = v1(i, k) * inv_sqrt_w;
            for (int j = 0; j < m; ++j)
                r_mid(i, j) += vi * v1(j, k);
        }
    }

    // r_small[a,b] = s_a^{−1/2} * r_mid[a,b] * s_b^{1/2}
    RealMatrix r_small = xt::zeros<double>({m, m});
    for (int a = 0; a < m; ++a) {
        const double inv_sa = 1.0 / std::sqrt(s_eig(first + a));
        for (int b = 0; b < m; ++b) {
            const double sb = std::sqrt(s_eig(first + b));
            r_small(a, b) = inv_sa * r_mid(a, b) * sb;
        }
    }

    // R = V[:,kept] r_small V[:,kept]ᵀ
    RealMatrix R = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int a = 0; a < m; ++a) {
                double row = 0.0;
                for (int b = 0; b < m; ++b)
                    row += r_small(a, b) * V(j, first + b);
                acc += V(i, first + a) * row;
            }
            R(i, j) = acc;
        }
    }
    return R;
}

} // namespace

X2cWorkspace::X2cWorkspace(const RealMatrix& S, const RealMatrix& T,
                           const RealMatrix& V, const RealMatrix& W, double c) {
    const int n = static_cast<int>(S.shape(0));
    if (static_cast<int>(S.shape(1)) != n || static_cast<int>(T.shape(0)) != n
        || static_cast<int>(T.shape(1)) != n
        || static_cast<int>(V.shape(0)) != n
        || static_cast<int>(V.shape(1)) != n
        || static_cast<int>(W.shape(0)) != n
        || static_cast<int>(W.shape(1)) != n) {
        throw std::invalid_argument(
            "X2cWorkspace: S/T/V/W must be square and equal-sized (got n="
            + std::to_string(n) + ")");
    }
    if (!(c > 0.0) || !std::isfinite(c))
        throw std::invalid_argument(
            "X2cWorkspace: speed of light c must be finite and positive");

    const double c2 = c * c;
    const double inv_4c2 = 0.25 / c2;
    const double inv_2c2 = 0.5 / c2;
    const int n2 = 2 * n;

    // Modified Dirac matrix h and metric M (kinetic-balance, spin-free).
    RealMatrix h = xt::zeros<double>({n2, n2});
    RealMatrix M = xt::zeros<double>({n2, n2});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h(i, j) = V(i, j);
            h(i, n + j) = T(i, j);
            h(n + i, j) = T(i, j);
            h(n + i, n + j) = W(i, j) * inv_4c2 - T(i, j);
            M(i, j) = S(i, j);
            M(n + i, n + j) = T(i, j) * inv_2c2;
        }
    }

    // Generalised eigenproblem h A = M A ε. PySCF happy path takes the
    // upper half a[:, n:] of the ascending spectrum (positive-energy
    // branch). Filter ε > −c² and keep the n largest if the split is not
    // exactly half-and-half.
    RealMatrix A_mat;
    RealVector eigs;
    try {
        auto sol = linalg::diagonalize_fock_generalized(h, M);
        eigs = std::move(sol.first);
        A_mat = std::move(sol.second);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(
            std::string("X2cWorkspace: Dirac generalised eigenproblem "
                        "failed (metric not positive-definite?): ")
            + e.what());
    }

    // Lowest positive-energy eigenvalue of the Dirac matrix.
    lowest_positive_dirac_ = eigs(n2 - 1);
    bool found_pos = false;
    for (int k = 0; k < n2; ++k) {
        if (eigs(k) > -c2) {
            lowest_positive_dirac_ = eigs(k);
            found_pos = true;
            break;
        }
    }
    if (!found_pos)
        throw std::runtime_error(
            "X2cWorkspace: no positive-energy Dirac solutions (ε > −c²)");

    // Large / small component blocks of the n highest positive states.
    // Walk descending so column 0 holds the most positive solution; the
    // ordering is immaterial for X = C_S C_L⁻¹.
    RealMatrix Cl = xt::zeros<double>({n, n});
    RealMatrix Cs = xt::zeros<double>({n, n});
    int col = 0;
    for (int k = n2 - 1; k >= 0 && col < n; --k) {
        if (eigs(k) <= -c2)
            continue;
        for (int i = 0; i < n; ++i) {
            Cl(i, col) = A_mat(i, k);
            Cs(i, col) = A_mat(n + i, k);
        }
        ++col;
    }
    if (col < n)
        throw std::runtime_error(
            "X2cWorkspace: fewer than n positive-energy Dirac solutions "
            "(got "
            + std::to_string(col) + ", need " + std::to_string(n) + ")");

    // X = C_S C_L⁻¹  ⇔  solve C_Lᵀ Xᵀ = C_Sᵀ
    RealMatrix X;
    try {
        const RealMatrix ClT = xt::transpose(Cl);
        const RealMatrix CsT = xt::transpose(Cs);
        const RealMatrix XT = xt::linalg::solve(ClT, CsT);
        X = xt::transpose(XT);
    } catch (...) {
        throw std::runtime_error(
            "X2cWorkspace: large-component block C_L is singular — cannot "
            "form the decoupling matrix X");
    }

    // S̃ = S + X† T X / (2c²)
    const RealMatrix TX = xt::linalg::dot(T, X);
    const RealMatrix XtTX = xt::linalg::dot(xt::transpose(X), TX);
    const RealMatrix S_tilde = S + XtTX * inv_2c2;

    // h1 = V + T X + X† T + X† (W/(4c²) − T) X
    //    = V + TX + (TX)† − X†TX + X†WX/(4c²)   (PySCF `_get_hcore_fw`)
    const RealMatrix XtWX =
        xt::linalg::dot(xt::transpose(X), xt::linalg::dot(W, X));
    RealMatrix h1 = V + TX + xt::transpose(TX) - XtTX + XtWX * inv_4c2;
    h1 = 0.5 * (h1 + xt::transpose(h1));

    const RealMatrix R = get_r(S, S_tilde);
    hcore_ = xt::linalg::dot(xt::transpose(R), xt::linalg::dot(h1, R));
    hcore_ = 0.5 * (hcore_ + xt::transpose(hcore_));
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
