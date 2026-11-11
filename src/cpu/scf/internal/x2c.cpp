/// @file cpu/scf/internal/x2c.cpp
/// @brief One-step spin-free x2c-1e decoupling (X2cWorkspace).

#include "atomiverse/cpu/scf/internal/x2c.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

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

/// @brief Modified Dirac matrix and metric, PySCF `_get_h0_s0` / `_gen_h1_s1`.
///
/// h = [[V, T], [T, W/(4c²)−T]], M = [[S, 0], [0, T/(2c²)]].
/// `h` and `m` are zero on entry; blocks that stay zero are not written.
void fill_modified_dirac(RealMatrix& h, RealMatrix& m, const RealMatrix& v,
                         const RealMatrix& t, const RealMatrix& w,
                         const RealMatrix& s, double inv_4c2, double inv_2c2) {
    const int n = static_cast<int>(v.shape(0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h(i, j) = v(i, j);
            h(i, n + j) = t(i, j);
            h(n + i, j) = t(i, j);
            h(n + i, n + j) = w(i, j) * inv_4c2 - t(i, j);
            m(i, j) = s(i, j);
            m(n + i, n + j) = t(i, j) * inv_2c2;
        }
    }
}

/// @brief X = C_S C_L⁻¹, i.e. solve C_Lᵀ Xᵀ = C_Sᵀ (PySCF `solve(cl.T, cs.T).T`).
RealMatrix x_from_components(const RealMatrix& Cl, const RealMatrix& Cs) {
    return xt::transpose(
        xt::linalg::solve(xt::transpose(Cl), xt::transpose(Cs)));
}

/// @brief Picture-changed coefficient: large block over small block.
void write_fw(RealMatrix& c_fw, const RealMatrix& large,
              const RealMatrix& small) {
    const int n = static_cast<int>(large.shape(0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c_fw(i, j) = large(i, j);
            c_fw(n + i, j) = small(i, j);
        }
    }
}

/// @brief R¹ of JCP 135, 084114 (2011) Eq. (34). PySCF `_get_r1`.
///
/// `s_nesc0_vbas` is already v_sᵀ S̃ v_s (the caller’s `s_nesc0_vbas`).
/// `s1_ao` / `s_nesc1_ao` are still in the AO basis.
RealMatrix get_r1(const RealVector& w_sqrt, const RealMatrix& v_s,
                  const RealMatrix& s_nesc0_vbas, const RealMatrix& s1_ao,
                  const RealMatrix& s_nesc1_ao, const RealVector& wr0_sqrt,
                  const RealMatrix& vr0) {
    const int n = static_cast<int>(w_sqrt.size());
    RealVector w_invsqrt = xt::zeros<double>({n});
    RealVector wr0_invsqrt = xt::zeros<double>({n});
    for (int i = 0; i < n; ++i) {
        w_invsqrt(i) = 1.0 / w_sqrt(i);
        wr0_invsqrt(i) = 1.0 / wr0_sqrt(i);
    }

    const RealMatrix s1 =
        xt::linalg::dot(xt::transpose(v_s), xt::linalg::dot(s1_ao, v_s));
    const RealMatrix sn1 =
        xt::linalg::dot(xt::transpose(v_s), xt::linalg::dot(s_nesc1_ao, v_s));

    RealMatrix s1_sqrt = xt::zeros<double>({n, n});
    RealMatrix s1_invsqrt = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            s1_sqrt(i, j) = s1(i, j) / (w_sqrt(i) + w_sqrt(j));
            const double num = (w_invsqrt(i) * w_invsqrt(i)) * s1(i, j)
                               * (w_invsqrt(j) * w_invsqrt(j));
            s1_invsqrt(i, j) = num / -(w_invsqrt(i) + w_invsqrt(j));
        }
    }

    RealMatrix R1_mid = xt::linalg::dot(s1_invsqrt, s_nesc0_vbas);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            R1_mid(i, j) *= w_invsqrt(j);
    {
        const RealMatrix sym = R1_mid + xt::transpose(R1_mid);
        R1_mid = sym;
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            R1_mid(i, j) += w_invsqrt(i) * sn1(i, j) * w_invsqrt(j);

    R1_mid = xt::linalg::dot(xt::transpose(vr0), xt::linalg::dot(R1_mid, vr0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            R1_mid(i, j) /= -(wr0_invsqrt(i) + wr0_invsqrt(j));
            R1_mid(i, j) *= (wr0_invsqrt(i) * wr0_invsqrt(i))
                            * (wr0_invsqrt(j) * wr0_invsqrt(j));
        }
    }

    RealMatrix vr0_wr0_sqrt = vr0;
    RealMatrix vr0_s0_sqrt = xt::transpose(vr0);
    RealMatrix vr0_s0_invsqrt = xt::transpose(vr0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            vr0_wr0_sqrt(i, j) *= wr0_invsqrt(j);
            vr0_s0_sqrt(i, j) *= w_sqrt(j);
            vr0_s0_invsqrt(i, j) *= w_invsqrt(j);
        }
    }

    RealMatrix R1 = xt::linalg::dot(xt::transpose(vr0_s0_invsqrt),
                                    xt::linalg::dot(R1_mid, vr0_s0_sqrt));
    R1 = R1
         + xt::linalg::dot(s1_invsqrt,
                           xt::linalg::dot(vr0_wr0_sqrt, vr0_s0_sqrt));
    R1 = R1
         + xt::linalg::dot(
             xt::transpose(vr0_s0_invsqrt),
             xt::linalg::dot(xt::transpose(vr0_wr0_sqrt), s1_sqrt));
    return xt::linalg::dot(v_s, xt::linalg::dot(R1, xt::transpose(v_s)));
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
    fill_modified_dirac(h, M, V, T, W, S, inv_4c2, inv_2c2);

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
        X = x_from_components(Cl, Cs);
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

xt::xtensor<double, 4> x2c_hcore_gradient(
    const xt::xtensor<double, 2>& S_in, const xt::xtensor<double, 2>& T_in,
    const xt::xtensor<double, 2>& V_in, const xt::xtensor<double, 2>& W_in,
    const xt::xtensor<double, 4>& dS, const xt::xtensor<double, 4>& dT,
    const xt::xtensor<double, 4>& dV, const xt::xtensor<double, 4>& dW,
    double c) {
    const int n = static_cast<int>(S_in.shape(0));
    const auto square = [&](const xt::xtensor<double, 2>& M, const char* name) {
        if (static_cast<int>(M.shape(0)) != n
            || static_cast<int>(M.shape(1)) != n)
            throw std::invalid_argument(std::string("x2c_hcore_gradient: ")
                                        + name + " must be n×n with n="
                                        + std::to_string(n));
    };
    square(S_in, "S");
    square(T_in, "T");
    square(V_in, "V");
    square(W_in, "W");
    if (!(c > 0.0) || !std::isfinite(c))
        throw std::invalid_argument(
            "x2c_hcore_gradient: speed of light c must be finite and positive");

    int natm = -1;
    const auto check4 = [&](const xt::xtensor<double, 4>& A, const char* name) {
        if (static_cast<int>(A.shape(1)) != 3
            || static_cast<int>(A.shape(2)) != n
            || static_cast<int>(A.shape(3)) != n)
            throw std::invalid_argument(std::string("x2c_hcore_gradient: ")
                                        + name + " must be [n_atoms, 3, n, n]");
        const int na = static_cast<int>(A.shape(0));
        if (natm < 0)
            natm = na;
        else if (na != natm)
            throw std::invalid_argument(
                "x2c_hcore_gradient: derivative tensors disagree on n_atoms");
    };
    check4(dS, "dS");
    check4(dT, "dT");
    check4(dV, "dV");
    check4(dW, "dW");

    // RealMatrix is column-major; the integral blocks are row-major.
    const RealMatrix S = S_in;
    const RealMatrix T = T_in;
    const RealMatrix V = V_in;
    const RealMatrix Wp = W_in;

    const double c2 = c * c;
    const double inv_4c2 = 0.25 / c2;
    const double inv_2c2 = 0.5 / c2;
    const int n2 = 2 * n;

    // Modified Dirac matrix and metric. PySCF sfx2c1e_grad._get_h0_s0.
    RealMatrix h0 = xt::zeros<double>({n2, n2});
    RealMatrix M0 = xt::zeros<double>({n2, n2});
    fill_modified_dirac(h0, M0, V, T, Wp, S, inv_4c2, inv_2c2);

    RealVector e0;
    RealMatrix c0;
    try {
        auto sol = linalg::diagonalize_fock_generalized(h0, M0);
        e0 = std::move(sol.first);
        c0 = std::move(sol.second);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(
            std::string("x2c_hcore_gradient: Dirac generalised eigenproblem "
                        "failed: ")
            + e.what());
    }
    // PySCF takes the upper half of the ascending spectrum as the
    // positive-energy block (columns n:). The gap about −c² is O(c²).
    if (!(e0(n - 1) < -c2) || !(e0(n) > -c2))
        throw std::runtime_error(
            "x2c_hcore_gradient: Dirac spectrum does not split into n "
            "negative and n positive states (ε ≷ −c²) — refusing to build "
            "dh^X2C");

    // Positive-energy block is columns [n, 2n). PySCF `c0[:nao, nao:]`.
    const RealMatrix Cl = xt::view(c0, xt::range(0, n), xt::range(n, n2));
    const RealMatrix Cs = xt::view(c0, xt::range(n, n2), xt::range(n, n2));
    RealMatrix X;
    try {
        X = x_from_components(Cl, Cs);
    } catch (...) {
        throw std::runtime_error(
            "x2c_hcore_gradient: large-component block C_L is singular — "
            "cannot form the decoupling matrix X");
    }

    // Overlap roots for R and R¹. No 1e-14 projection: that is
    // sfx2c1e_grad._gen_first_order_quantities, not x2c._get_r.
    auto [w_s_raw, v_s_raw] = xt::linalg::eigh(S);
    RealVector w_s = w_s_raw;
    RealMatrix v_s = v_s_raw;
    RealVector w_sqrt = xt::zeros<double>({n});
    for (int i = 0; i < n; ++i) {
        if (!(w_s(i) > 1e-14))
            throw std::runtime_error(
                "x2c_hcore_gradient: overlap eigenvalue "
                + std::to_string(w_s(i))
                + " is below 1e-14 — the x2c-1e gradient "
                  "(sfx2c1e_grad._get_r1) needs a positive-definite "
                  "overlap; canonical projection of R is not this path");
        w_sqrt(i) = std::sqrt(w_s(i));
    }

    const RealMatrix TX = xt::linalg::dot(T, X);
    const RealMatrix t0x0 = TX * inv_2c2;
    const RealMatrix S_tilde = S + xt::linalg::dot(xt::transpose(X), t0x0);
    const RealMatrix s_nesc0_vbas =
        xt::linalg::dot(xt::transpose(v_s), xt::linalg::dot(S_tilde, v_s));

    RealMatrix R0_mid = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            R0_mid(i, j) = s_nesc0_vbas(i, j) / (w_sqrt(i) * w_sqrt(j));
    auto [wr0_raw, vr0_raw] = xt::linalg::eigh(R0_mid);
    RealVector wr0 = wr0_raw;
    RealMatrix vr0 = vr0_raw;
    RealVector wr0_sqrt = xt::zeros<double>({n});
    for (int i = 0; i < n; ++i) {
        if (!(wr0(i) > 1e-14))
            throw std::runtime_error(
                "x2c_hcore_gradient: NESC metric eigenvalue "
                + std::to_string(wr0(i)) + " is not positive — cannot form R¹");
        wr0_sqrt(i) = std::sqrt(wr0(i));
    }

    RealMatrix vr0_scaled = vr0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            vr0_scaled(i, j) /= wr0_sqrt(j);
    RealMatrix R0b = xt::linalg::dot(vr0_scaled, xt::transpose(vr0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            R0b(i, j) *= w_sqrt(j);
            R0b(i, j) /= w_sqrt(i);
        }
    const RealMatrix R0 =
        xt::linalg::dot(v_s, xt::linalg::dot(R0b, xt::transpose(v_s)));

    RealMatrix c_fw0 = xt::zeros<double>({n2, n});
    write_fw(c_fw0, R0, xt::linalg::dot(X, R0));
    const RealMatrix h0_fw_half = xt::linalg::dot(h0, c_fw0);

    // (e_p − e_q); degenerate pairs (|Δε| < 1e-7) get a huge denominator
    // so their response coefficient vanishes. PySCF `_gen_first_order_quantities`.
    constexpr double kDegen = 1e-7;
    RealMatrix epq = xt::zeros<double>({n2, n2});
    for (int p = 0; p < n2; ++p)
        for (int q = 0; q < n2; ++q) {
            const double dep = e0(p) - e0(q);
            epq(p, q) = (std::abs(dep) < kDegen) ? 1e200 : dep;
        }

    xt::xtensor<double, 4> out = xt::zeros<double>({natm, 3, n, n});
    for (int a = 0; a < natm; ++a) {
        for (int d = 0; d < 3; ++d) {
            const RealMatrix s1 = xt::view(dS, a, d, xt::all(), xt::all());
            const RealMatrix t1 = xt::view(dT, a, d, xt::all(), xt::all());
            const RealMatrix v1 = xt::view(dV, a, d, xt::all(), xt::all());
            const RealMatrix w1 = xt::view(dW, a, d, xt::all(), xt::all());

            RealMatrix h1 = xt::zeros<double>({n2, n2});
            RealMatrix m1 = xt::zeros<double>({n2, n2});
            fill_modified_dirac(h1, m1, v1, t1, w1, s1, inv_4c2, inv_2c2);

            const RealMatrix h1mo =
                xt::linalg::dot(xt::transpose(c0), xt::linalg::dot(h1, c0));
            const RealMatrix s1mo =
                xt::linalg::dot(xt::transpose(c0), xt::linalg::dot(m1, c0));

            // Response of the positive-energy eigenvectors only.
            RealMatrix c1_mo = xt::zeros<double>({n2, n});
            for (int p = 0; p < n2; ++p) {
                for (int i = 0; i < n; ++i) {
                    const int q = n + i;
                    const double f = h1mo(p, q) - s1mo(p, q) * e0(q);
                    c1_mo(p, i) = f / -epq(p, q);
                }
            }
            const RealMatrix c1_ao = xt::linalg::dot(c0, c1_mo);
            const RealMatrix cl1 = xt::view(c1_ao, xt::range(0, n), xt::all());
            const RealMatrix cs1 = xt::view(c1_ao, xt::range(n, n2), xt::all());
            const RealMatrix tmp = cs1 - xt::linalg::dot(X, cl1);
            RealMatrix x1;
            try {
                x1 = x_from_components(Cl, tmp);
            } catch (...) {
                throw std::runtime_error(
                    "x2c_hcore_gradient: large-component block C_L is "
                    "singular while solving for X¹");
            }

            RealMatrix s_nesc1 = xt::linalg::dot(xt::transpose(x1), t0x0);
            {
                const RealMatrix sym = s_nesc1 + xt::transpose(s_nesc1);
                s_nesc1 = sym;
            }
            const RealMatrix m1_ss =
                xt::view(m1, xt::range(n, n2), xt::range(n, n2));
            s_nesc1 =
                s_nesc1
                + xt::linalg::dot(xt::transpose(X), xt::linalg::dot(m1_ss, X));
            s_nesc1 = s_nesc1 + s1;

            const RealMatrix R1 =
                get_r1(w_sqrt, v_s, s_nesc0_vbas, s1, s_nesc1, wr0_sqrt, vr0);

            RealMatrix c_fw1 = xt::zeros<double>({n2, n});
            const RealMatrix small =
                xt::linalg::dot(X, R1) + xt::linalg::dot(x1, R0);
            write_fw(c_fw1, R1, small);

            // hfw¹ = c_fw¹† h c_fw⁰ + h.c. + c_fw⁰† h¹ c_fw⁰.
            RealMatrix hfw = xt::linalg::dot(xt::transpose(c_fw1), h0_fw_half);
            {
                const RealMatrix sym = hfw + xt::transpose(hfw);
                hfw = sym;
            }
            hfw = hfw
                  + xt::linalg::dot(xt::transpose(c_fw0),
                                    xt::linalg::dot(h1, c_fw0));
            hfw = 0.5 * (hfw + xt::transpose(hfw));
            xt::view(out, a, d, xt::all(), xt::all()) = hfw;
        }
    }
    return out;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
