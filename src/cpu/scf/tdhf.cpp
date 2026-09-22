/// @file cpu/scf/tdhf.cpp
/// @brief TDHFPack (TDA + TDHF/RPA lowest-roots solvers over the link-08
///        OrbitalHessian A/B kernel) and the RHF::tda()/RHF::tdhf() method
///        surface (stability pre-check, transition dipoles, oscillator
///        strengths). @see internal/tdhf_pack.h, tdhf.h.

#include "atomiverse/cpu/scf/internal/tdhf_pack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/scf/basis_eval.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/alda_kernel.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/one_electron.h"
#include "atomiverse/cpu/scf/internal/stability.h"
#include "atomiverse/cpu/scf/internal/tda_gradient.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/rks.h"
#include "atomiverse/cpu/scf/tdhf.h"
#include "atomiverse/cpu/scf/uhf.h"
#include "atomiverse/cpu/scf/uks.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cpu/scf/xc_grid_engine.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

/// Negative-ω² guard [Hartree²]: a reduced-RPA eigenvalue below −kOmega2Tol
/// is a genuine channel instability (imaginary ω) and throws; the band
/// [−kOmega2Tol, +kOmega2Tol] is a numerically-zero root whose ω^{∓1/2}
/// back-substitution is ill-conditioned and throws too. No fallback.
constexpr double kOmega2Tol = 1e-10;

/// Positive-definiteness floor on (A − B) eigenvalues [Hartree]: the
/// Hermitian reduction needs (A − B)^{±1/2}.
constexpr double kAmBFloor = 1e-10;

/// Davidson residual threshold ‖r‖₂ [Hartree] per root.
constexpr double kDavidsonResTol = 1e-9;

/// Davidson expansion-vector survival threshold after orthogonalization.
constexpr double kNewDirTol = 1e-10;

double vdot(const RealVector& a, const RealVector& b) {
    double s = 0.0;
    for (std::size_t r = 0; r < a.size(); ++r)
        s += a(r) * b(r);
    return s;
}

/// Two modified-Gram-Schmidt passes of t against the orthonormal set V;
/// returns the surviving norm (0 ⇒ no new direction) and normalizes t.
double orthonormalize_against(RealVector& t, const std::vector<RealVector>& V) {
    for (int pass = 0; pass < 2; ++pass)
        for (const RealVector& v : V) {
            const double proj = vdot(v, t);
            for (std::size_t r = 0; r < t.size(); ++r)
                t(r) -= proj * v(r);
        }
    const double tn = std::sqrt(vdot(t, t));
    if (tn < kNewDirTol)
        return 0.0;
    for (std::size_t r = 0; r < t.size(); ++r)
        t(r) /= tn;
    return tn;
}

/// Deterministic Davidson seeds: coordinate vectors of the k smallest
/// entries of `diag` (ties break to the lowest index).
std::vector<int> seed_indices(const RealVector& diag, int k) {
    const int n = static_cast<int>(diag.size());
    std::vector<int> order(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        order[static_cast<std::size_t>(i)] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](int p, int q) { return diag(p) < diag(q); });
    order.resize(static_cast<std::size_t>(k));
    return order;
}

/// (A−B)^{p} for the Hermitian RPA reduction (p = ±1/2). The shared
/// `linalg::sym_power` CLAMPS eigenvalues at the floor, but the reduction
/// needs (A−B) genuinely positive definite — preserve the pre-refactor
/// throw-on-λ_min≤floor guard here, then defer the power to the shared
/// primitive (bit-identical on a PD reference; the extra symmetrization the
/// old copy did is absorbed downstream when M = TᵀMpT is symmetrized).
RealMatrix checked_sym_power(const RealMatrix& M, double p, double floor,
                             const char* what) {
    auto [s, U] = xt::linalg::eigh(M);
    (void)U;
    if (s(0) <= floor)
        throw std::runtime_error(
            std::string(what)
            + ": (A - B) is not positive definite "
              "(lowest eigenvalue = "
            + std::to_string(s(0))
            + " Ha) — the Hermitian RPA reduction is undefined for this "
              "reference (no fallback)");
    return linalg::sym_power(M, p, floor, what);
}

/// The reduced-RPA ω² gate: negative beyond tolerance ⇒ channel instability
/// (imaginary ω), |ω²| ≈ 0 ⇒ ill-conditioned back-substitution. Throws.
void check_omega2(double w2, int root) {
    if (w2 < -kOmega2Tol)
        throw std::runtime_error(
            "TDHFPack::rpa: reduced RPA eigenvalue omega^2 = "
            + std::to_string(w2) + " Ha^2 (root " + std::to_string(root)
            + ") is negative — the channel's (A + B) block is unstable and "
              "the full RPA excitation energy is imaginary; use tda() or "
              "reconverge the reference along the instability mode "
              "(no fallback)");
    if (w2 <= kOmega2Tol)
        throw std::runtime_error(
            "TDHFPack::rpa: reduced RPA eigenvalue omega^2 = "
            + std::to_string(w2) + " Ha^2 (root " + std::to_string(root)
            + ") is numerically zero — the (X, Y) back-substitution is "
              "ill-conditioned (no fallback)");
}

} // namespace

void TDHFPack::check_n_states_(int n_states) const {
    if (n_states < 1 || n_states > hessian_.n_ov())
        throw std::invalid_argument(
            "TDHFPack: n_states = " + std::to_string(n_states)
            + " outside [1, n_ov = " + std::to_string(hessian_.n_ov())
            + "] — the rotation space has only n_ov roots (no fallback)");
}

TDHFPack::Roots TDHFPack::tda(int n_states) const {
    check_n_states_(n_states);
    return (hessian_.n_ov() <= dense_threshold) ? tda_dense_(n_states)
                                                : tda_davidson_(n_states);
}

TDHFPack::Roots TDHFPack::rpa(int n_states) const {
    check_n_states_(n_states);
    return (hessian_.n_ov() <= dense_threshold) ? rpa_dense_(n_states)
                                                : rpa_davidson_(n_states);
}

TDHFPack::Roots TDHFPack::tda_dense_(int n_states) const {
    const RealMatrix A =
        0.5 * (hessian_.dense_plus(ch_) + hessian_.dense_minus(ch_));
    auto [w, U] = xt::linalg::eigh(A); // ascending
    const std::size_t n = A.shape()[0];
    const std::size_t k = static_cast<std::size_t>(n_states);
    Roots out;
    out.omega = xt::zeros<double>({k});
    out.X = xt::zeros<double>({k, n});
    out.Y = xt::zeros<double>({k, n});
    for (std::size_t s = 0; s < k; ++s) {
        out.omega(s) = w(s);
        for (std::size_t r = 0; r < n; ++r)
            out.X(s, r) = U(r, s);
    }
    return out;
}

TDHFPack::Roots TDHFPack::rpa_dense_(int n_states) const {
    const RealMatrix Mp = hessian_.dense_plus(ch_);
    const RealMatrix Mm = hessian_.dense_minus(ch_);
    const RealMatrix T = checked_sym_power(Mm, 0.5, kAmBFloor, "TDHFPack::rpa");
    const RealMatrix Ti =
        checked_sym_power(Mm, -0.5, kAmBFloor, "TDHFPack::rpa");

    RealMatrix M = xt::linalg::dot(T, RealMatrix(xt::linalg::dot(Mp, T)));
    const std::size_t n = M.shape()[0];
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = r + 1; c < n; ++c) {
            const double v = 0.5 * (M(r, c) + M(c, r));
            M(r, c) = v;
            M(c, r) = v;
        }
    auto [w2, Z] = xt::linalg::eigh(M); // ascending ω²

    const std::size_t k = static_cast<std::size_t>(n_states);
    Roots out;
    out.omega = xt::zeros<double>({k});
    out.X = xt::zeros<double>({k, n});
    out.Y = xt::zeros<double>({k, n});
    for (std::size_t s = 0; s < k; ++s) {
        check_omega2(w2(s), static_cast<int>(s));
        const double omega = std::sqrt(w2(s));
        out.omega(s) = omega;
        // (X+Y) = ω^{-1/2} T z, (X−Y) = ω^{1/2} T^{-1} z ⇒ XᵀX − YᵀY = zᵀz = 1.
        for (std::size_t r = 0; r < n; ++r) {
            double xpy = 0.0, xmy = 0.0;
            for (std::size_t c = 0; c < n; ++c) {
                xpy += T(r, c) * Z(c, s);
                xmy += Ti(r, c) * Z(c, s);
            }
            xpy /= std::sqrt(omega);
            xmy *= std::sqrt(omega);
            out.X(s, r) = 0.5 * (xpy + xmy);
            out.Y(s, r) = 0.5 * (xpy - xmy);
        }
    }
    return out;
}

TDHFPack::Roots TDHFPack::tda_davidson_(int n_states) const {
    const int n = hessian_.n_ov();
    const int k = n_states;
    const RealVector diag = hessian_.diagonal_plus(ch_);
    const int max_subspace = std::min(n, std::max(60, 12 * k));
    const int max_iter = 300;

    std::vector<RealVector> V, W; // orthonormal subspace and A·V products
    for (int idx : seed_indices(diag, k)) {
        RealVector b = xt::zeros<double>({static_cast<std::size_t>(n)});
        b(idx) = 1.0;
        V.push_back(b);
        W.push_back(
            0.5 * (hessian_.sigma_plus(b, ch_) + hessian_.sigma_minus(b, ch_)));
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        const int m = static_cast<int>(V.size());
        RealMatrix H = xt::zeros<double>(
            {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
        for (int p = 0; p < m; ++p)
            for (int q = p; q < m; ++q) {
                const double s = 0.5
                                 * (vdot(V[static_cast<std::size_t>(p)],
                                         W[static_cast<std::size_t>(q)])
                                    + vdot(V[static_cast<std::size_t>(q)],
                                           W[static_cast<std::size_t>(p)]));
                H(p, q) = s;
                H(q, p) = s;
            }
        auto [w, S] = xt::linalg::eigh(H);

        // Ritz vectors + residuals for the k lowest roots.
        std::vector<RealVector> x(static_cast<std::size_t>(k)),
            res(static_cast<std::size_t>(k));
        bool all_conv = true;
        for (int s = 0; s < k; ++s) {
            RealVector xs = xt::zeros<double>({static_cast<std::size_t>(n)});
            RealVector as = xt::zeros<double>({static_cast<std::size_t>(n)});
            for (int p = 0; p < m; ++p) {
                const double c = S(p, s);
                for (int r = 0; r < n; ++r) {
                    xs(r) += c * V[static_cast<std::size_t>(p)](r);
                    as(r) += c * W[static_cast<std::size_t>(p)](r);
                }
            }
            RealVector rs = as - w(s) * xs;
            if (std::sqrt(vdot(rs, rs)) >= kDavidsonResTol)
                all_conv = false;
            x[static_cast<std::size_t>(s)] = std::move(xs);
            res[static_cast<std::size_t>(s)] = std::move(rs);
        }

        bool expanded = false;
        bool cap_blocked = false;
        if (!all_conv && m < n) {
            for (int s = 0; s < k; ++s) {
                if (std::sqrt(vdot(res[static_cast<std::size_t>(s)],
                                   res[static_cast<std::size_t>(s)]))
                    < kDavidsonResTol)
                    continue;
                if (static_cast<int>(V.size()) >= max_subspace) {
                    cap_blocked = true;
                    break;
                }
                RealVector t = xt::zeros<double>({static_cast<std::size_t>(n)});
                for (int r = 0; r < n; ++r) {
                    double denom = w(s) - diag(r);
                    if (std::abs(denom) < 1e-8)
                        denom = (denom < 0.0) ? -1e-8 : 1e-8;
                    t(r) = res[static_cast<std::size_t>(s)](r) / denom;
                }
                if (orthonormalize_against(t, V) > 0.0) {
                    V.push_back(t);
                    W.push_back(0.5
                                * (hessian_.sigma_plus(t, ch_)
                                   + hessian_.sigma_minus(t, ch_)));
                    expanded = true;
                }
            }
        }
        if (!all_conv && cap_blocked && !expanded)
            throw std::runtime_error(
                "TDHFPack::tda: Davidson did not converge within the "
                "subspace cap ("
                + std::to_string(max_subspace)
                + " vectors, n_ov = " + std::to_string(n) + ") — no fallback");

        if (all_conv || m == n || !expanded) {
            Roots out;
            out.omega = xt::zeros<double>({static_cast<std::size_t>(k)});
            out.X = xt::zeros<double>(
                {static_cast<std::size_t>(k), static_cast<std::size_t>(n)});
            out.Y = xt::zeros<double>(
                {static_cast<std::size_t>(k), static_cast<std::size_t>(n)});
            for (int s = 0; s < k; ++s) {
                out.omega(s) = w(s);
                RealVector& xs = x[static_cast<std::size_t>(s)];
                const double xn = std::sqrt(vdot(xs, xs));
                for (int r = 0; r < n; ++r)
                    out.X(s, r) = xs(r) / xn;
            }
            return out;
        }
    }
    throw std::runtime_error(
        "TDHFPack::tda: Davidson did not converge in 300 iterations "
        "(n_ov = "
        + std::to_string(hessian_.n_ov()) + ") — no fallback");
}

TDHFPack::Roots TDHFPack::rpa_davidson_(int n_states) const {
    // Reduced-subspace RPA (Stratmann-Scuseria-Frisch 1998): expand ONE
    // orthonormal subspace V, project P = Vᵀ(A+B)V and Q = Vᵀ(A−B)V via σ±
    // products, and solve the ω² problem Q^{1/2} P Q^{1/2} z = ω² z densely
    // in the m-dimensional subspace only.
    const int n = hessian_.n_ov();
    const int k = n_states;
    const RealVector diag = hessian_.diagonal_plus(ch_);
    const int max_subspace = std::min(n, std::max(60, 12 * k));
    const int max_iter = 300;

    std::vector<RealVector> V, Wp, Wm; // subspace, σ₊·V, σ₋·V
    for (int idx : seed_indices(diag, k)) {
        RealVector b = xt::zeros<double>({static_cast<std::size_t>(n)});
        b(idx) = 1.0;
        V.push_back(b);
        Wp.push_back(hessian_.sigma_plus(b, ch_));
        Wm.push_back(hessian_.sigma_minus(b, ch_));
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        const int m = static_cast<int>(V.size());
        RealMatrix P = xt::zeros<double>(
            {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
        RealMatrix Q = P;
        for (int p = 0; p < m; ++p)
            for (int q = p; q < m; ++q) {
                const double sp = 0.5
                                  * (vdot(V[static_cast<std::size_t>(p)],
                                          Wp[static_cast<std::size_t>(q)])
                                     + vdot(V[static_cast<std::size_t>(q)],
                                            Wp[static_cast<std::size_t>(p)]));
                const double sm = 0.5
                                  * (vdot(V[static_cast<std::size_t>(p)],
                                          Wm[static_cast<std::size_t>(q)])
                                     + vdot(V[static_cast<std::size_t>(q)],
                                            Wm[static_cast<std::size_t>(p)]));
                P(p, q) = sp;
                P(q, p) = sp;
                Q(p, q) = sm;
                Q(q, p) = sm;
            }
        const RealMatrix Qh =
            checked_sym_power(Q, 0.5, kAmBFloor, "TDHFPack::rpa(davidson)");
        const RealMatrix Qih =
            checked_sym_power(Q, -0.5, kAmBFloor, "TDHFPack::rpa(davidson)");
        RealMatrix M = xt::linalg::dot(Qh, RealMatrix(xt::linalg::dot(P, Qh)));
        for (int r = 0; r < m; ++r)
            for (int c = r + 1; c < m; ++c) {
                const double v = 0.5 * (M(r, c) + M(c, r));
                M(r, c) = v;
                M(c, r) = v;
            }
        auto [w2, Z] = xt::linalg::eigh(M);

        // Back-substituted Ritz pairs + the two residuals per root:
        //   r₁ = (A+B)(X+Y) − ω(X−Y),  r₂ = (A−B)(X−Y) − ω(X+Y).
        std::vector<double> omega(static_cast<std::size_t>(k));
        std::vector<RealVector> xpy(static_cast<std::size_t>(k)),
            xmy(static_cast<std::size_t>(k)), r1(static_cast<std::size_t>(k)),
            r2(static_cast<std::size_t>(k));
        bool all_conv = true;
        for (int s = 0; s < k; ++s) {
            check_omega2(w2(s), s);
            const double om = std::sqrt(w2(s));
            omega[static_cast<std::size_t>(s)] = om;
            // Subspace coordinates: (X+Y) = V·(Q^{1/2} z / √ω),
            // (X−Y) = V·(Q^{-1/2} z · √ω); (X+Y)ᵀ(X−Y) = zᵀz = 1.
            RealVector c = xt::zeros<double>({static_cast<std::size_t>(m)});
            RealVector d = c;
            for (int p = 0; p < m; ++p) {
                double ch_c = 0.0, ch_d = 0.0;
                for (int q = 0; q < m; ++q) {
                    ch_c += Qh(p, q) * Z(q, s);
                    ch_d += Qih(p, q) * Z(q, s);
                }
                c(p) = ch_c / std::sqrt(om);
                d(p) = ch_d * std::sqrt(om);
            }
            RealVector xp = xt::zeros<double>({static_cast<std::size_t>(n)});
            RealVector xm = xp, s1 = xp, s2 = xp;
            for (int p = 0; p < m; ++p)
                for (int r = 0; r < n; ++r) {
                    xp(r) += c(p) * V[static_cast<std::size_t>(p)](r);
                    xm(r) += d(p) * V[static_cast<std::size_t>(p)](r);
                    s1(r) += c(p) * Wp[static_cast<std::size_t>(p)](r);
                    s2(r) += d(p) * Wm[static_cast<std::size_t>(p)](r);
                }
            RealVector rr1 = s1 - om * xm;
            RealVector rr2 = s2 - om * xp;
            if (std::sqrt(vdot(rr1, rr1)) >= kDavidsonResTol
                || std::sqrt(vdot(rr2, rr2)) >= kDavidsonResTol)
                all_conv = false;
            xpy[static_cast<std::size_t>(s)] = std::move(xp);
            xmy[static_cast<std::size_t>(s)] = std::move(xm);
            r1[static_cast<std::size_t>(s)] = std::move(rr1);
            r2[static_cast<std::size_t>(s)] = std::move(rr2);
        }

        bool expanded = false;
        bool cap_blocked = false;
        if (!all_conv && m < n) {
            for (int s = 0; s < k; ++s) {
                for (const RealVector* rv :
                     {&r1[static_cast<std::size_t>(s)],
                      &r2[static_cast<std::size_t>(s)]}) {
                    if (std::sqrt(vdot(*rv, *rv)) < kDavidsonResTol)
                        continue;
                    if (static_cast<int>(V.size()) >= max_subspace) {
                        cap_blocked = true;
                        break;
                    }
                    RealVector t =
                        xt::zeros<double>({static_cast<std::size_t>(n)});
                    for (int r = 0; r < n; ++r) {
                        double denom =
                            omega[static_cast<std::size_t>(s)] - diag(r);
                        if (std::abs(denom) < 1e-8)
                            denom = (denom < 0.0) ? -1e-8 : 1e-8;
                        t(r) = (*rv)(r) / denom;
                    }
                    if (orthonormalize_against(t, V) > 0.0) {
                        V.push_back(t);
                        Wp.push_back(hessian_.sigma_plus(t, ch_));
                        Wm.push_back(hessian_.sigma_minus(t, ch_));
                        expanded = true;
                    }
                }
            }
        }

        if (!all_conv && cap_blocked && !expanded)
            throw std::runtime_error(
                "TDHFPack::rpa: Davidson did not converge within the "
                "subspace cap ("
                + std::to_string(max_subspace)
                + " vectors, n_ov = " + std::to_string(n) + ") — no fallback");

        if (all_conv || m == n || !expanded) {
            Roots out;
            out.omega = xt::zeros<double>({static_cast<std::size_t>(k)});
            out.X = xt::zeros<double>(
                {static_cast<std::size_t>(k), static_cast<std::size_t>(n)});
            out.Y = xt::zeros<double>(
                {static_cast<std::size_t>(k), static_cast<std::size_t>(n)});
            for (int s = 0; s < k; ++s) {
                out.omega(s) = omega[static_cast<std::size_t>(s)];
                for (int r = 0; r < n; ++r) {
                    out.X(s, r) = 0.5
                                  * (xpy[static_cast<std::size_t>(s)](r)
                                     + xmy[static_cast<std::size_t>(s)](r));
                    out.Y(s, r) = 0.5
                                  * (xpy[static_cast<std::size_t>(s)](r)
                                     - xmy[static_cast<std::size_t>(s)](r));
                }
            }
            return out;
        }
    }
    throw std::runtime_error(
        "TDHFPack::rpa: Davidson did not converge in 300 iterations "
        "(n_ov = "
        + std::to_string(hessian_.n_ov()) + ") — no fallback");
}

} // namespace internal

// ── RHF method surface: tda() / tdhf() beside stability() ────────────────

namespace {

internal::Channel internal_channel(ExcitationChannel ch) {
    return (ch == ExcitationChannel::triplet) ? internal::Channel::triplet
                                              : internal::Channel::singlet;
}

/// Semi-local \(f^{\mathrm{xc}}\) kernel via central FD of \(V_{\mathrm{xc}}\)
/// on the KS grid (closed-shell singlet). Same convention as hybrid TDA /
/// full LR-TDDFT (PySCF get_ab hybrid path). Returns K such that A and B
/// both receive K under the adiabatic approximation.
RealMatrix fxc_kernel_fd(const SlaterDeterminant& det,
                         const internal::IntegralView& pack, XCType xc,
                         GridLevel grid_level, int n_radial, int n_angular) {
    if (!det.restricted())
        throw std::runtime_error(
            "fxc_kernel_fd: restricted closed-shell determinant required");
    if (pack.shells.empty())
        throw std::runtime_error(
            "fxc_kernel_fd: internal::IntegralWorkspace has no shells");

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& C = det.coefficients(Spin::alpha);
    const RealVector& f = det.occupations(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    std::vector<int> occ, virt;
    for (int p = 0; p < n_mo; ++p)
        (f(p) > 0.0 ? occ : virt).push_back(p);
    const int no = static_cast<int>(occ.size());
    const int nv = static_cast<int>(virt.size());
    const std::size_t nov =
        static_cast<std::size_t>(no) * static_cast<std::size_t>(nv);

    RealMatrix C_work = C;
    RealMatrix Pa0 = det.density(Spin::alpha);
    if (pack.spherical()) {
        const auto& c2s = pack.c2s;
        C_work = xt::linalg::dot(c2s, C);
        Pa0 = xt::linalg::dot(
            c2s, RealMatrix(xt::linalg::dot(Pa0, xt::transpose(c2s))));
    }
    const int n_basis = static_cast<int>(C_work.shape()[0]);

    const MolecularGrid grid(pack.atoms.data(),
                             static_cast<int>(pack.atoms.size()), grid_level,
                             n_radial, n_angular);

    // Central FD step on the density matrix [a.u.]; 1e-4 matches PySCF
    // kernel elements to ~1e-9 Ha on H2/STO-3G.
    constexpr double kEps = 1e-4;

    auto orbital_product = [&](int i_mo, int a_mo) {
        RealMatrix dm = xt::zeros<double>({static_cast<std::size_t>(n_basis),
                                           static_cast<std::size_t>(n_basis)});
        for (int mu = 0; mu < n_basis; ++mu)
            for (int nu = 0; nu < n_basis; ++nu)
                dm(mu, nu) = C_work(mu, static_cast<std::size_t>(i_mo))
                             * C_work(nu, static_cast<std::size_t>(a_mo));
        return dm;
    };

    RealMatrix K = xt::zeros<double>({nov, nov});
    for (int a = 0; a < nv; ++a) {
        for (int i = 0; i < no; ++i) {
            const int p = a * no + i;
            const int i_mo = occ[static_cast<std::size_t>(i)];
            const int a_mo = virt[static_cast<std::size_t>(a)];
            const RealMatrix dm = orbital_product(i_mo, a_mo);
            // Hermitized product P^{ia} = ½(C_i C_aᵀ + C_a C_iᵀ).
            // Closed-shell singlet: δP_α = δP_β = P^{ia}.
            RealMatrix dm_sym = 0.5 * (dm + xt::transpose(dm));

            RealMatrix Pa_p = Pa0 + kEps * dm_sym;
            RealMatrix Pa_m = Pa0 - kEps * dm_sym;
            const XCGridResult xp =
                evaluate_xc_batched(grid, pack.shells, n_basis, Pa_p, Pa_p, xc);
            const XCGridResult xm =
                evaluate_xc_batched(grid, pack.shells, n_basis, Pa_m, Pa_m, xc);

            RealMatrix dVa = (xp.Vxc_alpha - xm.Vxc_alpha) / (2.0 * kEps);

            for (int b = 0; b < nv; ++b) {
                for (int j = 0; j < no; ++j) {
                    const int q = b * no + j;
                    const int j_mo = occ[static_cast<std::size_t>(j)];
                    const int b_mo = virt[static_cast<std::size_t>(b)];
                    const RealMatrix dm_q = orbital_product(j_mo, b_mo);
                    double axc = 0.0;
                    for (int mu = 0; mu < n_basis; ++mu)
                        for (int nu = 0; nu < n_basis; ++nu)
                            axc += dVa(mu, nu) * dm_q(mu, nu);
                    K(static_cast<std::size_t>(p),
                      static_cast<std::size_t>(q)) = axc;
                }
            }
        }
    }
    return K;
}

/// Dense Hermitian RPA from explicit A, B (Casida full LR-TDDFT). Same
/// reduction as TDHFPack::rpa_dense_: (A−B)^{±1/2} factors + ω² gate.
internal::TDHFPack::Roots rpa_from_ab(const RealMatrix& A, const RealMatrix& B,
                                      int n_states) {
    constexpr double kFloor = 1e-10;
    constexpr double kW2Tol = 1e-10;
    const std::size_t n = A.shape()[0];
    if (A.shape()[1] != n || B.shape()[0] != n || B.shape()[1] != n)
        throw std::invalid_argument(
            "rpa_from_ab: A/B must be square and equal");
    if (n_states < 1 || static_cast<std::size_t>(n_states) > n)
        throw std::invalid_argument("rpa_from_ab: n_states outside [1, n]");

    RealMatrix Mp = A + B;
    RealMatrix Mm = A - B;
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = r + 1; c < n; ++c) {
            const double vp = 0.5 * (Mp(r, c) + Mp(c, r));
            Mp(r, c) = vp;
            Mp(c, r) = vp;
            const double vm = 0.5 * (Mm(r, c) + Mm(c, r));
            Mm(r, c) = vm;
            Mm(c, r) = vm;
        }

    // (A−B)^{p} via the shared linalg::sym_power, guarding positive-
    // definiteness first (the primitive CLAMPS; keep the throw so behavior is
    // unchanged — see the file-scope sym_power note above).
    auto make_power = [&](const RealMatrix& M, double p) {
        auto [s, U] = xt::linalg::eigh(M);
        (void)U;
        if (s(0) <= kFloor)
            throw std::runtime_error(
                "RKS::tddft: (A - B) is not positive definite "
                "(lowest eigenvalue = "
                + std::to_string(s(0))
                + " Ha) — Hermitian RPA reduction undefined (no fallback)");
        return linalg::sym_power(M, p, kFloor, "RKS::tddft");
    };

    const RealMatrix T = make_power(Mm, 0.5);
    const RealMatrix Ti = make_power(Mm, -0.5);
    RealMatrix M = xt::linalg::dot(T, RealMatrix(xt::linalg::dot(Mp, T)));
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = r + 1; c < n; ++c) {
            const double v = 0.5 * (M(r, c) + M(c, r));
            M(r, c) = v;
            M(c, r) = v;
        }
    auto [w2, Z] = xt::linalg::eigh(M);

    const std::size_t k = static_cast<std::size_t>(n_states);
    internal::TDHFPack::Roots out;
    out.omega = xt::zeros<double>({k});
    out.X = xt::zeros<double>({k, n});
    out.Y = xt::zeros<double>({k, n});
    for (std::size_t s = 0; s < k; ++s) {
        if (w2(s) < -kW2Tol)
            throw std::runtime_error(
                "RKS::tddft: reduced RPA omega^2 = " + std::to_string(w2(s))
                + " Ha^2 (root " + std::to_string(s)
                + ") is negative — imaginary excitation energy (no fallback)");
        if (w2(s) <= kW2Tol)
            throw std::runtime_error("RKS::tddft: reduced RPA omega^2 = "
                                     + std::to_string(w2(s)) + " Ha^2 (root "
                                     + std::to_string(s)
                                     + ") is numerically zero (no fallback)");
        const double omega = std::sqrt(w2(s));
        out.omega(s) = omega;
        for (std::size_t r = 0; r < n; ++r) {
            double xpy = 0.0, xmy = 0.0;
            for (std::size_t c = 0; c < n; ++c) {
                xpy += T(r, c) * Z(c, s);
                xmy += Ti(r, c) * Z(c, s);
            }
            xpy /= std::sqrt(omega);
            xmy *= std::sqrt(omega);
            out.X(s, r) = 0.5 * (xpy + xmy);
            out.Y(s, r) = 0.5 * (xpy - xmy);
        }
    }
    return out;
}

/// Package pack roots into the public result: eV energies for both
/// channels; singlet channel adds length-gauge transition dipoles through
/// the link-03 MultipoleIntegrals kernel (μ_0n = √2 Σ_ia (X+Y)_ia ⟨i|r|a⟩)
/// and f = (2/3)·ω[Ha]·|μ|²; the triplet channel is spin-forbidden (μ = 0,
/// f = 0). Two callers (tda / tdhf) — second-caller extraction rule.
ExcitedStatesResult
excited_states_result(const SlaterDeterminant& det,
                      const internal::IntegralView& pack,
                      const internal::TDHFPack::Roots& roots,
                      ExcitationChannel channel, bool tda_flag) {
    const std::size_t k = roots.omega.size();
    ExcitedStatesResult out;
    out.channel = channel;
    out.tda = tda_flag;
    out.energy_ev.resize(k);
    out.oscillator_strength.assign(k, 0.0);
    out.transition_dipole_au = xt::zeros<double>({k, std::size_t(3)});
    for (std::size_t s = 0; s < k; ++s)
        out.energy_ev[s] = roots.omega(s) * units::kHartreeInEv;
    if (channel == ExcitationChannel::triplet)
        return out; // spin-forbidden from the singlet ground state

    // AO dipole matrices (engine Cartesian convention; spherical bases
    // sandwich with the pack's c2s), origin at 0 — transition moments
    // between orthogonal states are origin-independent.
    const double origin[3] = {0.0, 0.0, 0.0};
    const MultipoleIntegrals mi{pack.shells.data(),
                                static_cast<int>(pack.shells.size()), origin};

    // occ/virt split by occupation (f = 1 ⇔ occupied) — the same
    // convention as OrbitalHessian's composite index p = a·n_occ + i.
    const RealMatrix& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const RealVector& f = det.occupations(SlaterDeterminant::Spin::alpha);
    const std::size_t n_ao = C.shape()[0];
    const int n_mo = static_cast<int>(C.shape()[1]);
    std::vector<int> occ, virt;
    for (int p = 0; p < n_mo; ++p)
        (f(p) > 0.0 ? occ : virt).push_back(p);
    const int no = static_cast<int>(occ.size());
    const int nv = static_cast<int>(virt.size());

    RealMatrix Co = xt::zeros<double>({n_ao, static_cast<std::size_t>(no)});
    RealMatrix Cv = xt::zeros<double>({n_ao, static_cast<std::size_t>(nv)});
    for (int i = 0; i < no; ++i)
        for (std::size_t mu = 0; mu < n_ao; ++mu)
            Co(mu, static_cast<std::size_t>(i)) = C(
                mu, static_cast<std::size_t>(occ[static_cast<std::size_t>(i)]));
    for (int a = 0; a < nv; ++a)
        for (std::size_t mu = 0; mu < n_ao; ++mu)
            Cv(mu, static_cast<std::size_t>(a)) =
                C(mu,
                  static_cast<std::size_t>(virt[static_cast<std::size_t>(a)]));

    for (int axis = 0; axis < 3; ++axis) {
        xt::xtensor<double, 2> D = mi.dipole(axis);
        if (pack.spherical()) {
            const auto& c2s = pack.c2s;
            D = xt::linalg::dot(
                xt::transpose(c2s),
                xt::xtensor<double, 2>(xt::linalg::dot(D, c2s)));
        }
        // ⟨φ_i|r_axis|φ_a⟩ = (C_occᵀ D C_virt)_(i,a).
        const RealMatrix d_mo = xt::linalg::dot(
            xt::transpose(Co), RealMatrix(xt::linalg::dot(D, Cv)));
        for (std::size_t s = 0; s < k; ++s) {
            double mu_s = 0.0;
            for (int a = 0; a < nv; ++a)
                for (int i = 0; i < no; ++i) {
                    const std::size_t p = static_cast<std::size_t>(a * no + i);
                    mu_s += (roots.X(s, p) + roots.Y(s, p)) * d_mo(i, a);
                }
            out.transition_dipole_au(s, axis) = std::sqrt(2.0) * mu_s;
        }
    }
    for (std::size_t s = 0; s < k; ++s) {
        const double mu2 =
            out.transition_dipole_au(s, 0) * out.transition_dipole_au(s, 0)
            + out.transition_dipole_au(s, 1) * out.transition_dipole_au(s, 1)
            + out.transition_dipole_au(s, 2) * out.transition_dipole_au(s, 2);
        out.oscillator_strength[s] = (2.0 / 3.0) * roots.omega(s) * mu2;
    }
    return out;
}

} // namespace

ExcitedStatesResult RHF::tda(int n_states, ExcitationChannel channel) const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::tda: no SCF result — call run(frame) first");
    const SlaterDeterminant& det = determinant();
    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();
    const internal::OrbitalHessian hessian{det, eri};
    const internal::TDHFPack tdhf_pack{hessian, internal_channel(channel)};
    return excited_states_result(det, pack, tdhf_pack.tda(n_states), channel,
                                 /*tda_flag=*/true);
}

ExcitedStatesResult RKS::tda(int n_states, ExcitationChannel channel) const {
    if (!has_result())
        throw std::runtime_error(
            "RKS::tda: no SCF result — call run(frame) first");
    const SlaterDeterminant& det = determinant();
    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();
    const internal::OrbitalHessian hessian{det, eri};
    const internal::Channel ch = internal_channel(channel);

    // Package dense A-matrix roots into ExcitedStatesResult (shared by the
    // LSDA ALDA and hybrid f^xc paths).
    auto package_dense_tda = [&](const RealMatrix& A, bool alda_flag) {
        auto [w, U] = xt::linalg::eigh(A);
        const std::size_t n = A.shape()[0];
        const std::size_t k = static_cast<std::size_t>(n_states);
        if (n_states < 1 || static_cast<std::size_t>(n_states) > n)
            throw std::invalid_argument("RKS::tda: n_states outside [1, n_ov]");
        internal::TDHFPack::Roots roots;
        roots.omega = xt::zeros<double>({k});
        roots.X = xt::zeros<double>({k, n});
        roots.Y = xt::zeros<double>({k, n});
        for (std::size_t s = 0; s < k; ++s) {
            roots.omega(s) = w(s);
            for (std::size_t r = 0; r < n; ++r)
                roots.X(s, r) = U(r, s);
        }
        auto out = excited_states_result(det, pack, roots, channel, true);
        out.alda = alda_flag;
        return out;
    };

    // LSDA: Casida TDA A + full MO-product ALDA f^xc on the KS grid
    // (Casida 1995; Dreuw & Head-Gordon 2005 DOI 10.1021/cr0505627).
    // A_ia,jb += 2 (ia|f_xc|jb) (singlet) or (ia|f_xc|jb) (triplet), with
    // (ia|f_xc|jb) = ∫ φ_i φ_a f_ALDA(n) φ_j φ_b dτ from Dirac-exchange
    // second derivative on the molecular quadrature.
    if (xc == XCType::LSDA && hessian.n_ov() <= 512) {
        RealMatrix A = 0.5 * (hessian.dense_plus(ch) + hessian.dense_minus(ch));
        const MolecularGrid grid(pack.atoms.data(),
                                 static_cast<int>(pack.atoms.size()),
                                 grid_level, n_radial, n_angular);
        const RealMatrix K =
            internal::alda_kernel_matrix(det, pack, grid, channel);
        const double coulomb_spin =
            (channel == ExcitationChannel::singlet) ? 2.0 : 1.0;
        const std::size_t n = A.shape()[0];
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = 0; q < n; ++q)
                A(p, q) += coulomb_spin * K(p, q);
        return package_dense_tda(A, /*alda_flag=*/true);
    }

    // Global hybrid (B3LYP / PBE0): Casida TDA with exact-exchange fraction
    // α_hf on A plus the semi-local f^xc kernel from a central finite
    // difference of V_xc on the KS grid (Dreuw & Head-Gordon 2005;
    // matches PySCF tdscf.TDA / get_ab hybrid path).
    //
    //   ¹A = δε + 2(ia|jb) − α_hf (ij|ab) + A_xc
    const double alpha_hf = hybrid_exchange_fraction(xc);
    if (alpha_hf > 0.0 && hessian.n_ov() <= 512
        && channel == ExcitationChannel::singlet) {
        RealMatrix A = hessian.dense_A(ch, alpha_hf);
        const RealMatrix K =
            fxc_kernel_fd(det, pack, xc, grid_level, n_radial, n_angular);
        const std::size_t n = A.shape()[0];
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = 0; q < n; ++q)
                A(p, q) += K(p, q);
        return package_dense_tda(A, /*alda_flag=*/false);
    }

    // Pure GGA (PBE) and other residual paths: CIS@RKS (HF-like A on KS
    // orbitals; no semi-local f^xc) — hybrid/ALDA axes are above.
    // Full LR-TDDFT for pure GGA is RKS::tddft().
    const internal::TDHFPack tdhf_pack{hessian, ch};
    auto out =
        excited_states_result(det, pack, tdhf_pack.tda(n_states), channel,
                              /*tda_flag=*/true);
    out.alda = false;
    return out;
}

ExcitedStatesResult RKS::tddft(int n_states, ExcitationChannel channel) const {
    if (!has_result())
        throw std::runtime_error(
            "RKS::tddft: no SCF result — call run(frame) first");
    const SlaterDeterminant& det = determinant();
    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();
    const internal::OrbitalHessian hessian{det, eri};
    const internal::Channel ch = internal_channel(channel);

    if (n_states < 1 || n_states > hessian.n_ov())
        throw std::invalid_argument(
            "RKS::tddft: n_states = " + std::to_string(n_states)
            + " outside [1, n_ov = " + std::to_string(hessian.n_ov()) + "]");
    if (hessian.n_ov() > 512)
        throw std::runtime_error(
            "RKS::tddft: n_ov = " + std::to_string(hessian.n_ov())
            + " exceeds dense threshold 512 (no fallback)");

    // Casida A,B with exact-exchange fraction α_hf; pure functionals use α=0.
    const double alpha_hf = hybrid_exchange_fraction(xc);
    RealMatrix A = hessian.dense_A(ch, alpha_hf);
    RealMatrix B = hessian.dense_B(ch, alpha_hf);
    bool alda_flag = false;

    // Adiabatic f^xc on both A and B (Casida 1995; Dreuw & Head-Gordon 2005).
    if (xc == XCType::LSDA) {
        const MolecularGrid grid(pack.atoms.data(),
                                 static_cast<int>(pack.atoms.size()),
                                 grid_level, n_radial, n_angular);
        const RealMatrix K =
            internal::alda_kernel_matrix(det, pack, grid, channel);
        const double coulomb_spin =
            (channel == ExcitationChannel::singlet) ? 2.0 : 1.0;
        const std::size_t n = A.shape()[0];
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = 0; q < n; ++q) {
                const double v = coulomb_spin * K(p, q);
                A(p, q) += v;
                B(p, q) += v;
            }
        alda_flag = true;
    } else if (channel == ExcitationChannel::singlet) {
        // Pure GGA (PBE/BLYP) and global hybrids: FD of V_xc on the KS grid.
        const RealMatrix K =
            fxc_kernel_fd(det, pack, xc, grid_level, n_radial, n_angular);
        const std::size_t n = A.shape()[0];
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = 0; q < n; ++q) {
                A(p, q) += K(p, q);
                B(p, q) += K(p, q);
            }
    }
    // Triplet non-LSDA: Coulomb/exchange blocks only (no FD kernel yet).

    const internal::TDHFPack::Roots roots = rpa_from_ab(A, B, n_states);
    auto out =
        excited_states_result(det, pack, roots, channel, /*tda_flag=*/false);
    out.alda = alda_flag;
    return out;
}

std::vector<double> RHF::tda_forces(cpu::Frame& frame, int state, double h,
                                    bool analytic) const {
    auto make_rhf = [this]() {
        RHF r;
        r.basis = this->basis;
        r.tol = this->tol;
        r.max_iter = this->max_iter;
        r.charge = this->charge;
        return r;
    };

    if (analytic) {
        RHF probe = make_rhf();
        const bool prev = probe.compute_eri;
        probe.compute_eri = true;
        probe.run(frame);
        probe.compute_eri = prev;
        if (!probe.converged())
            throw std::runtime_error("RHF::tda_forces: SCF did not converge");
        auto es0 = probe.tda(state + 1);
        if (state < 0 || state >= static_cast<int>(es0.energy_ev.size()))
            throw std::invalid_argument("RHF::tda_forces: state out of range");
        return internal::tda_analytic_forces(frame, probe.integral_view_(),
                                             probe.determinant(),
                                             probe.energy(), state);
    }

    if (!(h > 0.0))
        throw std::invalid_argument("RHF::tda_forces: h must be > 0");
    const int n = frame.nlocal();
    {
        RHF probe = make_rhf();
        probe.run(frame);
        if (!probe.converged())
            throw std::runtime_error("RHF::tda_forces: SCF did not converge");
        auto es0 = probe.tda(state + 1);
        if (state < 0 || state >= static_cast<int>(es0.energy_ev.size()))
            throw std::invalid_argument("RHF::tda_forces: state out of range");
    }

    std::vector<double> F(static_cast<std::size_t>(3 * n), 0.0);
    StorageReal* coords[3] = {frame.span(kAtom_x), frame.span(kAtom_y),
                              frame.span(kAtom_z)};
    for (int a = 0; a < n; ++a) {
        for (int d = 0; d < 3; ++d) {
            const double x0 = coords[d][a];
            coords[d][a] = x0 + h;
            RHF rp = make_rhf();
            rp.run(frame);
            const double Ep =
                rp.tda(state + 1).energy_ev[static_cast<std::size_t>(state)];
            coords[d][a] = x0 - h;
            RHF rm = make_rhf();
            rm.run(frame);
            const double Em =
                rm.tda(state + 1).energy_ev[static_cast<std::size_t>(state)];
            coords[d][a] = x0;
            F[static_cast<std::size_t>(3 * a + d)] = -(Ep - Em) / (2.0 * h);
        }
    }
    RHF r0 = make_rhf();
    r0.run(frame);
    return F;
}

std::vector<double> RHF::tda_nac_vectors(cpu::Frame& frame, int state_i,
                                         int state_j) const {
    if (state_i == state_j)
        throw std::invalid_argument(
            "RHF::tda_nac_vectors: i and j must differ (no fallback)");
    // d_ij = (F_i − F_j) / (E_j − E_i)  [1/Å] (force eV/Å over energy eV).
    // Antisymmetry: d_ji = −d_ij exactly by construction (same F, E).
    // Phase: only when state_i < state_j, flip so the first large component
    // is non-negative; for state_i > state_j return −d_ji without re-phasing.
    const bool swap = state_i > state_j;
    const int lo = swap ? state_j : state_i;
    const int hi = swap ? state_i : state_j;
    // Use FD forces for the vector construction so the gap factor and force
    // difference share one consistent oracle (analytic CIS 2-RDM path is
    // still calibrating against FD on H2).
    auto Flo = tda_forces(frame, lo, 2e-3, /*analytic=*/false);
    auto Fhi = tda_forces(frame, hi, 2e-3, /*analytic=*/false);
    RHF r;
    r.basis = basis;
    r.tol = tol;
    r.max_iter = max_iter;
    r.charge = charge;
    r.run(frame);
    if (!r.converged())
        throw std::runtime_error("RHF::tda_nac_vectors: SCF did not converge");
    auto es = r.tda(hi + 1);
    const double Elo = es.energy_ev[static_cast<std::size_t>(lo)];
    const double Ehi = es.energy_ev[static_cast<std::size_t>(hi)];
    const double dE = Ehi - Elo;
    if (!(std::fabs(dE) > 1e-12))
        throw std::runtime_error("RHF::tda_nac_vectors: degenerate roots");
    std::vector<double> d(Flo.size());
    for (std::size_t k = 0; k < Flo.size(); ++k)
        d[k] = (Flo[k] - Fhi[k]) / dE; // d_lo,hi
    // Phase fix on the ordered pair only.
    for (double& v : d) {
        if (std::fabs(v) > 1e-8) {
            if (v < 0.0) {
                for (double& w : d)
                    w = -w;
            }
            break;
        }
    }
    if (swap) {
        for (double& w : d)
            w = -w; // d_hi,lo = −d_lo,hi
    }
    return d;
}

double RHF::tda_nac_strength(cpu::Frame& frame, int state_i, int state_j,
                             double /*h*/) const {
    auto d = tda_nac_vectors(frame, state_i, state_j);
    double n2 = 0.0;
    for (double v : d)
        n2 += v * v;
    return std::sqrt(n2);
}

ExcitedStatesResult RHF::tdhf(int n_states, ExcitationChannel channel) const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::tdhf: no SCF result — call run(frame) first");
    const SlaterDeterminant& det = determinant();
    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();
    const internal::OrbitalHessian hessian{det, eri};
    const internal::Channel ch = internal_channel(channel);

    // Link-08 stability pre-check: an unstable (A + B) block makes the full
    // RPA ω imaginary in this channel — fail loudly before solving.
    const internal::StabilityAnalysis analysis{hessian};
    const double lowest = analysis.lowest(ch);
    if (lowest < -1e-8)
        throw std::runtime_error(
            std::string("RHF::tdhf: the ")
            + (channel == ExcitationChannel::triplet ? "triplet" : "singlet")
            + " (A + B) block of the reference is unstable (lowest "
              "eigenvalue = "
            + std::to_string(lowest)
            + " Ha) — full RPA would give imaginary excitation energies; "
              "use tda() or reconverge the reference along the instability "
              "mode (stability().rotated_C) first (no fallback)");

    const internal::TDHFPack tdhf_pack{hessian, ch};
    return excited_states_result(det, pack, tdhf_pack.rpa(n_states), channel,
                                 /*tda_flag=*/false);
}

// ── UKS open-shell spin-conserving TDA ───────────────────────────────────

namespace {

/// @brief Four-index MO transform: out[p,q,r,s] = Σ C0(μ,p) C1(ν,q) C2(λ,r)
///        C3(σ,s) T[μνλσ] (chemist order). Same reshape/dot pattern as
///        OrbitalHessian::transform_block_.
xt::xtensor<double, 4> mo_transform_eri(const xt::xtensor<double, 4>& eri_ao,
                                        const RealMatrix& C0,
                                        const RealMatrix& C1,
                                        const RealMatrix& C2,
                                        const RealMatrix& C3) {
    xt::xtensor<double, 4> cur = eri_ao;
    const RealMatrix* Cs[4] = {&C3, &C2, &C1, &C0};
    for (int round = 0; round < 4; ++round) {
        const auto sh = cur.shape();
        auto flat = xt::reshape_view(cur, {sh[0] * sh[1] * sh[2], sh[3]});
        xt::xtensor<double, 2> h = xt::linalg::dot(flat, *Cs[round]);
        xt::xtensor<double, 4> next =
            xt::reshape_view(h, {sh[0], sh[1], sh[2], Cs[round]->shape()[1]});
        cur = xt::transpose(next, {3, 0, 1, 2});
    }
    return cur;
}

/// @brief Occupied / virtual column indices for integer occupations on one
///        spin channel (f = 1 ⇔ occupied, f = 0 ⇔ virtual).
void split_occ_virt(const RealVector& f, std::vector<int>& occ,
                    std::vector<int>& virt) {
    occ.clear();
    virt.clear();
    const int n_mo = static_cast<int>(f.size());
    for (int p = 0; p < n_mo; ++p) {
        const double fp = f(static_cast<std::size_t>(p));
        if (fp == 1.0)
            occ.push_back(p);
        else if (fp == 0.0)
            virt.push_back(p);
        else
            throw std::invalid_argument(
                "UKS::tda: fractional occupation f[" + std::to_string(p)
                + "] = " + std::to_string(fp)
                + " — open-shell TDA requires integer occupations");
    }
}

/// @brief Extract occupied / virtual coefficient blocks from C.
void coeff_blocks(const RealMatrix& C, const std::vector<int>& occ,
                  const std::vector<int>& virt, RealMatrix& Co,
                  RealMatrix& Cv) {
    const std::size_t n_ao = C.shape()[0];
    const int no = static_cast<int>(occ.size());
    const int nv = static_cast<int>(virt.size());
    Co = xt::zeros<double>({n_ao, static_cast<std::size_t>(no)});
    Cv = xt::zeros<double>({n_ao, static_cast<std::size_t>(nv)});
    for (int i = 0; i < no; ++i)
        for (std::size_t mu = 0; mu < n_ao; ++mu)
            Co(mu, static_cast<std::size_t>(i)) = C(
                mu, static_cast<std::size_t>(occ[static_cast<std::size_t>(i)]));
    for (int a = 0; a < nv; ++a)
        for (std::size_t mu = 0; mu < n_ao; ++mu)
            Cv(mu, static_cast<std::size_t>(a)) =
                C(mu,
                  static_cast<std::size_t>(virt[static_cast<std::size_t>(a)]));
}

/// @brief Spin-resolved ALDA \(f^{\mathrm{xc}}\) second derivatives of LSDA
///        (Slater + VWN5) at one density via central FD of \(v_{\mathrm{xc}}\).
/// @param ra Alpha density [a.u.⁻³].
/// @param rb Beta density [a.u.⁻³].
/// @param[out] faa ∂v_α/∂ρ_α [Hartree·Bohr³].
/// @param[out] fab ∂v_α/∂ρ_β [Hartree·Bohr³].
/// @param[out] fbb ∂v_β/∂ρ_β [Hartree·Bohr³].
void lsda_fxc_fd(double ra, double rb, double& faa, double& fab, double& fbb) {
    // Density floor and FD step chosen so ra±h and rb±h stay positive
    // (pow(ρ,…) in Slater/VWN5 NaNs on negative samples).
    constexpr double kFloor = 1e-12;
    constexpr double kH = 1e-6;
    if (!(ra > kFloor) && !(rb > kFloor)) {
        faa = fab = fbb = 0.0;
        return;
    }
    ra = std::max(ra, kFloor);
    rb = std::max(rb, kFloor);
    const double h_a = std::min(kH, 0.1 * ra);
    const double h_b = std::min(kH, 0.1 * rb);
    const XCOutput xp = evaluate_xc(XCType::LSDA, ra + h_a, rb);
    const XCOutput xm = evaluate_xc(XCType::LSDA, ra - h_a, rb);
    const XCOutput yp = evaluate_xc(XCType::LSDA, ra, rb + h_b);
    const XCOutput ym = evaluate_xc(XCType::LSDA, ra, rb - h_b);
    faa = (xp.vrho_alpha - xm.vrho_alpha) / (2.0 * h_a);
    fab = (yp.vrho_alpha - ym.vrho_alpha) / (2.0 * h_b);
    fbb = (yp.vrho_beta - ym.vrho_beta) / (2.0 * h_b);
    if (!std::isfinite(faa))
        faa = 0.0;
    if (!std::isfinite(fab))
        fab = 0.0;
    if (!std::isfinite(fbb))
        fbb = 0.0;
}

/// Workspace for open-shell LSDA Casida A (δε + Coulomb + spin-resolved fxc).
struct UksLsdaCasidaWork {
    RealMatrix A;
    int noa = 0, nva = 0, nob = 0, nvb = 0, na = 0, nb = 0, nov = 0;
    std::vector<int> occ_a, virt_a, occ_b, virt_b;
    /// Orbital-energy gap on each singles diagonal [Ha]; A − B = diag(de)
    /// for pure DFT / real MOs.
    std::vector<double> de;
};

/// @brief Build spin-conserving Casida A for UKS/LSDA (pure, α_hf = 0).
UksLsdaCasidaWork build_uks_lsda_casida_A(const SlaterDeterminant& det,
                                          const internal::IntegralView& pack,
                                          GridLevel grid_level, int n_radial,
                                          int n_angular, int n_states,
                                          const char* tag) {
    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& Ca = det.coefficients(Spin::alpha);
    const RealMatrix& Cb = det.coefficients(Spin::beta);
    const RealVector& ea = det.orbital_energies(Spin::alpha);
    const RealVector& eb = det.orbital_energies(Spin::beta);
    const RealVector& fa = det.occupations(Spin::alpha);
    const RealVector& fb = det.occupations(Spin::beta);

    UksLsdaCasidaWork w;
    split_occ_virt(fa, w.occ_a, w.virt_a);
    split_occ_virt(fb, w.occ_b, w.virt_b);
    w.noa = static_cast<int>(w.occ_a.size());
    w.nva = static_cast<int>(w.virt_a.size());
    w.nob = static_cast<int>(w.occ_b.size());
    w.nvb = static_cast<int>(w.virt_b.size());
    if (w.noa == 0 || w.nob == 0)
        throw std::runtime_error(std::string(tag)
                                 + ": empty occupied manifold on one spin "
                                   "(no fallback)");
    if (w.nva == 0 && w.nvb == 0)
        throw std::runtime_error(std::string(tag)
                                 + ": no virtual orbitals (no occ→virt "
                                   "singles)");
    w.na = w.noa * w.nva;
    w.nb = w.nob * w.nvb;
    w.nov = w.na + w.nb;
    if (n_states < 1 || n_states > w.nov)
        throw std::invalid_argument(
            std::string(tag) + ": n_states = " + std::to_string(n_states)
            + " outside [1, n_ov = " + std::to_string(w.nov) + "]");
    if (w.nov > 512)
        throw std::runtime_error(
            std::string(tag) + ": n_ov = " + std::to_string(w.nov)
            + " exceeds dense threshold 512 (no fallback)");

    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    RealMatrix Coa, Cva, Cob, Cvb;
    coeff_blocks(Ca, w.occ_a, w.virt_a, Coa, Cva);
    coeff_blocks(Cb, w.occ_b, w.virt_b, Cob, Cvb);

    const xt::xtensor<double, 4> iajb_aa =
        (w.na > 0) ? mo_transform_eri(eri, Coa, Cva, Coa, Cva)
                   : xt::zeros<double>({0, 0, 0, 0});
    const xt::xtensor<double, 4> iajb_bb =
        (w.nb > 0) ? mo_transform_eri(eri, Cob, Cvb, Cob, Cvb)
                   : xt::zeros<double>({0, 0, 0, 0});
    const xt::xtensor<double, 4> iajb_ab =
        (w.na > 0 && w.nb > 0) ? mo_transform_eri(eri, Coa, Cva, Cob, Cvb)
                               : xt::zeros<double>({0, 0, 0, 0});

    w.A = xt::zeros<double>(
        {static_cast<std::size_t>(w.nov), static_cast<std::size_t>(w.nov)});
    w.de.assign(static_cast<std::size_t>(w.nov), 0.0);

    for (int i = 0; i < w.noa; ++i)
        for (int a = 0; a < w.nva; ++a) {
            const int p = i * w.nva + a;
            const int ii = w.occ_a[static_cast<std::size_t>(i)];
            const int aa = w.virt_a[static_cast<std::size_t>(a)];
            const double gap = ea(static_cast<std::size_t>(aa))
                               - ea(static_cast<std::size_t>(ii));
            w.de[static_cast<std::size_t>(p)] = gap;
            w.A(static_cast<std::size_t>(p), static_cast<std::size_t>(p)) = gap;
        }
    for (int j = 0; j < w.nob; ++j)
        for (int b = 0; b < w.nvb; ++b) {
            const int q = w.na + j * w.nvb + b;
            const int jj = w.occ_b[static_cast<std::size_t>(j)];
            const int bb = w.virt_b[static_cast<std::size_t>(b)];
            const double gap = eb(static_cast<std::size_t>(bb))
                               - eb(static_cast<std::size_t>(jj));
            w.de[static_cast<std::size_t>(q)] = gap;
            w.A(static_cast<std::size_t>(q), static_cast<std::size_t>(q)) = gap;
        }

    for (int i = 0; i < w.noa; ++i)
        for (int a = 0; a < w.nva; ++a) {
            const int p = i * w.nva + a;
            for (int j = 0; j < w.noa; ++j)
                for (int b = 0; b < w.nva; ++b) {
                    const int q = j * w.nva + b;
                    w.A(static_cast<std::size_t>(p),
                        static_cast<std::size_t>(q)) +=
                        iajb_aa(static_cast<std::size_t>(i),
                                static_cast<std::size_t>(a),
                                static_cast<std::size_t>(j),
                                static_cast<std::size_t>(b));
                }
            for (int j = 0; j < w.nob; ++j)
                for (int b = 0; b < w.nvb; ++b) {
                    const int q = w.na + j * w.nvb + b;
                    const double v = iajb_ab(static_cast<std::size_t>(i),
                                             static_cast<std::size_t>(a),
                                             static_cast<std::size_t>(j),
                                             static_cast<std::size_t>(b));
                    w.A(static_cast<std::size_t>(p),
                        static_cast<std::size_t>(q)) += v;
                    w.A(static_cast<std::size_t>(q),
                        static_cast<std::size_t>(p)) += v;
                }
        }
    for (int i = 0; i < w.nob; ++i)
        for (int a = 0; a < w.nvb; ++a) {
            const int p = w.na + i * w.nvb + a;
            for (int j = 0; j < w.nob; ++j)
                for (int b = 0; b < w.nvb; ++b) {
                    const int q = w.na + j * w.nvb + b;
                    w.A(static_cast<std::size_t>(p),
                        static_cast<std::size_t>(q)) +=
                        iajb_bb(static_cast<std::size_t>(i),
                                static_cast<std::size_t>(a),
                                static_cast<std::size_t>(j),
                                static_cast<std::size_t>(b));
                }
        }

    if (pack.shells.empty())
        throw std::runtime_error(
            std::string(tag)
            + ": internal::IntegralWorkspace has no shells (no fallback)");
    const MolecularGrid grid(pack.atoms.data(),
                             static_cast<int>(pack.atoms.size()), grid_level,
                             n_radial, n_angular);

    RealMatrix Pa = det.density(Spin::alpha);
    RealMatrix Pb = det.density(Spin::beta);
    RealMatrix Ca_work = Ca;
    RealMatrix Cb_work = Cb;
    if (pack.spherical()) {
        const auto& c2s = pack.c2s;
        Pa = xt::linalg::dot(
            c2s, RealMatrix(xt::linalg::dot(Pa, xt::transpose(c2s))));
        Pb = xt::linalg::dot(
            c2s, RealMatrix(xt::linalg::dot(Pb, xt::transpose(c2s))));
        Ca_work = xt::linalg::dot(c2s, Ca);
        Cb_work = xt::linalg::dot(c2s, Cb);
        coeff_blocks(Ca_work, w.occ_a, w.virt_a, Coa, Cva);
        coeff_blocks(Cb_work, w.occ_b, w.virt_b, Cob, Cvb);
    }
    const int n_basis = static_cast<int>(Pa.shape()[0]);
    const int G = static_cast<int>(grid.points.size());

    std::vector<double> chi(static_cast<std::size_t>(n_basis));
    std::vector<double> xi_a(static_cast<std::size_t>(std::max(w.na, 1)));
    std::vector<double> xi_b(static_cast<std::size_t>(std::max(w.nb, 1)));
    std::vector<double> phi_oa(static_cast<std::size_t>(std::max(w.noa, 1)));
    std::vector<double> phi_va(static_cast<std::size_t>(std::max(w.nva, 1)));
    std::vector<double> phi_ob(static_cast<std::size_t>(std::max(w.nob, 1)));
    std::vector<double> phi_vb(static_cast<std::size_t>(std::max(w.nvb, 1)));

    for (int g = 0; g < G; ++g) {
        const auto& pt = grid.points[static_cast<std::size_t>(g)];
        const double weight = pt.weight;
        if (!(std::fabs(weight) > 0.0))
            continue;
        for (int mu = 0; mu < n_basis; ++mu)
            chi[static_cast<std::size_t>(mu)] = eval_basis(
                pack.shells[static_cast<std::size_t>(mu)], pt.x, pt.y, pt.z);

        auto dens = [&](const RealMatrix& P) {
            double rho = 0.0;
            for (int mu = 0; mu < n_basis; ++mu) {
                const double cmu = chi[static_cast<std::size_t>(mu)];
                if (!(std::fabs(cmu) > 1e-20))
                    continue;
                double row = 0.0;
                for (int nu = 0; nu < n_basis; ++nu)
                    row += P(mu, nu) * chi[static_cast<std::size_t>(nu)];
                rho += cmu * row;
            }
            return rho < 0.0 ? 0.0 : rho;
        };
        const double ra = dens(Pa);
        const double rb = dens(Pb);
        double faa = 0.0, fab = 0.0, fbb = 0.0;
        lsda_fxc_fd(ra, rb, faa, fab, fbb);
        if (!(std::fabs(faa) > 0.0 || std::fabs(fab) > 0.0
              || std::fabs(fbb) > 0.0))
            continue;

        for (int i = 0; i < w.noa; ++i) {
            double v = 0.0;
            for (int mu = 0; mu < n_basis; ++mu)
                v += chi[static_cast<std::size_t>(mu)]
                     * Coa(mu, static_cast<std::size_t>(i));
            phi_oa[static_cast<std::size_t>(i)] = v;
        }
        for (int a = 0; a < w.nva; ++a) {
            double v = 0.0;
            for (int mu = 0; mu < n_basis; ++mu)
                v += chi[static_cast<std::size_t>(mu)]
                     * Cva(mu, static_cast<std::size_t>(a));
            phi_va[static_cast<std::size_t>(a)] = v;
        }
        for (int i = 0; i < w.nob; ++i) {
            double v = 0.0;
            for (int mu = 0; mu < n_basis; ++mu)
                v += chi[static_cast<std::size_t>(mu)]
                     * Cob(mu, static_cast<std::size_t>(i));
            phi_ob[static_cast<std::size_t>(i)] = v;
        }
        for (int a = 0; a < w.nvb; ++a) {
            double v = 0.0;
            for (int mu = 0; mu < n_basis; ++mu)
                v += chi[static_cast<std::size_t>(mu)]
                     * Cvb(mu, static_cast<std::size_t>(a));
            phi_vb[static_cast<std::size_t>(a)] = v;
        }
        for (int i = 0; i < w.noa; ++i)
            for (int a = 0; a < w.nva; ++a)
                xi_a[static_cast<std::size_t>(i * w.nva + a)] =
                    phi_oa[static_cast<std::size_t>(i)]
                    * phi_va[static_cast<std::size_t>(a)];
        for (int i = 0; i < w.nob; ++i)
            for (int a = 0; a < w.nvb; ++a)
                xi_b[static_cast<std::size_t>(i * w.nvb + a)] =
                    phi_ob[static_cast<std::size_t>(i)]
                    * phi_vb[static_cast<std::size_t>(a)];

        const double wfaa = weight * faa;
        const double wfab = weight * fab;
        const double wfbb = weight * fbb;
        for (int p = 0; p < w.na; ++p) {
            const double xp = xi_a[static_cast<std::size_t>(p)];
            if (!(std::fabs(xp) > 0.0))
                continue;
            for (int q = p; q < w.na; ++q) {
                const double v = wfaa * xp * xi_a[static_cast<std::size_t>(q)];
                w.A(static_cast<std::size_t>(p), static_cast<std::size_t>(q)) +=
                    v;
                if (q != p)
                    w.A(static_cast<std::size_t>(q),
                        static_cast<std::size_t>(p)) += v;
            }
            for (int q = 0; q < w.nb; ++q) {
                const double v = wfab * xp * xi_b[static_cast<std::size_t>(q)];
                w.A(static_cast<std::size_t>(p),
                    static_cast<std::size_t>(w.na + q)) += v;
                w.A(static_cast<std::size_t>(w.na + q),
                    static_cast<std::size_t>(p)) += v;
            }
        }
        for (int p = 0; p < w.nb; ++p) {
            const double xp = xi_b[static_cast<std::size_t>(p)];
            if (!(std::fabs(xp) > 0.0))
                continue;
            for (int q = p; q < w.nb; ++q) {
                const double v = wfbb * xp * xi_b[static_cast<std::size_t>(q)];
                w.A(static_cast<std::size_t>(w.na + p),
                    static_cast<std::size_t>(w.na + q)) += v;
                if (q != p)
                    w.A(static_cast<std::size_t>(w.na + q),
                        static_cast<std::size_t>(w.na + p)) += v;
            }
        }
    }

    for (int p = 0; p < w.nov; ++p)
        for (int q = p + 1; q < w.nov; ++q) {
            const double s =
                0.5
                * (w.A(static_cast<std::size_t>(p), static_cast<std::size_t>(q))
                   + w.A(static_cast<std::size_t>(q),
                         static_cast<std::size_t>(p)));
            w.A(static_cast<std::size_t>(p), static_cast<std::size_t>(q)) = s;
            w.A(static_cast<std::size_t>(q), static_cast<std::size_t>(p)) = s;
        }
    return w;
}

/// Package open-shell (X,Y) into ExcitedStatesResult with length-gauge μ, f.
/// Full RPA uses X+Y; TDA uses X only (Y = 0). No √2 closed-shell factor.
ExcitedStatesResult package_uks_excited(const SlaterDeterminant& det,
                                        const internal::IntegralView& pack,
                                        const UksLsdaCasidaWork& w,
                                        const internal::TDHFPack::Roots& roots,
                                        bool tda_flag) {
    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& Ca = det.coefficients(Spin::alpha);
    const RealMatrix& Cb = det.coefficients(Spin::beta);
    RealMatrix Coa, Cva, Cob, Cvb;
    coeff_blocks(Ca, w.occ_a, w.virt_a, Coa, Cva);
    coeff_blocks(Cb, w.occ_b, w.virt_b, Cob, Cvb);

    const std::size_t k = roots.omega.size();
    ExcitedStatesResult out;
    out.channel = ExcitationChannel::singlet; // no closed-shell S/T split
    out.tda = tda_flag;
    out.alda = true;
    out.energy_ev.resize(k);
    out.oscillator_strength.assign(k, 0.0);
    out.transition_dipole_au = xt::zeros<double>({k, std::size_t(3)});
    for (std::size_t s = 0; s < k; ++s)
        out.energy_ev[s] = roots.omega(s) * units::kHartreeInEv;

    const double origin[3] = {0.0, 0.0, 0.0};
    const MultipoleIntegrals mi{pack.shells.data(),
                                static_cast<int>(pack.shells.size()), origin};
    for (int axis = 0; axis < 3; ++axis) {
        xt::xtensor<double, 2> D = mi.dipole(axis);
        if (pack.spherical()) {
            const auto& c2s = pack.c2s;
            D = xt::linalg::dot(
                xt::transpose(c2s),
                xt::xtensor<double, 2>(xt::linalg::dot(D, c2s)));
        }
        RealMatrix d_mo_a = xt::zeros<double>(
            {static_cast<std::size_t>(w.noa), static_cast<std::size_t>(w.nva)});
        RealMatrix d_mo_b = xt::zeros<double>(
            {static_cast<std::size_t>(w.nob), static_cast<std::size_t>(w.nvb)});
        if (w.na > 0)
            d_mo_a = xt::linalg::dot(xt::transpose(Coa),
                                     RealMatrix(xt::linalg::dot(D, Cva)));
        if (w.nb > 0)
            d_mo_b = xt::linalg::dot(xt::transpose(Cob),
                                     RealMatrix(xt::linalg::dot(D, Cvb)));
        for (std::size_t s = 0; s < k; ++s) {
            double mu_s = 0.0;
            for (int i = 0; i < w.noa; ++i)
                for (int a = 0; a < w.nva; ++a) {
                    const int p = i * w.nva + a;
                    const double xy = roots.X(s, static_cast<std::size_t>(p))
                                      + roots.Y(s, static_cast<std::size_t>(p));
                    mu_s += xy
                            * d_mo_a(static_cast<std::size_t>(i),
                                     static_cast<std::size_t>(a));
                }
            for (int i = 0; i < w.nob; ++i)
                for (int a = 0; a < w.nvb; ++a) {
                    const int p = w.na + i * w.nvb + a;
                    const double xy = roots.X(s, static_cast<std::size_t>(p))
                                      + roots.Y(s, static_cast<std::size_t>(p));
                    mu_s += xy
                            * d_mo_b(static_cast<std::size_t>(i),
                                     static_cast<std::size_t>(a));
                }
            out.transition_dipole_au(s, axis) = mu_s;
        }
    }
    for (std::size_t s = 0; s < k; ++s) {
        const double mu2 =
            out.transition_dipole_au(s, 0) * out.transition_dipole_au(s, 0)
            + out.transition_dipole_au(s, 1) * out.transition_dipole_au(s, 1)
            + out.transition_dipole_au(s, 2) * out.transition_dipole_au(s, 2);
        out.oscillator_strength[s] = (2.0 / 3.0) * roots.omega(s) * mu2;
    }
    return out;
}

} // namespace

ExcitedStatesResult UKS::tda(int n_states) const {
    if (!has_result())
        throw std::runtime_error(
            "UKS::tda: no SCF result — call run(frame) first");
    const SlaterDeterminant& det = determinant();
    if (det.restricted() || det.generalized())
        throw std::runtime_error(
            "UKS::tda: requires an unrestricted (UKS) determinant");
    if (xc != XCType::LSDA)
        throw std::runtime_error(
            "UKS::tda: only LSDA is implemented (pure open-shell ALDA); "
            "requested functional is not LSDA (no fallback)");

    const internal::IntegralView pack = integral_view_();
    const UksLsdaCasidaWork w = build_uks_lsda_casida_A(
        det, pack, grid_level, n_radial, n_angular, n_states, "UKS::tda");

    auto [evals, U] = xt::linalg::eigh(w.A);
    const std::size_t k = static_cast<std::size_t>(n_states);
    const std::size_t n = static_cast<std::size_t>(w.nov);
    internal::TDHFPack::Roots roots;
    roots.omega = xt::zeros<double>({k});
    roots.X = xt::zeros<double>({k, n});
    roots.Y = xt::zeros<double>({k, n});
    for (std::size_t s = 0; s < k; ++s) {
        roots.omega(s) = evals(s);
        for (std::size_t r = 0; r < n; ++r)
            roots.X(s, r) = U(r, s);
    }
    return package_uks_excited(det, pack, w, roots, /*tda_flag=*/true);
}

ExcitedStatesResult UKS::tddft(int n_states) const {
    if (!has_result())
        throw std::runtime_error(
            "UKS::tddft: no SCF result — call run(frame) first");
    const SlaterDeterminant& det = determinant();
    if (det.restricted() || det.generalized())
        throw std::runtime_error(
            "UKS::tddft: requires an unrestricted (UKS) determinant");
    if (xc != XCType::LSDA)
        throw std::runtime_error(
            "UKS::tddft: only LSDA is implemented (pure open-shell ALDA); "
            "requested functional is not LSDA (no fallback)");

    const internal::IntegralView pack = integral_view_();
    const UksLsdaCasidaWork w = build_uks_lsda_casida_A(
        det, pack, grid_level, n_radial, n_angular, n_states, "UKS::tddft");

    // Pure LSDA, real MOs: B = Coulomb + fxc = A − diag(ε_a − ε_i).
    RealMatrix B = w.A;
    for (int p = 0; p < w.nov; ++p)
        B(static_cast<std::size_t>(p), static_cast<std::size_t>(p)) -=
            w.de[static_cast<std::size_t>(p)];

    const internal::TDHFPack::Roots roots = rpa_from_ab(w.A, B, n_states);
    return package_uks_excited(det, pack, w, roots, /*tda_flag=*/false);
}

ExcitedStatesResult UHF::sf_tda(int n_states) const {
    if (!has_result())
        throw std::runtime_error(
            "UHF::sf_tda: no SCF result — call run(frame) first");
    const SlaterDeterminant& det = determinant();
    if (det.restricted() || det.generalized())
        throw std::runtime_error(
            "UHF::sf_tda: requires an unrestricted (UHF) determinant");

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& Ca = det.coefficients(Spin::alpha);
    const RealMatrix& Cb = det.coefficients(Spin::beta);
    const RealVector& ea = det.orbital_energies(Spin::alpha);
    const RealVector& eb = det.orbital_energies(Spin::beta);
    const RealVector& fa = det.occupations(Spin::alpha);
    const RealVector& fb = det.occupations(Spin::beta);

    std::vector<int> occ_a, virt_a, occ_b, virt_b;
    split_occ_virt(fa, occ_a, virt_a);
    split_occ_virt(fb, occ_b, virt_b);
    const int noa = static_cast<int>(occ_a.size());
    const int nva = static_cast<int>(virt_a.size());
    const int nob = static_cast<int>(occ_b.size());
    const int nvb = static_cast<int>(virt_b.size());

    // High-spin convention: flip majority → minority (Ms decreases for
    // n_alpha ≥ n_beta). Collinear SF-CIS space is one spin-flip block only.
    const bool alpha_to_beta = (noa >= nob);
    const int no = alpha_to_beta ? noa : nob;
    const int nv = alpha_to_beta ? nvb : nva;
    if (no == 0 || nv == 0)
        throw std::runtime_error(
            "UHF::sf_tda: empty SF manifold (need majority occupied and "
            "minority virtual orbitals — no fallback)");
    const int nov = no * nv;
    if (n_states < 1 || n_states > nov)
        throw std::invalid_argument(
            "UHF::sf_tda: n_states = " + std::to_string(n_states)
            + " outside [1, n_ov = " + std::to_string(nov) + "]");
    if (nov > 512)
        throw std::runtime_error(
            "UHF::sf_tda: n_ov = " + std::to_string(nov)
            + " exceeds dense threshold 512 (no fallback)");

    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    RealMatrix Co, Cv;
    if (alpha_to_beta) {
        RealMatrix dummy_v, dummy_o;
        coeff_blocks(Ca, occ_a, virt_a, Co, dummy_v);
        coeff_blocks(Cb, occ_b, virt_b, dummy_o, Cv);
    } else {
        RealMatrix dummy_v, dummy_o;
        coeff_blocks(Cb, occ_b, virt_b, Co, dummy_v);
        coeff_blocks(Ca, occ_a, virt_a, dummy_o, Cv);
    }

    // (ij|ab) with i,j majority-occ, a,b minority-virt — exchange block only.
    const xt::xtensor<double, 4> ijab = mo_transform_eri(eri, Co, Co, Cv, Cv);

    RealMatrix A = xt::zeros<double>(
        {static_cast<std::size_t>(nov), static_cast<std::size_t>(nov)});
    for (int i = 0; i < no; ++i) {
        for (int a = 0; a < nv; ++a) {
            const int p = i * nv + a;
            const int ii = alpha_to_beta ? occ_a[static_cast<std::size_t>(i)]
                                         : occ_b[static_cast<std::size_t>(i)];
            const int aa = alpha_to_beta ? virt_b[static_cast<std::size_t>(a)]
                                         : virt_a[static_cast<std::size_t>(a)];
            const double e_virt = alpha_to_beta
                                      ? eb(static_cast<std::size_t>(aa))
                                      : ea(static_cast<std::size_t>(aa));
            const double e_occ = alpha_to_beta
                                     ? ea(static_cast<std::size_t>(ii))
                                     : eb(static_cast<std::size_t>(ii));
            A(static_cast<std::size_t>(p), static_cast<std::size_t>(p)) =
                e_virt - e_occ;
            for (int j = 0; j < no; ++j) {
                for (int b = 0; b < nv; ++b) {
                    const int q = j * nv + b;
                    A(static_cast<std::size_t>(p),
                      static_cast<std::size_t>(q)) -=
                        ijab(static_cast<std::size_t>(i),
                             static_cast<std::size_t>(j),
                             static_cast<std::size_t>(a),
                             static_cast<std::size_t>(b));
                }
            }
        }
    }

    // Numerical symmetrize then dense Hermitian eigensolve.
    for (int p = 0; p < nov; ++p)
        for (int q = p + 1; q < nov; ++q) {
            const double s =
                0.5
                * (A(static_cast<std::size_t>(p), static_cast<std::size_t>(q))
                   + A(static_cast<std::size_t>(q),
                       static_cast<std::size_t>(p)));
            A(static_cast<std::size_t>(p), static_cast<std::size_t>(q)) = s;
            A(static_cast<std::size_t>(q), static_cast<std::size_t>(p)) = s;
        }

    auto [w, U] = xt::linalg::eigh(A);
    (void)U;
    const std::size_t k = static_cast<std::size_t>(n_states);
    ExcitedStatesResult out;
    // No closed-shell S/T split; SF roots are Ms-changing. Reuse `triplet` as
    // the spin-forbidden channel tag (μ = 0).
    out.channel = ExcitationChannel::triplet;
    out.tda = true;
    out.alda = false;
    out.energy_ev.resize(k);
    out.oscillator_strength.assign(k, 0.0);
    out.transition_dipole_au = xt::zeros<double>({k, std::size_t(3)});
    for (std::size_t s = 0; s < k; ++s)
        out.energy_ev[s] = w(s) * units::kHartreeInEv;
    return out;
}

namespace {

/// Dense non-symmetric eigenproblem via the shared `linalg::eig_nonsymmetric`.
/// Returns real eigenvalues with wr > 0 and |wi| ≤ imag_tol, sorted ascending.
std::vector<double> sf_rpa_positive_roots_(const RealMatrix& A,
                                           const RealMatrix& B,
                                           double imag_tol = 1e-8) {
    const int nov = static_cast<int>(A.shape()[0]);
    if (static_cast<int>(A.shape()[1]) != nov
        || static_cast<int>(B.shape()[0]) != nov
        || static_cast<int>(B.shape()[1]) != nov)
        throw std::invalid_argument(
            "UHF::sf_tddft: A/B must be square and equal size");
    // Non-Hermitian RPA: [[A, B], [-B, -A]] (X;Y) = ω (X;Y). Spectrum is
    // ± pairs when stable; collinear SF may leave (A−B) indefinite so the
    // Hermitian Casida reduction is not used. RealMatrix is column-major, so
    // the logical block layout below matches the old hand-rolled column-major
    // buffer bit-for-bit under dgeev.
    const std::size_t n = static_cast<std::size_t>(2 * nov);
    RealMatrix M = xt::zeros<double>({n, n});
    for (int i = 0; i < nov; ++i) {
        for (int j = 0; j < nov; ++j) {
            const double aij =
                A(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            const double bij =
                B(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            M(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = aij;
            M(static_cast<std::size_t>(i), static_cast<std::size_t>(j + nov)) =
                bij;
            M(static_cast<std::size_t>(i + nov), static_cast<std::size_t>(j)) =
                -bij;
            M(static_cast<std::size_t>(i + nov),
              static_cast<std::size_t>(j + nov)) = -aij;
        }
    }
    RealVector wr, wi;
    linalg::eig_nonsymmetric(M, wr, wi, nullptr, "UHF::sf_tddft");
    std::vector<double> pos;
    pos.reserve(static_cast<std::size_t>(nov));
    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(wi(i)) <= imag_tol && wr(i) > 1e-8)
            pos.push_back(wr(i));
    }
    std::sort(pos.begin(), pos.end());
    return pos;
}

} // namespace

ExcitedStatesResult UHF::sf_tddft(int n_states) const {
    if (!has_result())
        throw std::runtime_error(
            "UHF::sf_tddft: no SCF result — call run(frame) first");
    const SlaterDeterminant& det = determinant();
    if (det.restricted() || det.generalized())
        throw std::runtime_error(
            "UHF::sf_tddft: requires an unrestricted (UHF) determinant");

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& Ca = det.coefficients(Spin::alpha);
    const RealMatrix& Cb = det.coefficients(Spin::beta);
    const RealVector& ea = det.orbital_energies(Spin::alpha);
    const RealVector& eb = det.orbital_energies(Spin::beta);
    const RealVector& fa = det.occupations(Spin::alpha);
    const RealVector& fb = det.occupations(Spin::beta);

    std::vector<int> occ_a, virt_a, occ_b, virt_b;
    split_occ_virt(fa, occ_a, virt_a);
    split_occ_virt(fb, occ_b, virt_b);
    const int noa = static_cast<int>(occ_a.size());
    const int nva = static_cast<int>(virt_a.size());
    const int nob = static_cast<int>(occ_b.size());
    const int nvb = static_cast<int>(virt_b.size());

    const bool alpha_to_beta = (noa >= nob);
    const int no = alpha_to_beta ? noa : nob;
    const int nv = alpha_to_beta ? nvb : nva;
    if (no == 0 || nv == 0)
        throw std::runtime_error(
            "UHF::sf_tddft: empty SF manifold (need majority occupied and "
            "minority virtual orbitals — no fallback)");
    const int nov = no * nv;
    if (n_states < 1 || n_states > nov)
        throw std::invalid_argument(
            "UHF::sf_tddft: n_states = " + std::to_string(n_states)
            + " outside [1, n_ov = " + std::to_string(nov) + "]");
    // Doubled non-Hermitian space is 2*n_ov; cap so dgeev stays tiny.
    if (nov > 256)
        throw std::runtime_error(
            "UHF::sf_tddft: n_ov = " + std::to_string(nov)
            + " exceeds dense SF-RPA threshold 256 (no fallback)");

    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    RealMatrix Co, Cv;
    if (alpha_to_beta) {
        RealMatrix dummy_v, dummy_o;
        coeff_blocks(Ca, occ_a, virt_a, Co, dummy_v);
        coeff_blocks(Cb, occ_b, virt_b, dummy_o, Cv);
    } else {
        RealMatrix dummy_v, dummy_o;
        coeff_blocks(Cb, occ_b, virt_b, Co, dummy_v);
        coeff_blocks(Ca, occ_a, virt_a, dummy_o, Cv);
    }

    // A: exchange-only SF-CIS block (same as sf_tda).
    const xt::xtensor<double, 4> ijab = mo_transform_eri(eri, Co, Co, Cv, Cv);
    // B: exchange de-excitation (ib|aj) — Co,Cv,Cv,Co → [i,b,a,j].
    const xt::xtensor<double, 4> ibaj = mo_transform_eri(eri, Co, Cv, Cv, Co);

    RealMatrix A = xt::zeros<double>(
        {static_cast<std::size_t>(nov), static_cast<std::size_t>(nov)});
    RealMatrix B = xt::zeros<double>(
        {static_cast<std::size_t>(nov), static_cast<std::size_t>(nov)});
    for (int i = 0; i < no; ++i) {
        for (int a = 0; a < nv; ++a) {
            const int p = i * nv + a;
            const int ii = alpha_to_beta ? occ_a[static_cast<std::size_t>(i)]
                                         : occ_b[static_cast<std::size_t>(i)];
            const int aa = alpha_to_beta ? virt_b[static_cast<std::size_t>(a)]
                                         : virt_a[static_cast<std::size_t>(a)];
            const double e_virt = alpha_to_beta
                                      ? eb(static_cast<std::size_t>(aa))
                                      : ea(static_cast<std::size_t>(aa));
            const double e_occ = alpha_to_beta
                                     ? ea(static_cast<std::size_t>(ii))
                                     : eb(static_cast<std::size_t>(ii));
            A(static_cast<std::size_t>(p), static_cast<std::size_t>(p)) =
                e_virt - e_occ;
            for (int j = 0; j < no; ++j) {
                for (int b = 0; b < nv; ++b) {
                    const int q = j * nv + b;
                    A(static_cast<std::size_t>(p),
                      static_cast<std::size_t>(q)) -=
                        ijab(static_cast<std::size_t>(i),
                             static_cast<std::size_t>(j),
                             static_cast<std::size_t>(a),
                             static_cast<std::size_t>(b));
                    // B_ia,jb = -(ib|aj)
                    B(static_cast<std::size_t>(p),
                      static_cast<std::size_t>(q)) -=
                        ibaj(static_cast<std::size_t>(i),
                             static_cast<std::size_t>(b),
                             static_cast<std::size_t>(a),
                             static_cast<std::size_t>(j));
                }
            }
        }
    }

    for (int p = 0; p < nov; ++p)
        for (int q = p + 1; q < nov; ++q) {
            const double sa =
                0.5
                * (A(static_cast<std::size_t>(p), static_cast<std::size_t>(q))
                   + A(static_cast<std::size_t>(q),
                       static_cast<std::size_t>(p)));
            A(static_cast<std::size_t>(p), static_cast<std::size_t>(q)) = sa;
            A(static_cast<std::size_t>(q), static_cast<std::size_t>(p)) = sa;
            const double sb =
                0.5
                * (B(static_cast<std::size_t>(p), static_cast<std::size_t>(q))
                   + B(static_cast<std::size_t>(q),
                       static_cast<std::size_t>(p)));
            B(static_cast<std::size_t>(p), static_cast<std::size_t>(q)) = sb;
            B(static_cast<std::size_t>(q), static_cast<std::size_t>(p)) = sb;
        }

    const std::vector<double> pos = sf_rpa_positive_roots_(A, B);
    if (static_cast<int>(pos.size()) < n_states)
        throw std::runtime_error(
            "UHF::sf_tddft: only " + std::to_string(pos.size())
            + " positive real RPA roots (need " + std::to_string(n_states)
            + ") — possible SF-RPA instability or complex spectrum "
              "(no fallback)");

    const std::size_t k = static_cast<std::size_t>(n_states);
    ExcitedStatesResult out;
    out.channel = ExcitationChannel::triplet;
    out.tda = false;
    out.alda = false;
    out.energy_ev.resize(k);
    out.oscillator_strength.assign(k, 0.0);
    out.transition_dipole_au = xt::zeros<double>({k, std::size_t(3)});
    for (std::size_t s = 0; s < k; ++s)
        out.energy_ev[s] = pos[s] * units::kHartreeInEv;
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
